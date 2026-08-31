#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fake_flash.h"
#include "flash.h"

#define CHECK(condition)                                                      \
	do {                                                                  \
		if (!(condition)) {                                            \
			fprintf(stderr, "%s:%d: check failed: %s\n",            \
				__FILE__, __LINE__, #condition);                \
			exit(EXIT_FAILURE);                                   \
		}                                                             \
	} while (0)

struct fixture {
	struct flash_log log;
	struct flash_chunk scratch;
	struct fake_flash fake;
};

struct scan_progress_capture {
	uint32_t discover_calls;
	uint32_t cleanup_calls;
	uint32_t erase_calls;
	uint32_t discover_completed;
	uint32_t cleanup_completed;
	uint32_t total;
};

static void capture_scan_progress(void *context,
				  enum flash_log_scan_phase phase,
				  uint32_t completed_sectors,
				  uint32_t total_sectors)
{
	struct scan_progress_capture *capture = context;

	capture->total = total_sectors;
	if (phase == FLASH_LOG_SCAN_DISCOVER) {
		capture->discover_calls++;
		capture->discover_completed = completed_sectors;
	} else if (phase == FLASH_LOG_SCAN_CLEANUP) {
		capture->cleanup_calls++;
		capture->cleanup_completed = completed_sectors;
	} else {
		capture->erase_calls++;
	}
}

static void fixture_init(struct fixture *fixture, uint32_t sector_count)
{
	enum flash_log_result result;

	fake_flash_init(&fixture->fake, sector_count);
	result = flash_log_init(
		&fixture->log, sector_count, &fixture->scratch,
		&fake_flash_storage_ops, &fixture->fake,
		&fake_flash_transport_ops, &fixture->fake);
	CHECK(result == FLASH_LOG_OK);
}

static void assert_ring_state(const struct flash_log *log)
{
	CHECK(log->read_sector < log->sector_count);
	CHECK(log->write_sector < log->sector_count);
	CHECK(!(flash_log_is_empty(log) && flash_log_is_full(log)));
}

static void write_log(struct fixture *fixture, uint32_t log_id,
		      uint32_t chunk_count)
{
	uint32_t assigned_log_id;

	fixture->log.next_log_id = log_id;
	CHECK(flash_log_begin(&fixture->log, &assigned_log_id) ==
	      FLASH_LOG_OK);
	CHECK(assigned_log_id == log_id);

	for (uint32_t sequence = 0; sequence < chunk_count; sequence++) {
		uint8_t payload = (uint8_t)sequence;

		CHECK(flash_log_append(&fixture->log, &payload,
				       sizeof(payload)) == FLASH_LOG_OK);
	}
	CHECK(flash_log_close(&fixture->log) == FLASH_LOG_OK);
}

static void check_summary(const struct flash_transfer_summary *summary,
			  uint32_t log_id, uint32_t start_sector,
			  uint32_t end_sector, uint32_t sector_count)
{
	CHECK(summary->log_id == log_id);
	CHECK(summary->start_sector == start_sector);
	CHECK(summary->end_sector == end_sector);
	CHECK(summary->sector_count == sector_count);
	CHECK(summary->raw_crc != 0U);
}

static void test_empty_scan_cleans_dirty_media(void)
{
	struct fixture fixture;

	fixture_init(&fixture, 6);
	fake_flash_set_dirty(&fixture.fake, 3, 9, 0);

	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.fake.erase_count[3] == 1);
	CHECK(fixture.log.read_sector == 0);
	CHECK(fixture.log.write_sector == 0);
	CHECK(fixture.log.next_log_id == 0);
	CHECK(flash_log_is_empty(&fixture.log));
	CHECK(!flash_log_is_full(&fixture.log));
	assert_ring_state(&fixture.log);
}

static void test_scan_reports_discovery_and_cleanup_progress(void)
{
	struct scan_progress_capture capture = { 0 };
	struct fixture fixture;

	fixture_init(&fixture, 6);
	fake_flash_set_dirty(&fixture.fake, 3, 9, 0);

	CHECK(flash_log_scan_with_progress(
		      &fixture.log, capture_scan_progress, &capture) ==
	      FLASH_LOG_OK);
	CHECK(capture.discover_calls == 7);
	CHECK(capture.cleanup_calls == 7);
	CHECK(capture.erase_calls == 1);
	CHECK(capture.discover_completed == 6);
	CHECK(capture.cleanup_completed == 6);
	CHECK(capture.total == 6);
}

static void test_writer_creates_data_and_commit(void)
{
	struct fixture fixture;

	fixture_init(&fixture, 6);
	write_log(&fixture, 7, 1);

	CHECK(fixture.fake.sectors[0].state == FLASH_SECTOR_VALID);
	CHECK(fixture.fake.sectors[0].chunk.magic ==
	      FLASH_LOG_DATA_MAGIC);
	CHECK(fixture.fake.sectors[0].chunk.log_id == 7);
	CHECK(fixture.fake.sectors[0].chunk.sequence == 0);
	CHECK(flash_chunk_is_valid(&fixture.fake.sectors[0].chunk));
	CHECK(fixture.fake.sectors[1].chunk.magic ==
	      FLASH_LOG_COMMIT_MAGIC);
	CHECK(fixture.fake.sectors[1].chunk.sequence == 1);
	CHECK(flash_chunk_is_valid(&fixture.fake.sectors[1].chunk));
	CHECK(fixture.log.write_sector == 2);
	CHECK(!fixture.log.write_active);
	assert_ring_state(&fixture.log);
}

static void test_checkpoint_restores_closed_log_with_boundary_reads(void)
{
	struct flash_log_checkpoint checkpoint;
	struct flash_chunk restored_scratch;
	struct flash_log restored;
	struct fixture fixture;

	fixture_init(&fixture, 6);
	write_log(&fixture, 7, 1);
	flash_log_checkpoint_save(&fixture.log, &checkpoint);
	memset(fixture.fake.read_count, 0,
	       sizeof(fixture.fake.read_count));
	CHECK(flash_log_init(&restored, 6, &restored_scratch,
			     &fake_flash_storage_ops, &fixture.fake,
			     &fake_flash_transport_ops, &fixture.fake) ==
	      FLASH_LOG_OK);

	CHECK(flash_log_checkpoint_restore(&restored, &checkpoint) ==
	      FLASH_LOG_OK);
	CHECK(restored.read_sector == 0);
	CHECK(restored.write_sector == 2);
	CHECK(restored.read_log_id == 7);
	CHECK(restored.next_log_id == 8);
	CHECK(fixture.fake.read_count[0] == 1);
	CHECK(fixture.fake.read_count[1] == 1);
	CHECK(fixture.fake.read_count[2] == 1);
	CHECK(fixture.fake.read_count[3] == 0);
}

static void test_checkpoint_restores_empty_log_with_one_read(void)
{
	struct flash_log_checkpoint checkpoint;
	struct flash_chunk restored_scratch;
	struct flash_log restored;
	struct fixture fixture;

	fixture_init(&fixture, 6);
	fixture.log.next_log_id = 42;
	flash_log_checkpoint_save(&fixture.log, &checkpoint);
	CHECK(flash_log_init(&restored, 6, &restored_scratch,
			     &fake_flash_storage_ops, &fixture.fake,
			     &fake_flash_transport_ops, &fixture.fake) ==
	      FLASH_LOG_OK);

	CHECK(flash_log_checkpoint_restore(&restored, &checkpoint) ==
	      FLASH_LOG_OK);
	CHECK(flash_log_is_empty(&restored));
	CHECK(restored.next_log_id == 42);
	CHECK(fixture.fake.read_count[0] == 1);
}

static void test_checkpoint_rejects_dirty_write_sector(void)
{
	struct flash_log_checkpoint checkpoint;
	struct fixture fixture;

	fixture_init(&fixture, 6);
	write_log(&fixture, 7, 1);
	flash_log_checkpoint_save(&fixture.log, &checkpoint);
	fake_flash_set_dirty(&fixture.fake, checkpoint.write_sector, 8, 0);

	CHECK(flash_log_checkpoint_restore(&fixture.log, &checkpoint) ==
	      FLASH_LOG_CORRUPT);
}

static void test_checkpoint_rejects_noncommit_tail(void)
{
	struct flash_log_checkpoint checkpoint;
	struct fixture fixture;

	fixture_init(&fixture, 6);
	write_log(&fixture, 7, 1);
	flash_log_checkpoint_save(&fixture.log, &checkpoint);
	fake_flash_set_data(&fixture.fake, 1, 7, 1);

	CHECK(flash_log_checkpoint_restore(&fixture.log, &checkpoint) ==
	      FLASH_LOG_CORRUPT);
}

static void test_one_log_streams_data_and_commit_raw(void)
{
	struct fixture fixture;
	uint32_t free_sector_reads;

	fixture_init(&fixture, 6);
	write_log(&fixture, 7, 1);
	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);

	free_sector_reads = fixture.fake.read_count[2];
	CHECK(flash_log_read_one(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.fake.read_count[2] == free_sector_reads);
	CHECK(fixture.fake.start_count == 1);
	CHECK(fixture.fake.starts[0].log_id == 7);
	CHECK(fixture.fake.starts[0].start_sector == 0);
	CHECK(fixture.fake.sent_sector_count == 2);
	CHECK(fixture.fake.sent_sectors[0].sector == 0);
	CHECK(fixture.fake.sent_sectors[0].magic ==
	      FLASH_LOG_DATA_MAGIC);
	CHECK(fixture.fake.sent_sectors[1].sector == 1);
	CHECK(fixture.fake.sent_sectors[1].magic ==
	      FLASH_LOG_COMMIT_MAGIC);
	CHECK(fixture.fake.summary_count == 1);
	check_summary(&fixture.fake.summaries[0], 7, 0, 2, 2);
	CHECK(fixture.fake.erase_count[0] == 1);
	CHECK(fixture.fake.erase_count[1] == 1);
	CHECK(fixture.fake.erase_count[2] == 0);
	CHECK(flash_log_is_empty(&fixture.log));
	assert_ring_state(&fixture.log);
}

static void test_two_logs_stop_at_valid_new_id(void)
{
	struct fixture fixture;

	fixture_init(&fixture, 8);
	write_log(&fixture, 10, 2);
	write_log(&fixture, 11, 1);
	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);

	CHECK(flash_log_read_one(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.fake.sent_sector_count == 3);
	CHECK(fixture.fake.sent_sectors[0].sequence == 0);
	CHECK(fixture.fake.sent_sectors[1].sequence == 1);
	CHECK(fixture.fake.sent_sectors[2].magic ==
	      FLASH_LOG_COMMIT_MAGIC);
	check_summary(&fixture.fake.summaries[0], 10, 0, 3, 3);
	CHECK(fixture.fake.erase_count[0] == 1);
	CHECK(fixture.fake.erase_count[1] == 1);
	CHECK(fixture.fake.erase_count[2] == 1);
	CHECK(fixture.fake.erase_count[3] == 0);
	CHECK(fixture.log.read_sector == 3);
	CHECK(fixture.log.read_log_id == 11);

	CHECK(flash_log_read_one(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.fake.sent_sector_count == 5);
	CHECK(fixture.fake.sent_sectors[3].log_id == 11);
	CHECK(fixture.fake.sent_sectors[4].magic ==
	      FLASH_LOG_COMMIT_MAGIC);
	check_summary(&fixture.fake.summaries[1], 11, 3, 5, 2);
	CHECK(flash_log_is_empty(&fixture.log));
	assert_ring_state(&fixture.log);
}

static void test_wrapped_log(void)
{
	struct fixture fixture;

	fixture_init(&fixture, 6);
	fixture.log.read_sector = 4;
	fixture.log.write_sector = 4;
	write_log(&fixture, 20, 3);
	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.log.read_sector == 4);
	CHECK(fixture.log.write_sector == 2);

	CHECK(flash_log_read_one(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.fake.sent_sector_count == 4);
	CHECK(fixture.fake.sent_sectors[0].sector == 4);
	CHECK(fixture.fake.sent_sectors[1].sector == 5);
	CHECK(fixture.fake.sent_sectors[2].sector == 0);
	CHECK(fixture.fake.sent_sectors[3].sector == 1);
	CHECK(fixture.fake.sent_sectors[3].magic ==
	      FLASH_LOG_COMMIT_MAGIC);
	check_summary(&fixture.fake.summaries[0], 20, 4, 2, 4);
	CHECK(fixture.fake.erase_count[4] == 1);
	CHECK(fixture.fake.erase_count[5] == 1);
	CHECK(fixture.fake.erase_count[0] == 1);
	CHECK(fixture.fake.erase_count[1] == 1);
	CHECK(fixture.fake.erase_count[2] == 0);
	CHECK(flash_log_is_empty(&fixture.log));
	assert_ring_state(&fixture.log);
}

static void test_reserved_sector_full(void)
{
	struct fixture fixture;
	uint32_t reserved_sector_reads;

	fixture_init(&fixture, 6);
	fixture.log.read_sector = 1;
	fixture.log.write_sector = 1;
	write_log(&fixture, 30, 4);
	CHECK(fixture.log.read_sector == 1);
	CHECK(fixture.log.write_sector == 0);
	CHECK(flash_log_is_full(&fixture.log));
	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);
	CHECK(flash_log_is_full(&fixture.log));

	reserved_sector_reads = fixture.fake.read_count[0];
	CHECK(flash_log_read_one(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.fake.read_count[0] == reserved_sector_reads);
	CHECK(fixture.fake.sent_sector_count == 5);
	check_summary(&fixture.fake.summaries[0], 30, 1, 0, 5);
	CHECK(flash_log_is_empty(&fixture.log));
	CHECK(!flash_log_is_full(&fixture.log));
	assert_ring_state(&fixture.log);
}

static void test_writer_reserves_commit_sector(void)
{
	struct fixture fixture;
	uint8_t payload = 1;

	fixture_init(&fixture, 4);
	CHECK(flash_log_begin(&fixture.log, NULL) == FLASH_LOG_OK);
	CHECK(flash_log_append(&fixture.log, &payload, sizeof(payload)) ==
	      FLASH_LOG_OK);
	CHECK(flash_log_append(&fixture.log, &payload, sizeof(payload)) ==
	      FLASH_LOG_OK);
	CHECK(flash_log_append(&fixture.log, &payload, sizeof(payload)) ==
	      FLASH_LOG_FULL);
	CHECK(flash_log_close(&fixture.log) == FLASH_LOG_OK);
	CHECK(flash_log_is_full(&fixture.log));
	assert_ring_state(&fixture.log);
}

static void test_abort_erases_partial_log(void)
{
	struct fixture fixture;
	uint8_t payload = 1;

	fixture_init(&fixture, 6);
	CHECK(flash_log_begin(&fixture.log, NULL) == FLASH_LOG_OK);
	CHECK(flash_log_append(&fixture.log, &payload, sizeof(payload)) ==
	      FLASH_LOG_OK);
	CHECK(flash_log_append(&fixture.log, &payload, sizeof(payload)) ==
	      FLASH_LOG_OK);
	CHECK(flash_log_abort(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.fake.erase_count[0] == 1);
	CHECK(fixture.fake.erase_count[1] == 1);
	CHECK(fixture.log.read_sector == 0);
	CHECK(fixture.log.write_sector == 0);
	CHECK(fixture.log.next_log_id == 0);
	CHECK(flash_log_is_empty(&fixture.log));
	assert_ring_state(&fixture.log);
}

static void test_dirty_sector_inside_log_is_streamed_raw(void)
{
	struct fixture fixture;

	fixture_init(&fixture, 7);
	write_log(&fixture, 40, 2);
	fake_flash_set_dirty(&fixture.fake, 1, 40, 1);

	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.fake.erase_count[1] == 0);
	CHECK(flash_log_read_one(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.fake.sent_sector_count == 3);
	CHECK(fixture.fake.sent_sectors[1].sector == 1);
	CHECK(fixture.fake.sent_sectors[1].state ==
	      FLASH_SECTOR_DIRTY);
	check_summary(&fixture.fake.summaries[0], 40, 0, 3, 3);
	CHECK(fixture.fake.erase_count[1] == 1);
	CHECK(flash_log_is_empty(&fixture.log));
}

static void test_erased_gap_inside_log_is_streamed_raw(void)
{
	struct fixture fixture;

	fixture_init(&fixture, 7);
	write_log(&fixture, 50, 1);
	memcpy(&fixture.fake.sectors[2], &fixture.fake.sectors[1],
	       sizeof(fixture.fake.sectors[2]));
	memset(&fixture.fake.sectors[1].chunk, 0xff,
	       sizeof(fixture.fake.sectors[1].chunk));
	fixture.fake.sectors[1].state = FLASH_SECTOR_ERASED;

	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);
	CHECK(flash_log_read_one(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.fake.sent_sector_count == 3);
	CHECK(fixture.fake.sent_sectors[1].state ==
	      FLASH_SECTOR_ERASED);
	CHECK(fixture.fake.sent_sectors[2].magic ==
	      FLASH_LOG_COMMIT_MAGIC);
	check_summary(&fixture.fake.summaries[0], 50, 0, 3, 3);
	CHECK(flash_log_is_empty(&fixture.log));
}

static void test_transport_retry_then_erase(void)
{
	struct fixture fixture;
	const enum flash_transport_result responses[] = {
		FLASH_TRANSPORT_RETRY,
		FLASH_TRANSPORT_ERASE,
	};

	fixture_init(&fixture, 6);
	write_log(&fixture, 60, 1);
	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);
	fake_flash_script_finish(&fixture.fake, responses, 2);

	CHECK(flash_log_drain(&fixture.log, 3) == FLASH_LOG_OK);
	CHECK(fixture.fake.start_count == 2);
	CHECK(fixture.fake.sent_sector_count == 4);
	CHECK(fixture.fake.summary_count == 2);
	CHECK(fixture.fake.summaries[0].raw_crc ==
	      fixture.fake.summaries[1].raw_crc);
	CHECK(fixture.fake.erase_count[0] == 1);
	CHECK(fixture.fake.erase_count[1] == 1);
	CHECK(flash_log_is_empty(&fixture.log));
	assert_ring_state(&fixture.log);
}

static void test_transport_done_preserves_range(void)
{
	struct fixture fixture;
	const enum flash_transport_result response =
		FLASH_TRANSPORT_DONE;

	fixture_init(&fixture, 6);
	write_log(&fixture, 65, 1);
	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);
	fake_flash_script_finish(&fixture.fake, &response, 1);

	CHECK(flash_log_read_one(&fixture.log) ==
	      FLASH_LOG_TRANSPORT_DONE);
	CHECK(fixture.log.read_sector == 0);
	CHECK(fixture.log.write_sector == 2);
	CHECK(fixture.fake.erase_count[0] == 0);
	CHECK(fixture.fake.erase_count[1] == 0);
	CHECK(!flash_log_is_empty(&fixture.log));
}

static void test_scan_trims_incomplete_newest_dirty_tail(void)
{
	struct fixture fixture;
	uint8_t payload = 1;

	fixture_init(&fixture, 8);
	write_log(&fixture, 70, 1);
	CHECK(flash_log_begin(&fixture.log, NULL) == FLASH_LOG_OK);
	CHECK(flash_log_append(&fixture.log, &payload, sizeof(payload)) ==
	      FLASH_LOG_OK);
	fake_flash_set_dirty(&fixture.fake, 3, 71, 1);

	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.fake.erase_count[3] == 1);
	CHECK(fixture.fake.sectors[2].state == FLASH_SECTOR_VALID);
	CHECK(fixture.log.read_sector == 0);
	CHECK(fixture.log.write_sector == 3);
	CHECK(fixture.log.next_log_id == 72);

	CHECK(flash_log_read_one(&fixture.log) == FLASH_LOG_OK);
	check_summary(&fixture.fake.summaries[0], 70, 0, 2, 2);
	CHECK(fixture.log.read_sector == 2);
	CHECK(flash_log_read_one(&fixture.log) == FLASH_LOG_OK);
	check_summary(&fixture.fake.summaries[1], 71, 2, 3, 1);
	CHECK(fixture.fake.sent_sectors[2].magic ==
	      FLASH_LOG_DATA_MAGIC);
	CHECK(flash_log_is_empty(&fixture.log));
	assert_ring_state(&fixture.log);
}

static void test_scan_reclaims_dirty_free_arc(void)
{
	struct fixture fixture;

	fixture_init(&fixture, 6);
	write_log(&fixture, 75, 1);
	fake_flash_set_dirty(&fixture.fake, 2, 76, 0);

	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.fake.erase_count[2] == 1);
	CHECK(fixture.fake.sectors[2].state == FLASH_SECTOR_ERASED);
	CHECK(fixture.log.read_sector == 0);
	CHECK(fixture.log.write_sector == 2);
	CHECK(fixture.log.next_log_id == 76);
	assert_ring_state(&fixture.log);
}

static void test_scan_cleans_interrupted_old_erase(void)
{
	struct fixture fixture;

	fixture_init(&fixture, 6);
	write_log(&fixture, 80, 1);
	write_log(&fixture, 81, 1);
	CHECK(fake_flash_storage_ops.erase_sector(&fixture.fake, 0) == 0);
	fake_flash_set_dirty(&fixture.fake, 1, 80, 1);

	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.fake.erase_count[1] == 1);
	CHECK(fixture.fake.sectors[1].state == FLASH_SECTOR_ERASED);
	CHECK(fixture.log.read_sector == 2);
	CHECK(fixture.log.read_log_id == 81);
	CHECK(fixture.log.write_sector == 4);
	CHECK(flash_log_read_one(&fixture.log) == FLASH_LOG_OK);
	check_summary(&fixture.fake.summaries[0], 81, 2, 4, 2);
	CHECK(flash_log_is_empty(&fixture.log));
}

static void test_scan_rejects_valid_sector_in_free_arc(void)
{
	struct fixture fixture;

	fixture_init(&fixture, 6);
	fake_flash_set_data(&fixture.fake, 0, 90, 0);
	fake_flash_set_data(&fixture.fake, 2, 92, 0);
	fake_flash_set_data(&fixture.fake, 4, 91, 0);

	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_CORRUPT);
	CHECK(fixture.fake.erase_count[4] == 0);
	assert_ring_state(&fixture.log);
}

static void test_scan_rejects_all_sectors_valid(void)
{
	struct fixture fixture;

	fixture_init(&fixture, 6);
	for (uint32_t sequence = 0; sequence < 6; sequence++) {
		fake_flash_set_data(&fixture.fake, sequence, 100,
				    sequence);
	}
	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_CORRUPT);
	assert_ring_state(&fixture.log);
}

static void run_test(const char *name, void (*test)(void))
{
	printf("[ RUN      ] %s\n", name);
	test();
	printf("[       OK ] %s\n", name);
}

int main(void)
{
	run_test("empty_scan_cleans_dirty_media",
		 test_empty_scan_cleans_dirty_media);
	run_test("scan_reports_discovery_and_cleanup_progress",
		 test_scan_reports_discovery_and_cleanup_progress);
	run_test("writer_creates_data_and_commit",
		 test_writer_creates_data_and_commit);
	run_test("checkpoint_restores_closed_log_with_boundary_reads",
		 test_checkpoint_restores_closed_log_with_boundary_reads);
	run_test("checkpoint_restores_empty_log_with_one_read",
		 test_checkpoint_restores_empty_log_with_one_read);
	run_test("checkpoint_rejects_dirty_write_sector",
		 test_checkpoint_rejects_dirty_write_sector);
	run_test("checkpoint_rejects_noncommit_tail",
		 test_checkpoint_rejects_noncommit_tail);
	run_test("one_log_streams_data_and_commit_raw",
		 test_one_log_streams_data_and_commit_raw);
	run_test("two_logs_stop_at_valid_new_id",
		 test_two_logs_stop_at_valid_new_id);
	run_test("wrapped_log", test_wrapped_log);
	run_test("reserved_sector_full", test_reserved_sector_full);
	run_test("writer_reserves_commit_sector",
		 test_writer_reserves_commit_sector);
	run_test("abort_erases_partial_log",
		 test_abort_erases_partial_log);
	run_test("dirty_sector_inside_log_is_streamed_raw",
		 test_dirty_sector_inside_log_is_streamed_raw);
	run_test("erased_gap_inside_log_is_streamed_raw",
		 test_erased_gap_inside_log_is_streamed_raw);
	run_test("transport_retry_then_erase",
		 test_transport_retry_then_erase);
	run_test("transport_done_preserves_range",
		 test_transport_done_preserves_range);
	run_test("scan_trims_incomplete_newest_dirty_tail",
		 test_scan_trims_incomplete_newest_dirty_tail);
	run_test("scan_reclaims_dirty_free_arc",
		 test_scan_reclaims_dirty_free_arc);
	run_test("scan_cleans_interrupted_old_erase",
		 test_scan_cleans_interrupted_old_erase);
	run_test("scan_rejects_valid_sector_in_free_arc",
		 test_scan_rejects_valid_sector_in_free_arc);
	run_test("scan_rejects_all_sectors_valid",
		 test_scan_rejects_all_sectors_valid);
	printf("[  PASSED  ] 22 tests\n");
	return 0;
}
