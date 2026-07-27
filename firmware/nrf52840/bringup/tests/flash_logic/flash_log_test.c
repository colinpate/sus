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

static void test_empty_scan(void)
{
	struct fixture fixture;

	fixture_init(&fixture, 6);
	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.log.read_sector == 0);
	CHECK(fixture.log.write_sector == 0);
	CHECK(fixture.log.next_log_id == 0);
	CHECK(flash_log_is_empty(&fixture.log));
	CHECK(!flash_log_is_full(&fixture.log));
	assert_ring_state(&fixture.log);
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

static void test_one_chunk_log_round_trip(void)
{
	struct fixture fixture;
	uint32_t free_sector_reads;

	fixture_init(&fixture, 6);
	write_log(&fixture, 7, 1);
	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.log.read_sector == 0);
	CHECK(fixture.log.write_sector == 2);
	CHECK(fixture.log.next_log_id == 8);

	free_sector_reads = fixture.fake.read_count[2];
	CHECK(flash_log_read_one(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.fake.read_count[2] == free_sector_reads);
	CHECK(fixture.fake.sent_log_start_count == 1);
	CHECK(fixture.fake.sent_log_starts[0] == 7);
	CHECK(fixture.fake.sent_chunk_count == 1);
	CHECK(fixture.fake.sent_chunks[0].log_id == 7);
	CHECK(fixture.fake.sent_chunks[0].sequence == 0);
	CHECK(fixture.fake.sent_log_end_count == 1);
	CHECK(fixture.fake.erase_count[0] == 1);
	CHECK(fixture.fake.erase_count[1] == 1);
	CHECK(fixture.fake.erase_count[2] == 0);
	CHECK(flash_log_is_empty(&fixture.log));
	assert_ring_state(&fixture.log);
}

static void test_two_logs_preserve_boundary(void)
{
	struct fixture fixture;

	fixture_init(&fixture, 8);
	write_log(&fixture, 10, 2);
	write_log(&fixture, 11, 1);
	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);

	CHECK(flash_log_read_one(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.fake.sent_chunk_count == 2);
	CHECK(fixture.fake.sent_chunks[0].sequence == 0);
	CHECK(fixture.fake.sent_chunks[1].sequence == 1);
	CHECK(fixture.fake.erase_count[0] == 1);
	CHECK(fixture.fake.erase_count[1] == 1);
	CHECK(fixture.fake.erase_count[2] == 1);
	CHECK(fixture.fake.erase_count[3] == 0);
	CHECK(fixture.fake.sectors[3].state == FLASH_SECTOR_VALID);
	CHECK(fixture.log.read_sector == 3);

	CHECK(flash_log_read_one(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.fake.sent_chunk_count == 3);
	CHECK(fixture.fake.sent_chunks[2].log_id == 11);
	CHECK(fixture.fake.sent_chunks[2].sequence == 0);
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
	CHECK(fixture.fake.sent_chunk_count == 3);
	CHECK(fixture.fake.sent_chunks[0].sequence == 0);
	CHECK(fixture.fake.sent_chunks[1].sequence == 1);
	CHECK(fixture.fake.sent_chunks[2].sequence == 2);
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
	CHECK(!flash_log_is_empty(&fixture.log));
	CHECK(flash_log_is_full(&fixture.log));
	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);
	CHECK(flash_log_is_full(&fixture.log));

	reserved_sector_reads = fixture.fake.read_count[0];
	CHECK(flash_log_read_one(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.fake.read_count[0] == reserved_sector_reads);
	CHECK(fixture.fake.sent_chunk_count == 4);
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

static void test_dirty_sector_is_not_consumed(void)
{
	struct fixture fixture;

	fixture_init(&fixture, 6);
	write_log(&fixture, 40, 1);
	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);
	fake_flash_set_dirty(&fixture.fake, 0, 40, 0);

	CHECK(flash_log_read_one(&fixture.log) == FLASH_LOG_CORRUPT);
	CHECK(fixture.fake.sent_log_start_count == 0);
	CHECK(fixture.fake.sent_chunk_count == 0);
	CHECK(fixture.fake.sent_log_end_count == 0);
	CHECK(fixture.fake.erase_count[0] == 0);
	CHECK(fixture.log.read_sector == 0);
	CHECK(fixture.log.write_sector == 2);
	assert_ring_state(&fixture.log);
}

static void test_transport_retry_then_ack(void)
{
	struct fixture fixture;
	const enum flash_transport_result responses[] = {
		FLASH_TRANSPORT_ERROR,
		FLASH_TRANSPORT_ACK,
	};

	fixture_init(&fixture, 6);
	write_log(&fixture, 60, 1);
	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);
	fake_flash_script_log_end(&fixture.fake, responses, 2);

	CHECK(flash_log_drain(&fixture.log, 3) == FLASH_LOG_OK);
	CHECK(fixture.fake.sent_log_start_count == 2);
	CHECK(fixture.fake.sent_chunk_count == 2);
	CHECK(fixture.fake.sent_log_end_count == 2);
	CHECK(fixture.fake.erase_count[0] == 1);
	CHECK(fixture.fake.erase_count[1] == 1);
	CHECK(flash_log_is_empty(&fixture.log));
	assert_ring_state(&fixture.log);
}

static void test_scan_rejects_gap(void)
{
	struct fixture fixture;

	fixture_init(&fixture, 6);
	write_log(&fixture, 70, 1);
	memcpy(&fixture.fake.sectors[2], &fixture.fake.sectors[1],
	       sizeof(fixture.fake.sectors[2]));
	memset(&fixture.fake.sectors[1].chunk, 0xff,
	       sizeof(fixture.fake.sectors[1].chunk));
	fixture.fake.sectors[1].state = FLASH_SECTOR_ERASED;

	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_CORRUPT);
	CHECK(fixture.fake.erase_count[0] == 0);
	CHECK(fixture.fake.erase_count[2] == 0);
	assert_ring_state(&fixture.log);
}

static void test_scan_reports_incomplete_tail(void)
{
	struct fixture fixture;
	uint8_t payload = 1;

	fixture_init(&fixture, 6);
	CHECK(flash_log_begin(&fixture.log, NULL) == FLASH_LOG_OK);
	CHECK(flash_log_append(&fixture.log, &payload, sizeof(payload)) ==
	      FLASH_LOG_OK);

	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_INCOMPLETE);
	CHECK(fixture.log.tail_incomplete);
	CHECK(flash_log_begin(&fixture.log, NULL) ==
	      FLASH_LOG_INCOMPLETE);
	CHECK(flash_log_read_one(&fixture.log) ==
	      FLASH_LOG_INCOMPLETE);
	CHECK(fixture.fake.erase_count[0] == 0);
	CHECK(flash_log_discard_incomplete(&fixture.log) ==
	      FLASH_LOG_OK);
	CHECK(fixture.fake.erase_count[0] == 1);
	CHECK(flash_log_is_empty(&fixture.log));
	CHECK(fixture.log.next_log_id == 0);
	CHECK(flash_log_begin(&fixture.log, NULL) == FLASH_LOG_OK);
	CHECK(flash_log_abort(&fixture.log) == FLASH_LOG_OK);
	assert_ring_state(&fixture.log);
}

static void test_scan_reclaims_dirty_tail_sector(void)
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

static void test_scan_rejects_all_sectors_valid(void)
{
	struct fixture fixture;

	fixture_init(&fixture, 6);
	for (uint32_t sequence = 0; sequence < 6; sequence++) {
		fake_flash_set_data(&fixture.fake, sequence, 80, sequence);
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
	run_test("empty_scan", test_empty_scan);
	run_test("writer_creates_data_and_commit",
		 test_writer_creates_data_and_commit);
	run_test("one_chunk_log_round_trip",
		 test_one_chunk_log_round_trip);
	run_test("two_logs_preserve_boundary",
		 test_two_logs_preserve_boundary);
	run_test("wrapped_log", test_wrapped_log);
	run_test("reserved_sector_full", test_reserved_sector_full);
	run_test("writer_reserves_commit_sector",
		 test_writer_reserves_commit_sector);
	run_test("abort_erases_partial_log",
		 test_abort_erases_partial_log);
	run_test("dirty_sector_is_not_consumed",
		 test_dirty_sector_is_not_consumed);
	run_test("transport_retry_then_ack",
		 test_transport_retry_then_ack);
	run_test("scan_rejects_gap", test_scan_rejects_gap);
	run_test("scan_reports_incomplete_tail",
		 test_scan_reports_incomplete_tail);
	run_test("scan_reclaims_dirty_tail_sector",
		 test_scan_reclaims_dirty_tail_sector);
	run_test("scan_rejects_all_sectors_valid",
		 test_scan_rejects_all_sectors_valid);
	printf("[  PASSED  ] 14 tests\n");
	return 0;
}
