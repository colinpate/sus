#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
	result = flash_log_init(&fixture->log, sector_count,
				&fixture->scratch, &fake_flash_ops,
				&fixture->fake);
	CHECK(result == FLASH_LOG_OK);
}

static void assert_ring_state(const struct flash_log *log)
{
	CHECK(log->read_sector < log->sector_count);
	CHECK(log->write_sector < log->sector_count);
	CHECK(!(flash_log_is_empty(log) && flash_log_is_full(log)));
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

static void test_one_sector_log(void)
{
	struct fixture fixture;
	uint32_t free_sector_reads;

	fixture_init(&fixture, 6);
	fake_flash_set_valid(&fixture.fake, 0, 7, 0);
	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.log.read_sector == 0);
	CHECK(fixture.log.write_sector == 1);
	CHECK(fixture.log.next_log_id == 8);

	free_sector_reads = fixture.fake.read_count[1];
	CHECK(flash_log_read_one(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.fake.read_count[1] == free_sector_reads);
	CHECK(fixture.fake.sent_log_start_count == 1);
	CHECK(fixture.fake.sent_log_starts[0] == 7);
	CHECK(fixture.fake.sent_chunk_count == 1);
	CHECK(fixture.fake.sent_chunks[0].log_id == 7);
	CHECK(fixture.fake.sent_chunks[0].sequence == 0);
	CHECK(fixture.fake.sent_log_end_count == 1);
	CHECK(fixture.fake.erase_count[0] == 1);
	CHECK(fixture.fake.erase_count[1] == 0);
	CHECK(flash_log_is_empty(&fixture.log));
	assert_ring_state(&fixture.log);
}

static void test_two_logs_preserve_boundary(void)
{
	struct fixture fixture;

	fixture_init(&fixture, 6);
	fake_flash_set_valid(&fixture.fake, 0, 10, 0);
	fake_flash_set_valid(&fixture.fake, 1, 10, 1);
	fake_flash_set_valid(&fixture.fake, 2, 11, 0);
	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);

	CHECK(flash_log_read_one(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.fake.sent_chunk_count == 2);
	CHECK(fixture.fake.sent_chunks[0].sequence == 0);
	CHECK(fixture.fake.sent_chunks[1].sequence == 1);
	CHECK(fixture.fake.erase_count[0] == 1);
	CHECK(fixture.fake.erase_count[1] == 1);
	CHECK(fixture.fake.erase_count[2] == 0);
	CHECK(fixture.fake.sectors[2].state == FLASH_SECTOR_VALID);
	CHECK(fixture.log.read_sector == 2);

	CHECK(flash_log_read_one(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.fake.sent_chunk_count == 3);
	CHECK(fixture.fake.sent_chunks[2].log_id == 11);
	CHECK(fixture.fake.sent_chunks[2].sequence == 0);
	CHECK(fixture.fake.erase_count[2] == 1);
	CHECK(flash_log_is_empty(&fixture.log));
	assert_ring_state(&fixture.log);
}

static void test_wrapped_log(void)
{
	struct fixture fixture;

	fixture_init(&fixture, 6);
	fake_flash_set_valid(&fixture.fake, 4, 20, 0);
	fake_flash_set_valid(&fixture.fake, 5, 20, 1);
	fake_flash_set_valid(&fixture.fake, 0, 20, 2);
	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.log.read_sector == 4);
	CHECK(fixture.log.write_sector == 1);

	CHECK(flash_log_read_one(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.fake.sent_chunk_count == 3);
	CHECK(fixture.fake.sent_chunks[0].sequence == 0);
	CHECK(fixture.fake.sent_chunks[1].sequence == 1);
	CHECK(fixture.fake.sent_chunks[2].sequence == 2);
	CHECK(fixture.fake.erase_count[4] == 1);
	CHECK(fixture.fake.erase_count[5] == 1);
	CHECK(fixture.fake.erase_count[0] == 1);
	CHECK(fixture.fake.erase_count[1] == 0);
	CHECK(flash_log_is_empty(&fixture.log));
	assert_ring_state(&fixture.log);
}

static void test_reserved_sector_full(void)
{
	struct fixture fixture;
	uint32_t reserved_sector_reads;

	fixture_init(&fixture, 6);
	for (uint32_t sequence = 0; sequence < 5; sequence++) {
		fake_flash_set_valid(&fixture.fake, sequence + 1U, 30,
				     sequence);
	}
	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.log.read_sector == 1);
	CHECK(fixture.log.write_sector == 0);
	CHECK(!flash_log_is_empty(&fixture.log));
	CHECK(flash_log_is_full(&fixture.log));

	reserved_sector_reads = fixture.fake.read_count[0];
	CHECK(flash_log_read_one(&fixture.log) == FLASH_LOG_OK);
	CHECK(fixture.fake.read_count[0] == reserved_sector_reads);
	CHECK(fixture.fake.sent_chunk_count == 5);
	CHECK(flash_log_is_empty(&fixture.log));
	CHECK(!flash_log_is_full(&fixture.log));
	assert_ring_state(&fixture.log);
}

static void test_dirty_oldest_sector_is_not_consumed(void)
{
	struct fixture fixture;

	fixture_init(&fixture, 6);
	fake_flash_set_valid(&fixture.fake, 0, 40, 0);
	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);
	fake_flash_set_dirty(&fixture.fake, 0, 40, 0);

	CHECK(flash_log_read_one(&fixture.log) == FLASH_LOG_CORRUPT);
	CHECK(fixture.fake.sent_log_start_count == 0);
	CHECK(fixture.fake.sent_chunk_count == 0);
	CHECK(fixture.fake.sent_log_end_count == 0);
	CHECK(fixture.fake.erase_count[0] == 0);
	CHECK(fixture.log.read_sector == 0);
	CHECK(fixture.log.write_sector == 1);
	assert_ring_state(&fixture.log);
}

static void test_dirty_middle_preserves_valid_prefix(void)
{
	struct fixture fixture;

	fixture_init(&fixture, 6);
	fake_flash_set_valid(&fixture.fake, 0, 50, 0);
	fake_flash_set_valid(&fixture.fake, 1, 50, 1);
	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);
	fake_flash_set_dirty(&fixture.fake, 1, 50, 1);

	CHECK(flash_log_read_one(&fixture.log) == FLASH_LOG_CORRUPT);
	CHECK(fixture.fake.sent_chunk_count == 1);
	CHECK(fixture.fake.sent_log_end_count == 0);
	CHECK(fixture.fake.erase_count[0] == 0);
	CHECK(fixture.fake.erase_count[1] == 0);
	CHECK(fixture.log.read_sector == 0);
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
	fake_flash_set_valid(&fixture.fake, 0, 60, 0);
	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_OK);
	fake_flash_script_log_end(&fixture.fake, responses, 2);

	CHECK(flash_log_drain(&fixture.log, 3) == FLASH_LOG_OK);
	CHECK(fixture.fake.sent_log_start_count == 2);
	CHECK(fixture.fake.sent_chunk_count == 2);
	CHECK(fixture.fake.sent_log_end_count == 2);
	CHECK(fixture.fake.erase_count[0] == 1);
	CHECK(flash_log_is_empty(&fixture.log));
	assert_ring_state(&fixture.log);
}

static void test_scan_rejects_gap(void)
{
	struct fixture fixture;

	fixture_init(&fixture, 6);
	fake_flash_set_valid(&fixture.fake, 0, 70, 0);
	fake_flash_set_valid(&fixture.fake, 2, 70, 1);
	CHECK(flash_log_scan(&fixture.log) == FLASH_LOG_CORRUPT);
	CHECK(fixture.fake.erase_count[0] == 0);
	CHECK(fixture.fake.erase_count[2] == 0);
	assert_ring_state(&fixture.log);
}

static void test_scan_rejects_all_sectors_valid(void)
{
	struct fixture fixture;

	fixture_init(&fixture, 6);
	for (uint32_t sequence = 0; sequence < 6; sequence++) {
		fake_flash_set_valid(&fixture.fake, sequence, 80, sequence);
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
	run_test("one_sector_log", test_one_sector_log);
	run_test("two_logs_preserve_boundary",
		 test_two_logs_preserve_boundary);
	run_test("wrapped_log", test_wrapped_log);
	run_test("reserved_sector_full", test_reserved_sector_full);
	run_test("dirty_oldest_sector_is_not_consumed",
		 test_dirty_oldest_sector_is_not_consumed);
	run_test("dirty_middle_preserves_valid_prefix",
		 test_dirty_middle_preserves_valid_prefix);
	run_test("transport_retry_then_ack",
		 test_transport_retry_then_ack);
	run_test("scan_rejects_gap", test_scan_rejects_gap);
	run_test("scan_rejects_all_sectors_valid",
		 test_scan_rejects_all_sectors_valid);
	printf("[  PASSED  ] 10 tests\n");
	return 0;
}
