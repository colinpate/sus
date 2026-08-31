#ifndef SUS_FLASH_SMOKE_TEST_H_
#define SUS_FLASH_SMOKE_TEST_H_

#include <stdbool.h>
#include <stdint.h>

struct flash_smoke_test_result {
	uint8_t jedec_id[3];
	bool matches_expected_id;
};

/*
 * Wake the external flash and read its JEDEC manufacturer/device ID.
 * This test is read-only and does not erase or program the chip.
 */
int flash_smoke_test_run(struct flash_smoke_test_result *result);

#endif /* SUS_FLASH_SMOKE_TEST_H_ */
