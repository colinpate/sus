#include "flash_smoke_test.h"

#include <errno.h>
#include <string.h>

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/flash.h>

#define FLASH_EXPECTED_MANUFACTURER 0xc2
#define FLASH_EXPECTED_MEMORY_TYPE 0x20
#define FLASH_EXPECTED_DENSITY 0x19

static const struct device *const flash =
	DEVICE_DT_GET(DT_NODELABEL(mx25l25645g));

int flash_smoke_test_run(struct flash_smoke_test_result *result)
{
	int err;

	if (result == NULL) {
		return -EINVAL;
	}
	memset(result, 0, sizeof(*result));

	if (!device_is_ready(flash)) {
		return -ENODEV;
	}

	err = flash_read_jedec_id(flash, result->jedec_id);
	if (err != 0) {
		return err;
	}

	result->matches_expected_id =
		result->jedec_id[0] == FLASH_EXPECTED_MANUFACTURER &&
		result->jedec_id[1] == FLASH_EXPECTED_MEMORY_TYPE &&
		result->jedec_id[2] == FLASH_EXPECTED_DENSITY;

	return 0;
}
