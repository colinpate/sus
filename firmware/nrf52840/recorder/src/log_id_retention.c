#include "log_id_retention.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/retained_mem.h>
#include <zephyr/sys/crc.h>

#define RETAINED_LOG_ID_MAGIC UINT32_C(0x4c535553)
#define RETAINED_LOG_ID_VERSION UINT32_C(1)

struct retained_log_id {
	uint32_t magic;
	uint32_t version;
	uint32_t next_log_id;
	uint32_t next_log_id_inverse;
	uint32_t crc;
};

static const struct device *const retained_memory =
	DEVICE_DT_GET(DT_ALIAS(retainedmemdevice));

static uint32_t retained_log_id_crc(const struct retained_log_id *state)
{
	return crc32_ieee((const uint8_t *)state,
			  offsetof(struct retained_log_id, crc));
}

static int retained_log_id_device_check(void)
{
	if (!device_is_ready(retained_memory)) {
		return -ENODEV;
	}
	if (retained_mem_size(retained_memory) <
	    (ssize_t)sizeof(struct retained_log_id)) {
		return -ENOSPC;
	}
	return 0;
}

int log_id_retention_load(uint32_t *next_log_id)
{
	struct retained_log_id state;
	int err;

	if (next_log_id == NULL) {
		return -EINVAL;
	}

	err = retained_log_id_device_check();
	if (err != 0) {
		return err;
	}

	err = retained_mem_read(retained_memory, 0, (uint8_t *)&state,
				sizeof(state));
	if (err != 0) {
		return err;
	}

	if (state.magic != RETAINED_LOG_ID_MAGIC ||
	    state.version != RETAINED_LOG_ID_VERSION ||
	    state.next_log_id_inverse != ~state.next_log_id ||
	    state.crc != retained_log_id_crc(&state)) {
		return -ENODATA;
	}

	*next_log_id = state.next_log_id;
	return 0;
}

int log_id_retention_store(uint32_t next_log_id)
{
	struct retained_log_id state = {
		.magic = RETAINED_LOG_ID_MAGIC,
		.version = RETAINED_LOG_ID_VERSION,
		.next_log_id = next_log_id,
		.next_log_id_inverse = ~next_log_id,
	};
	int err = retained_log_id_device_check();

	if (err != 0) {
		return err;
	}

	state.crc = retained_log_id_crc(&state);
	return retained_mem_write(retained_memory, 0,
				  (const uint8_t *)&state, sizeof(state));
}
