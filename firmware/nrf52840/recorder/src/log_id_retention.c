#include "log_id_retention.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/retained_mem.h>
#include <zephyr/sys/crc.h>

#define RETAINED_LOG_ID_MAGIC UINT32_C(0x4c535553)
#define RETAINED_LOG_ID_VERSION_LEGACY UINT32_C(1)
#define RETAINED_LOG_ID_VERSION UINT32_C(2)
#define RETAINED_LOG_STATE_CLEAN UINT32_C(0x434c454e)
#define RETAINED_LOG_STATE_DIRTY UINT32_C(0x44495254)

struct retained_log_id_legacy {
	uint32_t magic;
	uint32_t version;
	uint32_t next_log_id;
	uint32_t next_log_id_inverse;
	uint32_t crc;
};

struct retained_log_id {
	uint32_t magic;
	uint32_t version;
	uint32_t state;
	struct flash_log_checkpoint checkpoint;
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

static uint32_t
retained_log_id_legacy_crc(const struct retained_log_id_legacy *state)
{
	return crc32_ieee((const uint8_t *)state,
			  offsetof(struct retained_log_id_legacy, crc));
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

int log_id_retention_load(struct retained_log_state *result)
{
	struct retained_log_id state;
	struct retained_log_id_legacy legacy;
	int err;

	if (result == NULL) {
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
	memcpy(&legacy, &state, sizeof(legacy));

	if (legacy.magic == RETAINED_LOG_ID_MAGIC &&
	    legacy.version == RETAINED_LOG_ID_VERSION_LEGACY &&
	    legacy.next_log_id_inverse == ~legacy.next_log_id &&
	    legacy.crc == retained_log_id_legacy_crc(&legacy)) {
		result->checkpoint = (struct flash_log_checkpoint) {
			.next_log_id = legacy.next_log_id,
		};
		result->clean = false;
		return 0;
	}

	if (state.magic != RETAINED_LOG_ID_MAGIC ||
	    state.version != RETAINED_LOG_ID_VERSION ||
	    (state.state != RETAINED_LOG_STATE_CLEAN &&
	     state.state != RETAINED_LOG_STATE_DIRTY) ||
	    state.next_log_id_inverse != ~state.checkpoint.next_log_id ||
	    state.crc != retained_log_id_crc(&state)) {
		return -ENODATA;
	}

	result->checkpoint = state.checkpoint;
	result->clean = state.state == RETAINED_LOG_STATE_CLEAN;
	return 0;
}

int log_id_retention_store(const struct flash_log_checkpoint *checkpoint,
			   bool clean)
{
	struct retained_log_id state;
	int err;

	if (checkpoint == NULL) {
		return -EINVAL;
	}

	state = (struct retained_log_id) {
		.magic = RETAINED_LOG_ID_MAGIC,
		.version = RETAINED_LOG_ID_VERSION,
		.state = clean ? RETAINED_LOG_STATE_CLEAN :
				 RETAINED_LOG_STATE_DIRTY,
		.checkpoint = *checkpoint,
		.next_log_id_inverse = ~checkpoint->next_log_id,
	};
	err = retained_log_id_device_check();

	if (err != 0) {
		return err;
	}

	state.crc = retained_log_id_crc(&state);
	return retained_mem_write(retained_memory, 0,
				  (const uint8_t *)&state, sizeof(state));
}
