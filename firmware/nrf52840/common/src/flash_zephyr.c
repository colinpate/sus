#include "flash_zephyr.h"

#include <errno.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/flash.h>
#include <zephyr/sys/util.h>

#define MX25L25645G_NODE DT_NODELABEL(mx25l25645g)
#define MX25L25645G_BYTES (DT_PROP(MX25L25645G_NODE, size) / 8U)
#define MX25L25645G_SECTORS \
	(MX25L25645G_BYTES / FLASH_LOG_SECTOR_BYTES)

BUILD_ASSERT(DT_NODE_HAS_STATUS(MX25L25645G_NODE, okay),
	     "mx25l25645g devicetree node must be enabled");
BUILD_ASSERT((DT_PROP(MX25L25645G_NODE, size) % 8U) == 0U,
	     "SPI NOR size must contain a whole number of bytes");
BUILD_ASSERT((MX25L25645G_BYTES % FLASH_LOG_SECTOR_BYTES) == 0U,
	     "SPI NOR size must be a multiple of the log sector size");

static bool flash_zephyr_sector_is_erased(const struct flash_chunk *chunk)
{
	const uint8_t *bytes = (const uint8_t *)chunk;

	for (size_t i = 0; i < sizeof(*chunk); i++) {
		if (bytes[i] != UINT8_MAX) {
			return false;
		}
	}
	return true;
}

static bool flash_zephyr_sector_in_range(
	const struct flash_zephyr_storage *storage, uint32_t sector)
{
	return storage != NULL && storage->device != NULL &&
	       sector < storage->sector_count;
}

static off_t flash_zephyr_sector_offset(
	const struct flash_zephyr_storage *storage, uint32_t sector)
{
	return (off_t)storage->base_offset +
	       (off_t)sector * FLASH_LOG_SECTOR_BYTES;
}

static int flash_zephyr_read_sector(void *context, uint32_t sector,
				    struct flash_chunk *chunk,
				    enum flash_sector_state *state)
{
	struct flash_zephyr_storage *storage = context;
	int err;

	if (!flash_zephyr_sector_in_range(storage, sector) ||
	    chunk == NULL || state == NULL) {
		return -EINVAL;
	}

	err = flash_read(storage->device,
			 flash_zephyr_sector_offset(storage, sector),
			 chunk, sizeof(*chunk));
	if (err != 0) {
		return err;
	}

	if (flash_zephyr_sector_is_erased(chunk)) {
		*state = FLASH_SECTOR_ERASED;
	} else if (flash_chunk_is_valid(chunk)) {
		*state = FLASH_SECTOR_VALID;
	} else {
		*state = FLASH_SECTOR_DIRTY;
	}
	return 0;
}

static int flash_zephyr_write_sector(void *context, uint32_t sector,
				     const struct flash_chunk *chunk)
{
	struct flash_zephyr_storage *storage = context;

	if (!flash_zephyr_sector_in_range(storage, sector) ||
	    chunk == NULL || !flash_chunk_is_valid(chunk)) {
		return -EINVAL;
	}

	return flash_write(storage->device,
			   flash_zephyr_sector_offset(storage, sector),
			   chunk, sizeof(*chunk));
}

static int flash_zephyr_erase_sector(void *context, uint32_t sector)
{
	struct flash_zephyr_storage *storage = context;

	if (!flash_zephyr_sector_in_range(storage, sector)) {
		return -EINVAL;
	}

	return flash_erase(storage->device,
			   flash_zephyr_sector_offset(storage, sector),
			   FLASH_LOG_SECTOR_BYTES);
}

const struct flash_storage_ops flash_zephyr_storage_ops = {
	.read_sector = flash_zephyr_read_sector,
	.write_sector = flash_zephyr_write_sector,
	.erase_sector = flash_zephyr_erase_sector,
};

int flash_zephyr_storage_init(struct flash_zephyr_storage *storage,
			      const struct device *device,
			      uint32_t base_offset,
			      uint32_t sector_count)
{
	uint64_t end_offset;

	if (storage == NULL || device == NULL || sector_count < 3U ||
	    !device_is_ready(device) ||
	    (base_offset % FLASH_LOG_SECTOR_BYTES) != 0U) {
		return -EINVAL;
	}

	end_offset = (uint64_t)base_offset +
		     (uint64_t)sector_count * FLASH_LOG_SECTOR_BYTES;
	if (end_offset > MX25L25645G_BYTES) {
		return -ERANGE;
	}

	storage->device = device;
	storage->base_offset = base_offset;
	storage->sector_count = sector_count;
	return 0;
}

int flash_zephyr_storage_init_default(struct flash_zephyr_storage *storage)
{
	return flash_zephyr_storage_init(
		storage, DEVICE_DT_GET(MX25L25645G_NODE), 0U,
		MX25L25645G_SECTORS);
}
