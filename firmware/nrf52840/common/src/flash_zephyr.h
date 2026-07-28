#ifndef SUS_FLASH_ZEPHYR_H_
#define SUS_FLASH_ZEPHYR_H_

#include <stdint.h>

#include <zephyr/device.h>

#include "flash.h"

struct flash_zephyr_storage {
	const struct device *device;
	uint32_t base_offset;
	uint32_t sector_count;
};

int flash_zephyr_storage_init(struct flash_zephyr_storage *storage,
			      const struct device *device,
			      uint32_t base_offset,
			      uint32_t sector_count);
int flash_zephyr_storage_init_default(struct flash_zephyr_storage *storage);

extern const struct flash_storage_ops flash_zephyr_storage_ops;

#endif /* SUS_FLASH_ZEPHYR_H_ */
