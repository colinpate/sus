#ifndef SUS_TEST_FAKE_FLASH_H_
#define SUS_TEST_FAKE_FLASH_H_

#include <stdint.h>

#include "flash.h"

#define FAKE_FLASH_MAX_SECTORS 16U
#define FAKE_FLASH_MAX_EVENTS 64U

struct fake_sector {
	enum flash_sector_state state;
	struct flash_chunk chunk;
};

struct fake_transfer_start {
	uint32_t log_id;
	uint32_t start_sector;
};

struct fake_sent_sector {
	uint32_t sector;
	enum flash_sector_state state;
	uint32_t magic;
	uint32_t log_id;
	uint32_t sequence;
};

struct fake_flash {
	uint32_t sector_count;
	struct fake_sector sectors[FAKE_FLASH_MAX_SECTORS];
	uint32_t read_count[FAKE_FLASH_MAX_SECTORS];
	uint32_t write_count[FAKE_FLASH_MAX_SECTORS];
	uint32_t erase_count[FAKE_FLASH_MAX_SECTORS];

	struct fake_transfer_start starts[FAKE_FLASH_MAX_EVENTS];
	uint32_t start_count;
	struct fake_sent_sector sent_sectors[FAKE_FLASH_MAX_EVENTS];
	uint32_t sent_sector_count;
	struct flash_transfer_summary summaries[FAKE_FLASH_MAX_EVENTS];
	uint32_t summary_count;

	enum flash_transport_result finish_script[FAKE_FLASH_MAX_EVENTS];
	uint32_t finish_script_length;
	uint32_t finish_script_position;

	int32_t fail_read_sector;
	int32_t fail_write_sector;
	int32_t fail_erase_sector;
};

void fake_flash_init(struct fake_flash *fake, uint32_t sector_count);
void fake_flash_set_data(struct fake_flash *fake, uint32_t sector,
			 uint32_t log_id, uint32_t sequence);
void fake_flash_set_dirty(struct fake_flash *fake, uint32_t sector,
			  uint32_t log_id, uint32_t sequence);
void fake_flash_script_finish(
	struct fake_flash *fake,
	const enum flash_transport_result *responses,
	uint32_t response_count);

extern const struct flash_storage_ops fake_flash_storage_ops;
extern const struct flash_transport_ops fake_flash_transport_ops;

#endif /* SUS_TEST_FAKE_FLASH_H_ */
