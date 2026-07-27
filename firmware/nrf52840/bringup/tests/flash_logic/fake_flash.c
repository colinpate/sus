#include "fake_flash.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static int fake_read_sector(void *context, uint32_t sector,
			    struct flash_chunk *chunk,
			    enum flash_sector_state *state)
{
	struct fake_flash *fake = context;
	const struct fake_sector *source;

	assert(sector < fake->sector_count);
	fake->read_count[sector]++;
	if ((int32_t)sector == fake->fail_read_sector) {
		return -1;
	}

	source = &fake->sectors[sector];
	memset(chunk, 0, sizeof(*chunk));
	chunk->log_id = source->log_id;
	chunk->sequence = source->sequence;
	chunk->payload_length = 1U;
	chunk->payload[0] = (uint8_t)source->sequence;
	*state = source->state;

	return 0;
}

static int fake_erase_sector(void *context, uint32_t sector)
{
	struct fake_flash *fake = context;

	assert(sector < fake->sector_count);
	fake->erase_count[sector]++;
	if ((int32_t)sector == fake->fail_erase_sector) {
		return -1;
	}

	memset(&fake->sectors[sector], 0, sizeof(fake->sectors[sector]));
	fake->sectors[sector].state = FLASH_SECTOR_ERASED;
	return 0;
}

static int fake_send_log_start(void *context, uint32_t log_id)
{
	struct fake_flash *fake = context;

	assert(fake->sent_log_start_count < FAKE_FLASH_MAX_EVENTS);
	fake->sent_log_starts[fake->sent_log_start_count++] = log_id;
	return 0;
}

static int fake_send_chunk(void *context, const struct flash_chunk *chunk)
{
	struct fake_flash *fake = context;
	struct fake_sent_chunk *sent;

	assert(fake->sent_chunk_count < FAKE_FLASH_MAX_EVENTS);
	sent = &fake->sent_chunks[fake->sent_chunk_count++];
	sent->log_id = chunk->log_id;
	sent->sequence = chunk->sequence;
	return 0;
}

static enum flash_transport_result
fake_send_log_end(void *context, uint32_t log_id, uint32_t log_crc)
{
	struct fake_flash *fake = context;
	enum flash_transport_result result = FLASH_TRANSPORT_ACK;

	assert(fake->sent_log_end_count < FAKE_FLASH_MAX_EVENTS);
	fake->sent_log_ends[fake->sent_log_end_count] = log_id;
	fake->sent_log_crcs[fake->sent_log_end_count] = log_crc;
	fake->sent_log_end_count++;

	if (fake->end_script_position < fake->end_script_length) {
		result = fake->end_script[fake->end_script_position++];
	}
	return result;
}

static uint32_t fake_update_log_crc(void *context, uint32_t log_crc,
				    const struct flash_chunk *chunk)
{
	(void)context;
	return (log_crc * 33U) ^ chunk->log_id ^ chunk->sequence;
}

const struct flash_log_ops fake_flash_ops = {
	.read_sector = fake_read_sector,
	.erase_sector = fake_erase_sector,
	.send_log_start = fake_send_log_start,
	.send_chunk = fake_send_chunk,
	.send_log_end = fake_send_log_end,
	.update_log_crc = fake_update_log_crc,
};

void fake_flash_init(struct fake_flash *fake, uint32_t sector_count)
{
	assert(sector_count <= FAKE_FLASH_MAX_SECTORS);

	memset(fake, 0, sizeof(*fake));
	fake->sector_count = sector_count;
	fake->fail_read_sector = -1;
	fake->fail_erase_sector = -1;
	for (uint32_t sector = 0; sector < sector_count; sector++) {
		fake->sectors[sector].state = FLASH_SECTOR_ERASED;
	}
}

void fake_flash_set_valid(struct fake_flash *fake, uint32_t sector,
			  uint32_t log_id, uint32_t sequence)
{
	assert(sector < fake->sector_count);
	fake->sectors[sector].state = FLASH_SECTOR_VALID;
	fake->sectors[sector].log_id = log_id;
	fake->sectors[sector].sequence = sequence;
}

void fake_flash_set_dirty(struct fake_flash *fake, uint32_t sector,
			  uint32_t log_id, uint32_t sequence)
{
	assert(sector < fake->sector_count);
	fake->sectors[sector].state = FLASH_SECTOR_DIRTY;
	fake->sectors[sector].log_id = log_id;
	fake->sectors[sector].sequence = sequence;
}

void fake_flash_script_log_end(struct fake_flash *fake,
			       const enum flash_transport_result *responses,
			       uint32_t response_count)
{
	assert(response_count <= FAKE_FLASH_MAX_EVENTS);
	memcpy(fake->end_script, responses,
	       response_count * sizeof(fake->end_script[0]));
	fake->end_script_length = response_count;
	fake->end_script_position = 0;
}
