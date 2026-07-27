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

	assert(sector < fake->sector_count);
	fake->read_count[sector]++;
	if ((int32_t)sector == fake->fail_read_sector) {
		return -1;
	}

	memcpy(chunk, &fake->sectors[sector].chunk, sizeof(*chunk));
	*state = fake->sectors[sector].state;
	return 0;
}

static int fake_write_sector(void *context, uint32_t sector,
			     const struct flash_chunk *chunk)
{
	struct fake_flash *fake = context;

	assert(sector < fake->sector_count);
	fake->write_count[sector]++;
	if ((int32_t)sector == fake->fail_write_sector) {
		return -1;
	}
	if (fake->sectors[sector].state != FLASH_SECTOR_ERASED ||
	    !flash_chunk_is_valid(chunk)) {
		return -1;
	}

	fake->sectors[sector].state = FLASH_SECTOR_VALID;
	memcpy(&fake->sectors[sector].chunk, chunk, sizeof(*chunk));
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

	memset(&fake->sectors[sector].chunk, 0xff,
	       sizeof(fake->sectors[sector].chunk));
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
	sent->payload_length = chunk->payload_length;
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

const struct flash_storage_ops fake_flash_storage_ops = {
	.read_sector = fake_read_sector,
	.write_sector = fake_write_sector,
	.erase_sector = fake_erase_sector,
};

const struct flash_transport_ops fake_flash_transport_ops = {
	.send_log_start = fake_send_log_start,
	.send_chunk = fake_send_chunk,
	.send_log_end = fake_send_log_end,
};

void fake_flash_init(struct fake_flash *fake, uint32_t sector_count)
{
	assert(sector_count <= FAKE_FLASH_MAX_SECTORS);

	memset(fake, 0, sizeof(*fake));
	fake->sector_count = sector_count;
	fake->fail_read_sector = -1;
	fake->fail_write_sector = -1;
	fake->fail_erase_sector = -1;
	for (uint32_t sector = 0; sector < sector_count; sector++) {
		memset(&fake->sectors[sector].chunk, 0xff,
		       sizeof(fake->sectors[sector].chunk));
		fake->sectors[sector].state = FLASH_SECTOR_ERASED;
	}
}

void fake_flash_set_data(struct fake_flash *fake, uint32_t sector,
			 uint32_t log_id, uint32_t sequence)
{
	struct flash_chunk *chunk;

	assert(sector < fake->sector_count);
	chunk = &fake->sectors[sector].chunk;
	memset(chunk, 0xff, sizeof(*chunk));
	chunk->magic = FLASH_LOG_DATA_MAGIC;
	chunk->log_id = log_id;
	chunk->sequence = sequence;
	chunk->payload_length = 1U;
	chunk->payload[0] = (uint8_t)sequence;
	flash_chunk_finalize(chunk);
	fake->sectors[sector].state = FLASH_SECTOR_VALID;
}

void fake_flash_set_dirty(struct fake_flash *fake, uint32_t sector,
			  uint32_t log_id, uint32_t sequence)
{
	fake_flash_set_data(fake, sector, log_id, sequence);
	fake->sectors[sector].chunk.crc ^= UINT32_C(1);
	fake->sectors[sector].state = FLASH_SECTOR_DIRTY;
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
