#include "flash.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define FLASH_CRC_INITIAL UINT32_MAX

_Static_assert(sizeof(struct flash_chunk) == FLASH_LOG_SECTOR_BYTES,
	       "flash_chunk must occupy exactly one flash sector");
_Static_assert(offsetof(struct flash_chunk, payload) == 14U,
	       "flash_chunk header layout changed");
_Static_assert(offsetof(struct flash_chunk, crc) ==
		       FLASH_LOG_SECTOR_BYTES - sizeof(uint32_t),
	       "flash_chunk CRC must be the final word in the sector");

static bool flash_log_has_storage(const struct flash_storage_ops *storage)
{
	return storage != NULL && storage->read_sector != NULL &&
	       storage->write_sector != NULL && storage->erase_sector != NULL;
}

static bool
flash_log_has_transport(const struct flash_transport_ops *transport)
{
	return transport != NULL && transport->begin != NULL &&
	       transport->send_sector != NULL && transport->finish != NULL;
}

static uint32_t flash_crc32_update(uint32_t state, const void *data,
				   size_t length)
{
	const uint8_t *bytes = data;

	for (size_t i = 0; i < length; i++) {
		state ^= bytes[i];
		for (uint8_t bit = 0; bit < 8U; bit++) {
			uint32_t mask = 0U - (state & 1U);

			state = (state >> 1U) ^
				(UINT32_C(0xedb88320) & mask);
		}
	}

	return state;
}

static uint32_t flash_crc32_finish(uint32_t state)
{
	return ~state;
}

static void flash_store_u32_le(uint8_t output[4], uint32_t value)
{
	output[0] = (uint8_t)value;
	output[1] = (uint8_t)(value >> 8U);
	output[2] = (uint8_t)(value >> 16U);
	output[3] = (uint8_t)(value >> 24U);
}

static bool flash_chunk_header_is_valid(const struct flash_chunk *chunk)
{
	if (chunk->magic == FLASH_LOG_DATA_MAGIC) {
		return chunk->payload_length <= FLASH_LOG_PAYLOAD_BYTES;
	}
	if (chunk->magic == FLASH_LOG_COMMIT_MAGIC) {
		return chunk->payload_length ==
		       FLASH_LOG_COMMIT_PAYLOAD_BYTES;
	}
	return false;
}

void flash_chunk_finalize(struct flash_chunk *chunk)
{
	size_t crc_length =
		offsetof(struct flash_chunk, payload) + chunk->payload_length;
	uint32_t state =
		flash_crc32_update(FLASH_CRC_INITIAL, chunk, crc_length);

	chunk->crc = flash_crc32_finish(state);
}

bool flash_chunk_is_valid(const struct flash_chunk *chunk)
{
	size_t crc_length;
	uint32_t state;

	if (chunk == NULL || !flash_chunk_header_is_valid(chunk)) {
		return false;
	}

	crc_length =
		offsetof(struct flash_chunk, payload) + chunk->payload_length;
	state = flash_crc32_update(FLASH_CRC_INITIAL, chunk, crc_length);
	return chunk->crc == flash_crc32_finish(state);
}

static bool flash_chunk_is_older(const struct flash_chunk *candidate,
				 uint32_t oldest_log_id,
				 uint32_t oldest_sequence)
{
	return candidate->log_id < oldest_log_id ||
	       (candidate->log_id == oldest_log_id &&
		candidate->sequence < oldest_sequence);
}

static bool flash_chunk_is_newer(const struct flash_chunk *candidate,
				 uint32_t newest_log_id,
				 uint32_t newest_sequence)
{
	return candidate->log_id > newest_log_id ||
	       (candidate->log_id == newest_log_id &&
		candidate->sequence > newest_sequence);
}

static enum flash_log_result
flash_log_read_sector(struct flash_log *log, uint32_t sector,
		      enum flash_sector_state *state)
{
	if (log->storage->read_sector(log->storage_context, sector,
				      log->scratch, state) != 0) {
		return FLASH_LOG_IO_ERROR;
	}

	return FLASH_LOG_OK;
}

static enum flash_log_result
flash_log_erase_range(struct flash_log *log, uint32_t start, uint32_t end)
{
	uint32_t cursor = start;
	uint32_t traversed = 0;

	while (cursor != end) {
		if (log->storage->erase_sector(log->storage_context,
					      cursor) != 0) {
			return FLASH_LOG_IO_ERROR;
		}

		traversed++;
		if (traversed >= log->sector_count) {
			return FLASH_LOG_CORRUPT;
		}
		cursor = flash_log_next_sector(log, cursor);
	}

	return FLASH_LOG_OK;
}

static enum flash_log_result
flash_log_clean_free_arc(struct flash_log *log)
{
	uint32_t cursor = log->write_sector;
	uint32_t traversed = 0;

	while (cursor != log->read_sector) {
		enum flash_sector_state state;
		enum flash_log_result result =
			flash_log_read_sector(log, cursor, &state);

		if (result != FLASH_LOG_OK) {
			return result;
		}
		if (state == FLASH_SECTOR_VALID &&
		    flash_chunk_is_valid(log->scratch)) {
			/*
			 * A valid sector in the free arc means the recovered
			 * oldest/newest bounds do not describe one ring range.
			 */
			return FLASH_LOG_CORRUPT;
		}
		if (state == FLASH_SECTOR_DIRTY ||
		    (state == FLASH_SECTOR_VALID &&
		     !flash_chunk_is_valid(log->scratch))) {
			if (log->storage->erase_sector(log->storage_context,
						       cursor) != 0) {
				return FLASH_LOG_IO_ERROR;
			}
		}

		traversed++;
		if (traversed >= log->sector_count) {
			return FLASH_LOG_CORRUPT;
		}
		cursor = flash_log_next_sector(log, cursor);
	}

	return FLASH_LOG_OK;
}

static enum flash_log_result
flash_log_clean_empty_media(struct flash_log *log)
{
	for (uint32_t sector = 0; sector < log->sector_count; sector++) {
		enum flash_sector_state state;
		enum flash_log_result result =
			flash_log_read_sector(log, sector, &state);

		if (result != FLASH_LOG_OK) {
			return result;
		}
		if (state != FLASH_SECTOR_ERASED &&
		    log->storage->erase_sector(log->storage_context,
					       sector) != 0) {
			return FLASH_LOG_IO_ERROR;
		}
	}

	return FLASH_LOG_OK;
}

enum flash_log_result flash_log_init(struct flash_log *log,
				     uint32_t sector_count,
				     struct flash_chunk *scratch,
				     const struct flash_storage_ops *storage,
				     void *storage_context,
				     const struct flash_transport_ops *transport,
				     void *transport_context)
{
	if (log == NULL || scratch == NULL || sector_count < 3U ||
	    !flash_log_has_storage(storage)) {
		return FLASH_LOG_INVALID_ARGUMENT;
	}

	memset(log, 0, sizeof(*log));
	log->sector_count = sector_count;
	log->scratch = scratch;
	log->storage = storage;
	log->storage_context = storage_context;
	log->transport = transport;
	log->transport_context = transport_context;
	log->active_crc_state = FLASH_CRC_INITIAL;

	return FLASH_LOG_OK;
}

uint32_t flash_log_next_sector(const struct flash_log *log, uint32_t sector)
{
	return (sector + 1U) % log->sector_count;
}

bool flash_log_is_empty(const struct flash_log *log)
{
	return log->read_sector == log->write_sector;
}

bool flash_log_is_full(const struct flash_log *log)
{
	return flash_log_next_sector(log, log->write_sector) ==
	       log->read_sector;
}

bool flash_log_can_append(const struct flash_log *log)
{
	uint32_t after_data =
		flash_log_next_sector(log, log->write_sector);

	return flash_log_next_sector(log, after_data) != log->read_sector;
}

void flash_log_checkpoint_save(const struct flash_log *log,
			       struct flash_log_checkpoint *checkpoint)
{
	if (log == NULL || checkpoint == NULL) {
		return;
	}

	checkpoint->sector_count = log->sector_count;
	checkpoint->read_sector = log->read_sector;
	checkpoint->write_sector = log->write_sector;
	checkpoint->read_log_id = log->read_log_id;
	checkpoint->next_log_id = log->next_log_id;
}

enum flash_log_result
flash_log_checkpoint_restore(struct flash_log *log,
			     const struct flash_log_checkpoint *checkpoint)
{
	enum flash_log_result result;
	enum flash_sector_state state;
	uint32_t previous_write_sector;

	if (log == NULL || checkpoint == NULL || log->scratch == NULL ||
	    !flash_log_has_storage(log->storage) ||
	    checkpoint->sector_count != log->sector_count ||
	    checkpoint->read_sector >= log->sector_count ||
	    checkpoint->write_sector >= log->sector_count) {
		return FLASH_LOG_INVALID_ARGUMENT;
	}

	/* The reserved write sector must always be ready for the next append. */
	result = flash_log_read_sector(log, checkpoint->write_sector, &state);
	if (result != FLASH_LOG_OK) {
		return result;
	}
	if (state != FLASH_SECTOR_ERASED) {
		return FLASH_LOG_CORRUPT;
	}

	if (checkpoint->read_sector != checkpoint->write_sector) {
		result = flash_log_read_sector(log, checkpoint->read_sector,
					       &state);
		if (result != FLASH_LOG_OK) {
			return result;
		}
		if (state != FLASH_SECTOR_VALID ||
		    log->scratch->log_id != checkpoint->read_log_id) {
			return FLASH_LOG_CORRUPT;
		}

		previous_write_sector = checkpoint->write_sector == 0U ?
					log->sector_count - 1U :
					checkpoint->write_sector - 1U;
		result = flash_log_read_sector(log, previous_write_sector,
					       &state);
		if (result != FLASH_LOG_OK) {
			return result;
		}
		if (state != FLASH_SECTOR_VALID ||
		    log->scratch->magic != FLASH_LOG_COMMIT_MAGIC ||
		    log->scratch->log_id + 1U != checkpoint->next_log_id) {
			return FLASH_LOG_CORRUPT;
		}
	}

	log->read_sector = checkpoint->read_sector;
	log->write_sector = checkpoint->write_sector;
	log->read_log_id = checkpoint->read_log_id;
	log->next_log_id = checkpoint->next_log_id;
	log->write_active = false;
	log->active_crc_state = FLASH_CRC_INITIAL;
	return FLASH_LOG_OK;
}

enum flash_log_result flash_log_scan(struct flash_log *log)
{
	uint32_t oldest_log_id = UINT32_MAX;
	uint32_t oldest_sequence = UINT32_MAX;
	uint32_t newest_log_id = 0;
	uint32_t newest_sequence = 0;
	uint32_t valid_sector_count = 0;

	if (log == NULL || log->scratch == NULL ||
	    !flash_log_has_storage(log->storage)) {
		return FLASH_LOG_INVALID_ARGUMENT;
	}

	log->read_sector = 0;
	log->write_sector = 0;
	log->read_log_id = 0;
	log->next_log_id = 0;
	log->write_active = false;

	for (uint32_t sector = 0; sector < log->sector_count; sector++) {
		enum flash_sector_state state;
		enum flash_log_result result =
			flash_log_read_sector(log, sector, &state);

		if (result != FLASH_LOG_OK) {
			return result;
		}
		if (state != FLASH_SECTOR_VALID ||
		    !flash_chunk_is_valid(log->scratch)) {
			continue;
		}

		if (valid_sector_count == 0U ||
		    flash_chunk_is_older(log->scratch, oldest_log_id,
					 oldest_sequence)) {
			oldest_log_id = log->scratch->log_id;
			oldest_sequence = log->scratch->sequence;
			log->read_sector = sector;
			log->read_log_id = log->scratch->log_id;
		}
		if (valid_sector_count == 0U ||
		    flash_chunk_is_newer(log->scratch, newest_log_id,
					 newest_sequence)) {
			newest_log_id = log->scratch->log_id;
			newest_sequence = log->scratch->sequence;
			log->write_sector = flash_log_next_sector(log, sector);
		}
		valid_sector_count++;
	}

	if (valid_sector_count >= log->sector_count) {
		return FLASH_LOG_CORRUPT;
	}
	if (valid_sector_count == 0U) {
		return flash_log_clean_empty_media(log);
	}

	log->next_log_id = newest_log_id + 1U;
	return flash_log_clean_free_arc(log);
}

enum flash_log_result flash_log_begin(struct flash_log *log,
				      uint32_t *log_id)
{
	if (log == NULL || !flash_log_has_storage(log->storage)) {
		return FLASH_LOG_INVALID_ARGUMENT;
	}
	if (log->write_active) {
		return FLASH_LOG_BAD_STATE;
	}
	if (flash_log_is_full(log)) {
		return FLASH_LOG_FULL;
	}

	log->write_active = true;
	log->active_log_id = log->next_log_id++;
	log->active_sequence = 0;
	log->active_start_sector = log->write_sector;
	log->active_crc_state = FLASH_CRC_INITIAL;
	if (log_id != NULL) {
		*log_id = log->active_log_id;
	}

	return FLASH_LOG_OK;
}

enum flash_log_result flash_log_append(struct flash_log *log,
				       const void *payload,
				       size_t payload_length)
{
	bool was_empty;

	if (log == NULL || payload == NULL ||
	    payload_length > FLASH_LOG_PAYLOAD_BYTES) {
		return FLASH_LOG_INVALID_ARGUMENT;
	}
	if (!log->write_active) {
		return FLASH_LOG_BAD_STATE;
	}
	if (!flash_log_can_append(log)) {
		return FLASH_LOG_FULL;
	}
	if (log->active_sequence == UINT32_MAX) {
		return FLASH_LOG_FULL;
	}

	was_empty = flash_log_is_empty(log);
	memset(log->scratch, 0xff, sizeof(*log->scratch));
	log->scratch->magic = FLASH_LOG_DATA_MAGIC;
	log->scratch->log_id = log->active_log_id;
	log->scratch->sequence = log->active_sequence;
	log->scratch->payload_length = (uint16_t)payload_length;
	memcpy(log->scratch->payload, payload, payload_length);
	flash_chunk_finalize(log->scratch);

	if (log->storage->write_sector(log->storage_context,
				       log->write_sector,
				       log->scratch) != 0) {
		return FLASH_LOG_IO_ERROR;
	}

	if (was_empty) {
		log->read_log_id = log->active_log_id;
	}
	log->active_crc_state = flash_crc32_update(
		log->active_crc_state, payload, payload_length);
	log->active_sequence++;
	log->write_sector =
		flash_log_next_sector(log, log->write_sector);
	return FLASH_LOG_OK;
}

enum flash_log_result flash_log_close(struct flash_log *log)
{
	uint32_t log_crc;
	bool was_empty;

	if (log == NULL || !flash_log_has_storage(log->storage)) {
		return FLASH_LOG_INVALID_ARGUMENT;
	}
	if (!log->write_active) {
		return FLASH_LOG_BAD_STATE;
	}
	if (flash_log_is_full(log)) {
		return FLASH_LOG_FULL;
	}

	was_empty = flash_log_is_empty(log);
	log_crc = flash_crc32_finish(log->active_crc_state);
	memset(log->scratch, 0xff, sizeof(*log->scratch));
	log->scratch->magic = FLASH_LOG_COMMIT_MAGIC;
	log->scratch->log_id = log->active_log_id;
	log->scratch->sequence = log->active_sequence;
	log->scratch->payload_length = FLASH_LOG_COMMIT_PAYLOAD_BYTES;
	flash_store_u32_le(log->scratch->payload, log_crc);
	flash_chunk_finalize(log->scratch);

	if (log->storage->write_sector(log->storage_context,
				       log->write_sector,
				       log->scratch) != 0) {
		return FLASH_LOG_IO_ERROR;
	}

	if (was_empty) {
		log->read_log_id = log->active_log_id;
	}
	log->write_sector =
		flash_log_next_sector(log, log->write_sector);
	log->write_active = false;
	return FLASH_LOG_OK;
}

enum flash_log_result flash_log_abort(struct flash_log *log)
{
	enum flash_log_result result;

	if (log == NULL || !flash_log_has_storage(log->storage)) {
		return FLASH_LOG_INVALID_ARGUMENT;
	}
	if (!log->write_active) {
		return FLASH_LOG_BAD_STATE;
	}

	result = flash_log_erase_range(log, log->active_start_sector,
				       log->write_sector);
	if (result != FLASH_LOG_OK) {
		return result;
	}

	log->write_sector = log->active_start_sector;
	log->next_log_id = log->active_log_id;
	log->write_active = false;
	log->active_crc_state = FLASH_CRC_INITIAL;
	return FLASH_LOG_OK;
}

enum flash_log_result flash_log_read_one(struct flash_log *log)
{
	struct flash_transfer_summary summary;
	uint32_t cursor;
	uint32_t write_snapshot;
	uint32_t raw_crc_state = FLASH_CRC_INITIAL;
	uint32_t next_read_log_id = log != NULL ? log->read_log_id : 0U;
	bool next_log_found = false;

	if (log == NULL || log->scratch == NULL ||
	    !flash_log_has_storage(log->storage) ||
	    !flash_log_has_transport(log->transport)) {
		return FLASH_LOG_INVALID_ARGUMENT;
	}
	if (log->write_active) {
		return FLASH_LOG_BAD_STATE;
	}
	if (flash_log_is_empty(log)) {
		return FLASH_LOG_EMPTY;
	}

	cursor = log->read_sector;
	write_snapshot = log->write_sector;
	memset(&summary, 0, sizeof(summary));
	summary.log_id = log->read_log_id;
	summary.start_sector = cursor;

	if (log->transport->begin(log->transport_context, summary.log_id,
				  summary.start_sector) != 0) {
		return FLASH_LOG_TRANSPORT_ERROR;
	}

	while (cursor != write_snapshot) {
		enum flash_sector_state state;
		enum flash_log_result result =
			flash_log_read_sector(log, cursor, &state);

		if (result != FLASH_LOG_OK) {
			return result;
		}
		if (state == FLASH_SECTOR_VALID &&
		    flash_chunk_is_valid(log->scratch) &&
		    log->scratch->log_id != summary.log_id) {
			next_read_log_id = log->scratch->log_id;
			next_log_found = true;
			break;
		}

		if (log->transport->send_sector(log->transport_context,
						cursor, log->scratch) != 0) {
			return FLASH_LOG_TRANSPORT_ERROR;
		}
		raw_crc_state = flash_crc32_update(
			raw_crc_state, log->scratch, sizeof(*log->scratch));
		summary.sector_count++;
		cursor = flash_log_next_sector(log, cursor);
	}

	summary.end_sector = cursor;
	summary.raw_crc = flash_crc32_finish(raw_crc_state);

	switch (log->transport->finish(log->transport_context, &summary)) {
	case FLASH_TRANSPORT_ERASE: {
		enum flash_log_result result =
			flash_log_erase_range(log, summary.start_sector,
					      summary.end_sector);

		if (result != FLASH_LOG_OK) {
			return result;
		}
		log->read_sector = summary.end_sector;
		if (next_log_found) {
			log->read_log_id = next_read_log_id;
		}
		return FLASH_LOG_OK;
	}
	case FLASH_TRANSPORT_DONE:
		return FLASH_LOG_TRANSPORT_DONE;
	case FLASH_TRANSPORT_RETRY:
	default:
		return FLASH_LOG_TRANSPORT_ERROR;
	}
}

enum flash_log_result flash_log_drain(struct flash_log *log,
				      uint8_t max_retries)
{
	uint8_t retries = 0;

	if (log == NULL) {
		return FLASH_LOG_INVALID_ARGUMENT;
	}

	while (!flash_log_is_empty(log)) {
		enum flash_log_result result = flash_log_read_one(log);

		if (result == FLASH_LOG_OK) {
			retries = 0;
			continue;
		}
		if (result != FLASH_LOG_TRANSPORT_ERROR) {
			return result;
		}
		if (retries >= max_retries) {
			return result;
		}
		retries++;
	}

	return FLASH_LOG_OK;
}
