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
	return transport != NULL && transport->send_log_start != NULL &&
	       transport->send_chunk != NULL && transport->send_log_end != NULL;
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

static uint32_t flash_load_u32_le(const uint8_t input[4])
{
	return (uint32_t)input[0] | ((uint32_t)input[1] << 8U) |
	       ((uint32_t)input[2] << 16U) |
	       ((uint32_t)input[3] << 24U);
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
flash_log_validate_journal(struct flash_log *log,
			   uint32_t valid_sector_count)
{
	uint32_t cursor = log->read_sector;
	uint32_t traversed = 0;
	uint32_t current_log_id = 0;
	uint32_t expected_sequence = 0;
	uint32_t log_crc_state = FLASH_CRC_INITIAL;
	uint32_t current_log_start = cursor;
	bool first_log = true;
	bool committed = false;

	while (cursor != log->write_sector) {
		enum flash_sector_state state;
		enum flash_log_result result =
			flash_log_read_sector(log, cursor, &state);

		if (result != FLASH_LOG_OK) {
			return result;
		}
		if (state != FLASH_SECTOR_VALID ||
		    !flash_chunk_is_valid(log->scratch)) {
			return FLASH_LOG_CORRUPT;
		}

		if (first_log || log->scratch->log_id != current_log_id) {
			if (!first_log && !committed) {
				return FLASH_LOG_CORRUPT;
			}
			current_log_id = log->scratch->log_id;
			current_log_start = cursor;
			expected_sequence = 0;
			log_crc_state = FLASH_CRC_INITIAL;
			committed = false;
			first_log = false;
		}

		if (log->scratch->magic == FLASH_LOG_DATA_MAGIC) {
			if (committed ||
			    log->scratch->sequence != expected_sequence) {
				return FLASH_LOG_CORRUPT;
			}
			log_crc_state = flash_crc32_update(
				log_crc_state, log->scratch->payload,
				log->scratch->payload_length);
			expected_sequence++;
		} else {
			uint32_t committed_crc =
				flash_load_u32_le(log->scratch->payload);

			if (committed ||
			    log->scratch->sequence != expected_sequence ||
			    committed_crc !=
				    flash_crc32_finish(log_crc_state)) {
				return FLASH_LOG_CORRUPT;
			}
			committed = true;
		}

		traversed++;
		if (traversed >= log->sector_count) {
			return FLASH_LOG_CORRUPT;
		}
		cursor = flash_log_next_sector(log, cursor);
	}

	if (traversed != valid_sector_count) {
		return FLASH_LOG_CORRUPT;
	}
	if (!committed) {
		log->incomplete_start_sector = current_log_start;
		log->incomplete_log_id = current_log_id;
	}
	return committed ? FLASH_LOG_OK : FLASH_LOG_INCOMPLETE;
}

static enum flash_log_result
flash_log_locate_committed_log(struct flash_log *log, uint32_t *end_sector,
			       uint32_t *log_id, uint32_t *log_crc)
{
	uint32_t cursor = log->read_sector;
	uint32_t expected_sequence = 0;
	uint32_t crc_state = FLASH_CRC_INITIAL;

	while (cursor != log->write_sector) {
		enum flash_sector_state state;
		enum flash_log_result result =
			flash_log_read_sector(log, cursor, &state);

		if (result != FLASH_LOG_OK) {
			return result;
		}
		if (state != FLASH_SECTOR_VALID ||
		    !flash_chunk_is_valid(log->scratch)) {
			return FLASH_LOG_CORRUPT;
		}

		if (expected_sequence == 0U) {
			*log_id = log->scratch->log_id;
		}
		if (log->scratch->log_id != *log_id) {
			return FLASH_LOG_INCOMPLETE;
		}

		if (log->scratch->magic == FLASH_LOG_DATA_MAGIC) {
			if (log->scratch->sequence != expected_sequence) {
				return FLASH_LOG_CORRUPT;
			}
			crc_state = flash_crc32_update(
				crc_state, log->scratch->payload,
				log->scratch->payload_length);
			expected_sequence++;
			cursor = flash_log_next_sector(log, cursor);
			continue;
		}

		if (log->scratch->sequence != expected_sequence ||
		    flash_load_u32_le(log->scratch->payload) !=
			    flash_crc32_finish(crc_state)) {
			return FLASH_LOG_CORRUPT;
		}

		*log_crc = flash_crc32_finish(crc_state);
		*end_sector = flash_log_next_sector(log, cursor);
		return FLASH_LOG_OK;
	}

	return FLASH_LOG_INCOMPLETE;
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

enum flash_log_result flash_log_scan(struct flash_log *log)
{
	uint32_t oldest_log_id = UINT32_MAX;
	uint32_t oldest_sequence = UINT32_MAX;
	uint32_t newest_log_id = 0;
	uint32_t newest_sequence = 0;
	uint32_t valid_sector_count = 0;
	uint32_t dirty_sector_count = 0;
	uint32_t dirty_sector = 0;

	if (log == NULL || log->scratch == NULL ||
	    !flash_log_has_storage(log->storage)) {
		return FLASH_LOG_INVALID_ARGUMENT;
	}

	log->read_sector = 0;
	log->write_sector = 0;
	log->next_log_id = 0;
	log->write_active = false;
	log->tail_incomplete = false;

	for (uint32_t sector = 0; sector < log->sector_count; sector++) {
		enum flash_sector_state state;
		enum flash_log_result result =
			flash_log_read_sector(log, sector, &state);

		if (result != FLASH_LOG_OK) {
			return result;
		}
		if (state == FLASH_SECTOR_DIRTY) {
			dirty_sector = sector;
			dirty_sector_count++;
			continue;
		}
		if (state == FLASH_SECTOR_ERASED) {
			continue;
		}
		if (!flash_chunk_is_valid(log->scratch)) {
			return FLASH_LOG_CORRUPT;
		}

		if (valid_sector_count == 0U ||
		    flash_chunk_is_older(log->scratch, oldest_log_id,
					 oldest_sequence)) {
			oldest_log_id = log->scratch->log_id;
			oldest_sequence = log->scratch->sequence;
			log->read_sector = sector;
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
	if (dirty_sector_count != 0U) {
		if (dirty_sector_count != 1U ||
		    dirty_sector != log->write_sector) {
			return FLASH_LOG_CORRUPT;
		}
		if (log->storage->erase_sector(log->storage_context,
					       dirty_sector) != 0) {
			return FLASH_LOG_IO_ERROR;
		}
	}
	if (valid_sector_count == 0U) {
		return FLASH_LOG_OK;
	}

	log->next_log_id = newest_log_id + 1U;
	enum flash_log_result result =
		flash_log_validate_journal(log, valid_sector_count);

	if (result == FLASH_LOG_INCOMPLETE) {
		log->tail_incomplete = true;
	}
	return result;
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
	if (log->tail_incomplete) {
		return FLASH_LOG_INCOMPLETE;
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

	if (log == NULL || !flash_log_has_storage(log->storage)) {
		return FLASH_LOG_INVALID_ARGUMENT;
	}
	if (!log->write_active) {
		return FLASH_LOG_BAD_STATE;
	}
	if (flash_log_is_full(log)) {
		return FLASH_LOG_FULL;
	}

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

enum flash_log_result
flash_log_discard_incomplete(struct flash_log *log)
{
	enum flash_log_result result;

	if (log == NULL || !flash_log_has_storage(log->storage)) {
		return FLASH_LOG_INVALID_ARGUMENT;
	}
	if (log->write_active) {
		return FLASH_LOG_BAD_STATE;
	}
	if (!log->tail_incomplete) {
		return FLASH_LOG_BAD_STATE;
	}

	result = flash_log_erase_range(log, log->incomplete_start_sector,
				       log->write_sector);
	if (result != FLASH_LOG_OK) {
		return result;
	}

	log->write_sector = log->incomplete_start_sector;
	log->next_log_id = log->incomplete_log_id;
	log->tail_incomplete = false;
	return FLASH_LOG_OK;
}

enum flash_log_result flash_log_read_one(struct flash_log *log)
{
	uint32_t end_sector;
	uint32_t log_id;
	uint32_t log_crc;
	uint32_t cursor;
	enum flash_log_result result;

	if (log == NULL || log->scratch == NULL ||
	    !flash_log_has_storage(log->storage) ||
	    !flash_log_has_transport(log->transport)) {
		return FLASH_LOG_INVALID_ARGUMENT;
	}
	if (flash_log_is_empty(log)) {
		return FLASH_LOG_EMPTY;
	}

	result = flash_log_locate_committed_log(
		log, &end_sector, &log_id, &log_crc);
	if (result != FLASH_LOG_OK) {
		return result;
	}

	if (log->transport->send_log_start(log->transport_context,
					   log_id) != 0) {
		return FLASH_LOG_TRANSPORT_ERROR;
	}

	cursor = log->read_sector;
	while (cursor != end_sector) {
		enum flash_sector_state state;

		result = flash_log_read_sector(log, cursor, &state);
		if (result != FLASH_LOG_OK) {
			return result;
		}
		if (state != FLASH_SECTOR_VALID ||
		    !flash_chunk_is_valid(log->scratch)) {
			return FLASH_LOG_CORRUPT;
		}
		if (log->scratch->magic == FLASH_LOG_COMMIT_MAGIC) {
			break;
		}
		if (log->transport->send_chunk(log->transport_context,
					       log->scratch) != 0) {
			return FLASH_LOG_TRANSPORT_ERROR;
		}
		cursor = flash_log_next_sector(log, cursor);
	}

	switch (log->transport->send_log_end(log->transport_context,
					     log_id, log_crc)) {
	case FLASH_TRANSPORT_ACK:
		result = flash_log_erase_range(log, log->read_sector,
					       end_sector);
		if (result != FLASH_LOG_OK) {
			return result;
		}
		log->read_sector = end_sector;
		return FLASH_LOG_OK;
	case FLASH_TRANSPORT_DONE:
		return FLASH_LOG_TRANSPORT_DONE;
	case FLASH_TRANSPORT_ERROR:
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
