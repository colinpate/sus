#include "flash.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

_Static_assert(sizeof(struct flash_chunk) == FLASH_LOG_SECTOR_BYTES,
	       "flash_chunk must occupy exactly one flash sector");
_Static_assert(offsetof(struct flash_chunk, crc) ==
		       FLASH_LOG_SECTOR_BYTES - sizeof(uint32_t),
	       "flash_chunk CRC must be the final word in the sector");

static bool flash_log_has_required_ops(const struct flash_log_ops *ops)
{
	return ops != NULL && ops->read_sector != NULL &&
	       ops->erase_sector != NULL && ops->send_log_start != NULL &&
	       ops->send_chunk != NULL && ops->send_log_end != NULL;
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
	if (log->ops->read_sector(log->ops_context, sector, log->scratch,
				  state) != 0) {
		return FLASH_LOG_IO_ERROR;
	}

	return FLASH_LOG_OK;
}

static enum flash_log_result
flash_log_validate_contiguous_range(struct flash_log *log,
				    uint32_t valid_sector_count)
{
	uint32_t cursor = log->read_sector;
	uint32_t traversed = 0;

	while (cursor != log->write_sector) {
		enum flash_sector_state state;
		enum flash_log_result result =
			flash_log_read_sector(log, cursor, &state);

		if (result != FLASH_LOG_OK) {
			return result;
		}
		if (state != FLASH_SECTOR_VALID) {
			return FLASH_LOG_CORRUPT;
		}

		traversed++;
		if (traversed >= log->sector_count) {
			return FLASH_LOG_CORRUPT;
		}
		cursor = flash_log_next_sector(log, cursor);
	}

	return traversed == valid_sector_count ? FLASH_LOG_OK :
						 FLASH_LOG_CORRUPT;
}

static enum flash_log_result
flash_log_erase_range(struct flash_log *log, uint32_t start, uint32_t end)
{
	uint32_t cursor = start;
	uint32_t traversed = 0;

	while (cursor != end) {
		if (log->ops->erase_sector(log->ops_context, cursor) != 0) {
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

enum flash_log_result flash_log_init(struct flash_log *log,
				     uint32_t sector_count,
				     struct flash_chunk *scratch,
				     const struct flash_log_ops *ops,
				     void *ops_context)
{
	if (log == NULL || scratch == NULL || sector_count < 2U ||
	    !flash_log_has_required_ops(ops)) {
		return FLASH_LOG_INVALID_ARGUMENT;
	}

	memset(log, 0, sizeof(*log));
	log->sector_count = sector_count;
	log->scratch = scratch;
	log->ops = ops;
	log->ops_context = ops_context;

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

enum flash_log_result flash_log_scan(struct flash_log *log)
{
	uint32_t oldest_log_id = UINT32_MAX;
	uint32_t oldest_sequence = UINT32_MAX;
	uint32_t newest_log_id = 0;
	uint32_t newest_sequence = 0;
	uint32_t valid_sector_count = 0;
	bool dirty_found = false;

	if (log == NULL || log->scratch == NULL ||
	    !flash_log_has_required_ops(log->ops)) {
		return FLASH_LOG_INVALID_ARGUMENT;
	}

	log->read_sector = 0;
	log->write_sector = 0;
	log->next_log_id = 0;

	for (uint32_t sector = 0; sector < log->sector_count; sector++) {
		enum flash_sector_state state;
		enum flash_log_result result =
			flash_log_read_sector(log, sector, &state);

		if (result != FLASH_LOG_OK) {
			return result;
		}
		if (state == FLASH_SECTOR_DIRTY) {
			dirty_found = true;
			continue;
		}
		if (state == FLASH_SECTOR_ERASED) {
			continue;
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

	if (dirty_found || valid_sector_count >= log->sector_count) {
		return FLASH_LOG_CORRUPT;
	}
	if (valid_sector_count == 0U) {
		return FLASH_LOG_OK;
	}

	log->next_log_id = newest_log_id + 1U;
	return flash_log_validate_contiguous_range(log, valid_sector_count);
}

enum flash_log_result flash_log_read_one(struct flash_log *log)
{
	uint32_t cursor;
	uint32_t log_crc = 0;
	uint32_t read_log_id = 0;
	bool first_chunk = true;

	if (log == NULL || log->scratch == NULL ||
	    !flash_log_has_required_ops(log->ops)) {
		return FLASH_LOG_INVALID_ARGUMENT;
	}
	if (flash_log_is_empty(log)) {
		return FLASH_LOG_EMPTY;
	}

	cursor = log->read_sector;
	while (cursor != log->write_sector) {
		enum flash_sector_state state;
		enum flash_log_result result =
			flash_log_read_sector(log, cursor, &state);

		if (result != FLASH_LOG_OK) {
			return result;
		}
		if (state != FLASH_SECTOR_VALID) {
			return FLASH_LOG_CORRUPT;
		}

		if (first_chunk) {
			read_log_id = log->scratch->log_id;
			if (log->ops->send_log_start(log->ops_context,
						     read_log_id) != 0) {
				return FLASH_LOG_TRANSPORT_ERROR;
			}
			first_chunk = false;
		} else if (log->scratch->log_id != read_log_id) {
			break;
		}

		if (log->ops->send_chunk(log->ops_context, log->scratch) != 0) {
			return FLASH_LOG_TRANSPORT_ERROR;
		}
		if (log->ops->update_log_crc != NULL) {
			log_crc = log->ops->update_log_crc(
				log->ops_context, log_crc, log->scratch);
		}
		cursor = flash_log_next_sector(log, cursor);
	}

	switch (log->ops->send_log_end(log->ops_context, read_log_id,
				       log_crc)) {
	case FLASH_TRANSPORT_ACK: {
		enum flash_log_result result =
			flash_log_erase_range(log, log->read_sector, cursor);

		if (result != FLASH_LOG_OK) {
			return result;
		}
		log->read_sector = cursor;
		return FLASH_LOG_OK;
	}
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
