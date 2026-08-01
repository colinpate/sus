#ifndef SUS_FLASH_H_
#define SUS_FLASH_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define FLASH_LOG_SECTOR_BYTES 4096U
#define FLASH_LOG_METADATA_BYTES 18U
#define FLASH_LOG_PAYLOAD_BYTES \
	(FLASH_LOG_SECTOR_BYTES - FLASH_LOG_METADATA_BYTES)
#define FLASH_LOG_DATA_MAGIC UINT32_C(0x44535553)
#define FLASH_LOG_COMMIT_MAGIC UINT32_C(0x45535553)
#define FLASH_LOG_COMMIT_PAYLOAD_BYTES 4U

struct flash_chunk {
	uint32_t magic;
	uint32_t log_id;
	uint32_t sequence;
	uint16_t payload_length;
	uint8_t payload[FLASH_LOG_PAYLOAD_BYTES];
	uint32_t crc;
};

enum flash_sector_state {
	FLASH_SECTOR_VALID,
	FLASH_SECTOR_ERASED,
	FLASH_SECTOR_DIRTY,
};

enum flash_transport_result {
	/* The host durably accepted the exact reported range. */
	FLASH_TRANSPORT_ERASE,
	/* Keep the range and send it again. */
	FLASH_TRANSPORT_RETRY,
	/* Stop sending and keep the range. */
	FLASH_TRANSPORT_DONE,
};

enum flash_log_result {
	FLASH_LOG_OK,
	FLASH_LOG_EMPTY,
	FLASH_LOG_FULL,
	FLASH_LOG_CORRUPT,
	FLASH_LOG_IO_ERROR,
	FLASH_LOG_TRANSPORT_ERROR,
	FLASH_LOG_TRANSPORT_DONE,
	FLASH_LOG_BAD_STATE,
	FLASH_LOG_INVALID_ARGUMENT,
};

enum flash_log_scan_phase {
	FLASH_LOG_SCAN_DISCOVER,
	FLASH_LOG_SCAN_CLEANUP,
	FLASH_LOG_SCAN_CLEANUP_ERASE,
};

typedef void (*flash_log_scan_progress_fn)(
	void *context, enum flash_log_scan_phase phase,
	uint32_t completed_sectors, uint32_t total_sectors);

struct flash_transfer_summary {
	uint32_t log_id;
	uint32_t start_sector;
	/* Half-open range: end_sector is not part of the transfer. */
	uint32_t end_sector;
	uint32_t sector_count;
	/* CRC-32 over all raw 4 KiB sectors in transfer order. */
	uint32_t raw_crc;
};

struct flash_storage_ops {
	int (*read_sector)(void *context, uint32_t sector,
			   struct flash_chunk *chunk,
			   enum flash_sector_state *state);
	int (*write_sector)(void *context, uint32_t sector,
			    const struct flash_chunk *chunk);
	int (*erase_sector)(void *context, uint32_t sector);
};

struct flash_transport_ops {
	int (*begin)(void *context, uint32_t log_id,
		     uint32_t start_sector);
	int (*send_sector)(void *context, uint32_t sector,
			   const struct flash_chunk *raw_sector);
	enum flash_transport_result
		(*finish)(void *context,
			  const struct flash_transfer_summary *summary);
};

struct flash_log {
	uint32_t read_sector;
	uint32_t write_sector;
	uint32_t read_log_id;
	uint32_t next_log_id;
	uint32_t sector_count;
	struct flash_chunk *scratch;
	const struct flash_storage_ops *storage;
	void *storage_context;
	const struct flash_transport_ops *transport;
	void *transport_context;

	bool write_active;
	uint32_t active_log_id;
	uint32_t active_sequence;
	uint32_t active_start_sector;
	uint32_t active_crc_state;
};

struct flash_log_checkpoint {
	uint32_t sector_count;
	uint32_t read_sector;
	uint32_t write_sector;
	uint32_t read_log_id;
	uint32_t next_log_id;
};

enum flash_log_result flash_log_init(struct flash_log *log,
				     uint32_t sector_count,
				     struct flash_chunk *scratch,
				     const struct flash_storage_ops *storage,
				     void *storage_context,
				     const struct flash_transport_ops *transport,
				     void *transport_context);

uint32_t flash_log_next_sector(const struct flash_log *log, uint32_t sector);
bool flash_log_is_empty(const struct flash_log *log);
bool flash_log_is_full(const struct flash_log *log);
bool flash_log_can_append(const struct flash_log *log);

void flash_log_checkpoint_save(const struct flash_log *log,
			       struct flash_log_checkpoint *checkpoint);
enum flash_log_result
flash_log_checkpoint_restore(struct flash_log *log,
			     const struct flash_log_checkpoint *checkpoint);

void flash_chunk_finalize(struct flash_chunk *chunk);
bool flash_chunk_is_valid(const struct flash_chunk *chunk);

enum flash_log_result flash_log_scan(struct flash_log *log);
enum flash_log_result flash_log_scan_with_progress(
	struct flash_log *log, flash_log_scan_progress_fn progress,
	void *progress_context);
enum flash_log_result flash_log_begin(struct flash_log *log,
				      uint32_t *log_id);
enum flash_log_result flash_log_append(struct flash_log *log,
				       const void *payload,
				       size_t payload_length);
enum flash_log_result flash_log_close(struct flash_log *log);
enum flash_log_result flash_log_abort(struct flash_log *log);
enum flash_log_result flash_log_read_one(struct flash_log *log);
enum flash_log_result flash_log_drain(struct flash_log *log,
				      uint8_t max_retries);

#endif /* SUS_FLASH_H_ */
