#ifndef SUS_FLASH_H_
#define SUS_FLASH_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define FLASH_LOG_SECTOR_BYTES 4096U
#define FLASH_LOG_METADATA_BYTES 18U
#define FLASH_LOG_PAYLOAD_BYTES \
	(FLASH_LOG_SECTOR_BYTES - FLASH_LOG_METADATA_BYTES)

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
	FLASH_TRANSPORT_ACK,
	FLASH_TRANSPORT_ERROR,
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
	FLASH_LOG_INVALID_ARGUMENT,
};

struct flash_log_ops {
	int (*read_sector)(void *context, uint32_t sector,
			   struct flash_chunk *chunk,
			   enum flash_sector_state *state);
	int (*erase_sector)(void *context, uint32_t sector);
	int (*send_log_start)(void *context, uint32_t log_id);
	int (*send_chunk)(void *context, const struct flash_chunk *chunk);
	enum flash_transport_result (*send_log_end)(void *context,
						    uint32_t log_id,
						    uint32_t log_crc);
	uint32_t (*update_log_crc)(void *context, uint32_t log_crc,
				   const struct flash_chunk *chunk);
};

struct flash_log {
	uint32_t read_sector;
	uint32_t write_sector;
	uint32_t next_log_id;
	uint32_t sector_count;
	struct flash_chunk *scratch;
	const struct flash_log_ops *ops;
	void *ops_context;
};

enum flash_log_result flash_log_init(struct flash_log *log,
				     uint32_t sector_count,
				     struct flash_chunk *scratch,
				     const struct flash_log_ops *ops,
				     void *ops_context);

uint32_t flash_log_next_sector(const struct flash_log *log, uint32_t sector);
bool flash_log_is_empty(const struct flash_log *log);
bool flash_log_is_full(const struct flash_log *log);

enum flash_log_result flash_log_scan(struct flash_log *log);
enum flash_log_result flash_log_read_one(struct flash_log *log);
enum flash_log_result flash_log_drain(struct flash_log *log,
				      uint8_t max_retries);

#endif /* SUS_FLASH_H_ */
