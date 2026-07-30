#ifndef SUS_FLASH_SERIAL_H_
#define SUS_FLASH_SERIAL_H_

#include <stdint.h>

#include <zephyr/device.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/atomic.h>

#include "flash.h"

#define FLASH_SERIAL_PROTOCOL_VERSION 1U
#define FLASH_SERIAL_MAX_DECODED_BYTES \
	(12U + 8U + FLASH_LOG_SECTOR_BYTES + 4U)
#define FLASH_SERIAL_MAX_ENCODED_BYTES \
	(FLASH_SERIAL_MAX_DECODED_BYTES + \
	 FLASH_SERIAL_MAX_DECODED_BYTES / 254U + 2U)

enum flash_serial_session_result {
	FLASH_SERIAL_NO_HOST,
	FLASH_SERIAL_SESSION_COMPLETE,
	FLASH_SERIAL_SESSION_ERROR,
};

struct flash_serial_transport {
	const struct device *uart;
	uint32_t sector_count;
	uint32_t token_counter;
	uint32_t active_token;
	uint32_t active_ordinal;
	struct k_sem tx_done;
	const uint8_t *tx_data;
	size_t tx_length;
	size_t tx_offset;
	atomic_t tx_active;
	uint8_t decoded[FLASH_SERIAL_MAX_DECODED_BYTES];
	uint8_t encoded[FLASH_SERIAL_MAX_ENCODED_BYTES];
};

int flash_serial_transport_init(struct flash_serial_transport *transport,
				const struct device *uart,
				uint32_t sector_count);

enum flash_serial_session_result
flash_serial_upload_session(struct flash_serial_transport *transport,
			    struct flash_log *log,
			    uint32_t hello_timeout_ms);

extern const struct flash_transport_ops flash_serial_transport_ops;

#endif /* SUS_FLASH_SERIAL_H_ */
