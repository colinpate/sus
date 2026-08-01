#include "flash_serial.h"

#include <errno.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <zephyr/drivers/uart.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/crc.h>

#define FLASH_SERIAL_MAGIC UINT32_C(0x50535553)
#define FLASH_SERIAL_HEADER_BYTES 12U
#define FLASH_SERIAL_CRC_BYTES 4U
#define FLASH_SERIAL_FRAME_TIMEOUT_MS 10000U

enum flash_serial_message {
	FLASH_SERIAL_HELLO = 1,
	FLASH_SERIAL_INFO = 2,
	FLASH_SERIAL_READ_NEXT = 3,
	FLASH_SERIAL_BEGIN = 4,
	FLASH_SERIAL_SECTOR = 5,
	FLASH_SERIAL_SECTOR_ACK = 6,
	FLASH_SERIAL_END = 7,
	FLASH_SERIAL_DISPOSITION = 8,
	FLASH_SERIAL_EMPTY = 9,
	FLASH_SERIAL_SESSION_DONE = 10,
	FLASH_SERIAL_ERROR = 11,
};

enum flash_serial_disposition {
	FLASH_SERIAL_DISPOSITION_ERASE = 1,
	FLASH_SERIAL_DISPOSITION_RETRY = 2,
	FLASH_SERIAL_DISPOSITION_DONE = 3,
};

struct flash_serial_frame {
	uint8_t type;
	uint32_t token;
	const uint8_t *payload;
	uint16_t payload_length;
};

static void store_u16_le(uint8_t output[2], uint16_t value)
{
	output[0] = (uint8_t)value;
	output[1] = (uint8_t)(value >> 8U);
}

static void store_u32_le(uint8_t output[4], uint32_t value)
{
	output[0] = (uint8_t)value;
	output[1] = (uint8_t)(value >> 8U);
	output[2] = (uint8_t)(value >> 16U);
	output[3] = (uint8_t)(value >> 24U);
}

static uint16_t load_u16_le(const uint8_t input[2])
{
	return (uint16_t)input[0] | ((uint16_t)input[1] << 8U);
}

static uint32_t load_u32_le(const uint8_t input[4])
{
	return (uint32_t)input[0] | ((uint32_t)input[1] << 8U) |
	       ((uint32_t)input[2] << 16U) |
	       ((uint32_t)input[3] << 24U);
}

static size_t cobs_encode(const uint8_t *input, size_t input_length,
			  uint8_t *output, size_t output_capacity)
{
	size_t read_index = 0;
	size_t write_index = 1;
	size_t code_index = 0;
	uint8_t code = 1;

	if (output_capacity == 0U) {
		return 0;
	}

	while (read_index < input_length) {
		if (input[read_index] == 0U) {
			if (code_index >= output_capacity) {
				return 0;
			}
			output[code_index] = code;
			code = 1;
			code_index = write_index++;
			if (write_index > output_capacity) {
				return 0;
			}
			read_index++;
			continue;
		}

		if (write_index >= output_capacity) {
			return 0;
		}
		output[write_index++] = input[read_index++];
		code++;
		if (code == UINT8_MAX) {
			if (code_index >= output_capacity) {
				return 0;
			}
			output[code_index] = code;
			code = 1;
			code_index = write_index++;
			if (write_index > output_capacity) {
				return 0;
			}
		}
	}

	if (code_index >= output_capacity) {
		return 0;
	}
	output[code_index] = code;
	return write_index;
}

static size_t cobs_decode_in_place(uint8_t *buffer, size_t encoded_length)
{
	size_t read_index = 0;
	size_t write_index = 0;

	while (read_index < encoded_length) {
		uint8_t code = buffer[read_index++];
		size_t copy_length;

		if (code == 0U) {
			return 0;
		}
		copy_length = (size_t)code - 1U;
		if (read_index + copy_length > encoded_length) {
			return 0;
		}
		for (size_t i = 0; i < copy_length; i++) {
			buffer[write_index++] = buffer[read_index++];
		}
		if (code != UINT8_MAX && read_index < encoded_length) {
			buffer[write_index++] = 0U;
		}
	}

	return write_index;
}

static void serial_uart_callback(const struct device *uart, void *user_data)
{
	struct flash_serial_transport *transport = user_data;

	if (uart_irq_update(uart) <= 0 || !uart_irq_tx_ready(uart) ||
	    atomic_get(&transport->tx_active) == 0) {
		return;
	}

	while (transport->tx_offset < transport->tx_length) {
		size_t remaining =
			transport->tx_length - transport->tx_offset;
		int written = uart_fifo_fill(
			uart, &transport->tx_data[transport->tx_offset],
			(int)remaining);

		if (written <= 0) {
			break;
		}
		transport->tx_offset += (size_t)written;
	}

	if (transport->tx_offset == transport->tx_length &&
	    atomic_cas(&transport->tx_active, 1, 0)) {
		uart_irq_tx_disable(uart);
		k_sem_give(&transport->tx_done);
	}
}

static int send_encoded(struct flash_serial_transport *transport,
			size_t encoded_length)
{
	k_sem_reset(&transport->tx_done);
	transport->tx_data = transport->encoded;
	transport->tx_length = encoded_length;
	transport->tx_offset = 0;
	atomic_set(&transport->tx_active, 1);
	uart_irq_tx_enable(transport->uart);

	if (k_sem_take(&transport->tx_done,
		       K_MSEC(FLASH_SERIAL_FRAME_TIMEOUT_MS)) != 0) {
		atomic_clear(&transport->tx_active);
		uart_irq_tx_disable(transport->uart);
		return -ETIMEDOUT;
	}
	return 0;
}

static int send_frame(struct flash_serial_transport *transport,
		      uint8_t type, uint32_t token,
		      const void *payload, uint16_t payload_length)
{
	size_t decoded_length = FLASH_SERIAL_HEADER_BYTES +
				(size_t)payload_length +
				FLASH_SERIAL_CRC_BYTES;
	size_t encoded_length;
	uint32_t crc;

	if (transport == NULL || transport->uart == NULL ||
	    decoded_length > sizeof(transport->decoded) ||
	    (payload_length != 0U && payload == NULL)) {
		return -EINVAL;
	}

	store_u32_le(&transport->decoded[0], FLASH_SERIAL_MAGIC);
	transport->decoded[4] = FLASH_SERIAL_PROTOCOL_VERSION;
	transport->decoded[5] = type;
	store_u16_le(&transport->decoded[6], payload_length);
	store_u32_le(&transport->decoded[8], token);
	if (payload_length != 0U) {
		memcpy(&transport->decoded[FLASH_SERIAL_HEADER_BYTES],
		       payload, payload_length);
	}
	crc = crc32_ieee(transport->decoded,
			 decoded_length - FLASH_SERIAL_CRC_BYTES);
	store_u32_le(&transport->decoded[
			     decoded_length - FLASH_SERIAL_CRC_BYTES],
		     crc);

	transport->encoded[0] = 0U;
	encoded_length = cobs_encode(
		transport->decoded, decoded_length,
		&transport->encoded[1],
		sizeof(transport->encoded) - 2U);
	if (encoded_length == 0U) {
		return -EMSGSIZE;
	}
	transport->encoded[encoded_length + 1U] = 0U;
	return send_encoded(transport, encoded_length + 2U);
}

static bool parse_frame(struct flash_serial_transport *transport,
			size_t encoded_length,
			struct flash_serial_frame *frame)
{
	size_t decoded_length =
		cobs_decode_in_place(transport->encoded, encoded_length);
	uint16_t payload_length;
	uint32_t expected_crc;
	uint32_t actual_crc;

	if (decoded_length <
	    FLASH_SERIAL_HEADER_BYTES + FLASH_SERIAL_CRC_BYTES) {
		return false;
	}
	payload_length = load_u16_le(&transport->encoded[6]);
	if (decoded_length != FLASH_SERIAL_HEADER_BYTES +
				      (size_t)payload_length +
				      FLASH_SERIAL_CRC_BYTES ||
	    load_u32_le(&transport->encoded[0]) !=
		    FLASH_SERIAL_MAGIC ||
	    transport->encoded[4] != FLASH_SERIAL_PROTOCOL_VERSION) {
		return false;
	}

	expected_crc = load_u32_le(
		&transport->encoded[decoded_length -
				    FLASH_SERIAL_CRC_BYTES]);
	actual_crc = crc32_ieee(
		transport->encoded,
		decoded_length - FLASH_SERIAL_CRC_BYTES);
	if (actual_crc != expected_crc) {
		return false;
	}

	frame->type = transport->encoded[5];
	frame->token = load_u32_le(&transport->encoded[8]);
	frame->payload = &transport->encoded[FLASH_SERIAL_HEADER_BYTES];
	frame->payload_length = payload_length;
	return true;
}

static int receive_frame(struct flash_serial_transport *transport,
			 uint32_t timeout_ms,
			 struct flash_serial_frame *frame)
{
	int64_t deadline = k_uptime_get() + timeout_ms;
	size_t encoded_length = 0;
	bool overflow = false;

	while (k_uptime_get() < deadline) {
		unsigned char byte;

		if (uart_poll_in(transport->uart, &byte) != 0) {
			k_msleep(1);
			continue;
		}
		if (byte != 0U) {
			if (!overflow &&
			    encoded_length < sizeof(transport->encoded)) {
				transport->encoded[encoded_length++] = byte;
			} else {
				overflow = true;
			}
			continue;
		}
		if (encoded_length == 0U && !overflow) {
			continue;
		}
		if (!overflow &&
		    parse_frame(transport, encoded_length, frame)) {
			return 0;
		}
		encoded_length = 0;
		overflow = false;
	}

	return -ETIMEDOUT;
}

static int wait_for_frame(struct flash_serial_transport *transport,
			  uint8_t expected_type, uint32_t expected_token,
			  uint32_t timeout_ms,
			  struct flash_serial_frame *frame)
{
	int64_t deadline = k_uptime_get() + timeout_ms;

	while (k_uptime_get() < deadline) {
		uint32_t remaining =
			(uint32_t)(deadline - k_uptime_get());
		int err = receive_frame(transport, remaining, frame);

		if (err != 0) {
			return err;
		}
		if (frame->type == expected_type &&
		    frame->token == expected_token) {
			return 0;
		}
	}

	return -ETIMEDOUT;
}

static int serial_transfer_begin(void *context, uint32_t log_id,
				 uint32_t start_sector)
{
	struct flash_serial_transport *transport = context;
	uint8_t payload[8];

	transport->active_token = ++transport->token_counter;
	if (transport->active_token == 0U) {
		transport->active_token = ++transport->token_counter;
	}
	transport->active_ordinal = 0;
	store_u32_le(&payload[0], log_id);
	store_u32_le(&payload[4], start_sector);
	return send_frame(transport, FLASH_SERIAL_BEGIN,
			  transport->active_token, payload,
			  sizeof(payload));
}

static int serial_send_sector(void *context, uint32_t sector,
			      const struct flash_chunk *raw_sector)
{
	struct flash_serial_transport *transport = context;
	struct flash_serial_frame frame;
	uint8_t *payload = &transport->encoded[0];
	uint32_t ordinal = transport->active_ordinal;
	int err;

	store_u32_le(&payload[0], ordinal);
	store_u32_le(&payload[4], sector);
	memcpy(&payload[8], raw_sector, sizeof(*raw_sector));
	err = send_frame(transport, FLASH_SERIAL_SECTOR,
			 transport->active_token, payload,
			 (uint16_t)(8U + sizeof(*raw_sector)));
	if (err != 0) {
		return err;
	}

	err = wait_for_frame(transport, FLASH_SERIAL_SECTOR_ACK,
			     transport->active_token,
			     FLASH_SERIAL_FRAME_TIMEOUT_MS, &frame);
	if (err != 0 || frame.payload_length != 4U ||
	    load_u32_le(frame.payload) != ordinal) {
		return -EIO;
	}
	transport->active_ordinal++;
	return 0;
}

static bool summary_matches(const uint8_t *payload,
			    const struct flash_transfer_summary *summary)
{
	return load_u32_le(&payload[4]) == summary->log_id &&
	       load_u32_le(&payload[8]) == summary->start_sector &&
	       load_u32_le(&payload[12]) == summary->end_sector &&
	       load_u32_le(&payload[16]) == summary->sector_count &&
	       load_u32_le(&payload[20]) == summary->raw_crc;
}

static enum flash_transport_result
serial_transfer_finish(void *context,
		       const struct flash_transfer_summary *summary)
{
	struct flash_serial_transport *transport = context;
	struct flash_serial_frame frame;
	uint8_t payload[20];
	uint32_t disposition;
	int err;

	store_u32_le(&payload[0], summary->log_id);
	store_u32_le(&payload[4], summary->start_sector);
	store_u32_le(&payload[8], summary->end_sector);
	store_u32_le(&payload[12], summary->sector_count);
	store_u32_le(&payload[16], summary->raw_crc);
	err = send_frame(transport, FLASH_SERIAL_END,
			 transport->active_token, payload,
			 sizeof(payload));
	if (err != 0) {
		return FLASH_TRANSPORT_RETRY;
	}

	err = wait_for_frame(transport, FLASH_SERIAL_DISPOSITION,
			     transport->active_token,
			     FLASH_SERIAL_FRAME_TIMEOUT_MS, &frame);
	if (err != 0 || frame.payload_length != 24U ||
	    !summary_matches(frame.payload, summary)) {
		return FLASH_TRANSPORT_RETRY;
	}

	disposition = load_u32_le(&frame.payload[0]);
	if (disposition == FLASH_SERIAL_DISPOSITION_ERASE) {
		return FLASH_TRANSPORT_ERASE;
	}
	if (disposition == FLASH_SERIAL_DISPOSITION_DONE) {
		return FLASH_TRANSPORT_DONE;
	}
	return FLASH_TRANSPORT_RETRY;
}

const struct flash_transport_ops flash_serial_transport_ops = {
	.begin = serial_transfer_begin,
	.send_sector = serial_send_sector,
	.finish = serial_transfer_finish,
};

int flash_serial_transport_init(struct flash_serial_transport *transport,
				const struct device *uart,
				uint32_t sector_count)
{
	int err;

	if (transport == NULL || uart == NULL || !device_is_ready(uart) ||
	    sector_count < 3U) {
		return -EINVAL;
	}

	memset(transport, 0, sizeof(*transport));
	transport->uart = uart;
	transport->sector_count = sector_count;
	transport->token_counter = k_cycle_get_32();
	k_sem_init(&transport->tx_done, 0, 1);
	err = uart_irq_callback_user_data_set(
		uart, serial_uart_callback, transport);
	if (err != 0) {
		return err;
	}

	/* The CDC ACM driver does not queue USB OUT transfers until RX is enabled. */
	uart_irq_rx_enable(uart);
	return 0;
}

static int send_info(struct flash_serial_transport *transport,
		     const struct flash_log *log)
{
	uint8_t payload[12];

	store_u32_le(&payload[0], FLASH_LOG_SECTOR_BYTES);
	store_u32_le(&payload[4], transport->sector_count);
	store_u32_le(&payload[8], flash_log_is_empty(log) ? 1U : 0U);
	return send_frame(transport, FLASH_SERIAL_INFO, 0U, payload,
			  sizeof(payload));
}

static void send_error(struct flash_serial_transport *transport,
		       enum flash_log_result result)
{
	uint8_t payload[4];

	store_u32_le(payload, (uint32_t)result);
	(void)send_frame(transport, FLASH_SERIAL_ERROR, 0U, payload,
			 sizeof(payload));
}

enum flash_serial_session_result
flash_serial_upload_session(struct flash_serial_transport *transport,
			    struct flash_log *log,
			    uint32_t hello_timeout_ms)
{
	struct flash_serial_frame frame;
	int err;

	if (transport == NULL || log == NULL ||
	    log->transport != &flash_serial_transport_ops) {
		return FLASH_SERIAL_SESSION_ERROR;
	}

	err = wait_for_frame(transport, FLASH_SERIAL_HELLO, 0U,
			     hello_timeout_ms, &frame);
	if (err != 0) {
		return FLASH_SERIAL_NO_HOST;
	}
	if (send_info(transport, log) != 0) {
		return FLASH_SERIAL_SESSION_ERROR;
	}

	while (true) {
		enum flash_log_result result;

		err = receive_frame(transport,
				    FLASH_SERIAL_FRAME_TIMEOUT_MS, &frame);
		if (err != 0) {
			return FLASH_SERIAL_SESSION_ERROR;
		}
		if (frame.type == FLASH_SERIAL_SESSION_DONE &&
		    frame.token == 0U) {
			return FLASH_SERIAL_SESSION_COMPLETE;
		}
		if (frame.type != FLASH_SERIAL_READ_NEXT ||
		    frame.token != 0U) {
			continue;
		}
		if (flash_log_is_empty(log)) {
			if (send_frame(transport, FLASH_SERIAL_EMPTY, 0U,
				       NULL, 0U) != 0) {
				return FLASH_SERIAL_SESSION_ERROR;
			}
			continue;
		}

		result = flash_log_read_one(log);
		if (result == FLASH_LOG_OK) {
			continue;
		}
		if (result == FLASH_LOG_TRANSPORT_DONE) {
			return FLASH_SERIAL_SESSION_COMPLETE;
		}
		if (result == FLASH_LOG_TRANSPORT_ERROR) {
			/* The range remains intact for another READ_NEXT. */
			continue;
		}
		send_error(transport, result);
		return FLASH_SERIAL_SESSION_ERROR;
	}
}
