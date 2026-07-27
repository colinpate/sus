#include "flash_smoke_test.h"

#include <errno.h>
#include <string.h>

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/spi.h>
#include <zephyr/kernel.h>

#define SUS_BOARD_NODE DT_NODELABEL(sus_board)

#define FLASH_CMD_RELEASE_DEEP_POWER_DOWN 0xab
#define FLASH_CMD_READ_JEDEC_ID 0x9f

#define FLASH_EXPECTED_MANUFACTURER 0xc2
#define FLASH_EXPECTED_MEMORY_TYPE 0x20
#define FLASH_EXPECTED_DENSITY 0x19

static const struct device *const spi = DEVICE_DT_GET(DT_NODELABEL(xiao_spi));
static const struct gpio_dt_spec flash_cs =
	GPIO_DT_SPEC_GET(SUS_BOARD_NODE, flash_cs_gpios);

static const struct spi_config flash_spi_config = {
	.frequency = 125000U,
	.operation = SPI_WORD_SET(8) | SPI_TRANSFER_MSB |
		     SPI_MODE_CPOL | SPI_MODE_CPHA,
	.slave = 0,
};

static int flash_spi_write(const uint8_t *data, size_t length)
{
	struct spi_buf buffer = {
		.buf = (void *)data,
		.len = length,
	};
	const struct spi_buf_set buffers = {
		.buffers = &buffer,
		.count = 1,
	};
	int err;

	gpio_pin_set_dt(&flash_cs, 1);
	k_busy_wait(2);
	err = spi_write(spi, &flash_spi_config, &buffers);
	k_busy_wait(2);
	gpio_pin_set_dt(&flash_cs, 0);

	return err;
}

int flash_smoke_test_run(struct flash_smoke_test_result *result)
{
	uint8_t release_command = FLASH_CMD_RELEASE_DEEP_POWER_DOWN;
	uint8_t tx_bytes[4] = { FLASH_CMD_READ_JEDEC_ID, 0, 0, 0 };
	uint8_t rx_bytes[4] = { 0 };
	struct spi_buf tx_buffer = {
		.buf = tx_bytes,
		.len = sizeof(tx_bytes),
	};
	struct spi_buf rx_buffer = {
		.buf = rx_bytes,
		.len = sizeof(rx_bytes),
	};
	const struct spi_buf_set tx = {
		.buffers = &tx_buffer,
		.count = 1,
	};
	const struct spi_buf_set rx = {
		.buffers = &rx_buffer,
		.count = 1,
	};
	int err;

	if (result == NULL) {
		return -EINVAL;
	}
	memset(result, 0, sizeof(*result));

	if (!device_is_ready(spi) || !gpio_is_ready_dt(&flash_cs)) {
		return -ENODEV;
	}

	err = gpio_pin_configure_dt(&flash_cs, GPIO_OUTPUT_INACTIVE);
	if (err != 0) {
		return err;
	}

	err = flash_spi_write(&release_command, sizeof(release_command));
	if (err != 0) {
		return err;
	}

	k_msleep(1);

	gpio_pin_set_dt(&flash_cs, 1);
	k_busy_wait(2);
	err = spi_transceive(spi, &flash_spi_config, &tx, &rx);
	k_busy_wait(2);
	gpio_pin_set_dt(&flash_cs, 0);
	if (err != 0) {
		return err;
	}

	memcpy(result->jedec_id, &rx_bytes[1], sizeof(result->jedec_id));
	result->matches_expected_id =
		result->jedec_id[0] == FLASH_EXPECTED_MANUFACTURER &&
		result->jedec_id[1] == FLASH_EXPECTED_MEMORY_TYPE &&
		result->jedec_id[2] == FLASH_EXPECTED_DENSITY;

	return 0;
}
