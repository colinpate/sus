#include <stdbool.h>
#include <stdint.h>

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/i2c.h>
#include <zephyr/drivers/spi.h>
#include <zephyr/init.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#define SUS_NODE DT_NODELABEL(sus_board)

#define LSM6DSO_ADDR_LOW 0x6a
#define LSM6DSO_ADDR_HIGH 0x6b
#define LSM6DSO_REG_WHO_AM_I 0x0f
#define LSM6DSO_WHO_AM_I_EXPECTED 0x6c

#define FLASH_CMD_RELEASE_DPD 0xab
#define FLASH_CMD_READ_JEDEC_ID 0x9f
#define FLASH_EXPECTED_MANUFACTURER 0xc2
#define FLASH_EXPECTED_MEMORY_TYPE 0x20
#define FLASH_EXPECTED_DENSITY_256MBIT 0x19

static const struct gpio_dt_spec vperiph_en =
	GPIO_DT_SPEC_GET(SUS_NODE, vperiph_en_gpios);
static const struct gpio_dt_spec flash_cs =
	GPIO_DT_SPEC_GET(SUS_NODE, flash_cs_gpios);
static const struct gpio_dt_spec imu_int =
	GPIO_DT_SPEC_GET(SUS_NODE, imu_int_gpios);
static const struct gpio_dt_spec i2c_sda =
	GPIO_DT_SPEC_GET(SUS_NODE, i2c_sda_gpios);
static const struct gpio_dt_spec i2c_scl =
	GPIO_DT_SPEC_GET(SUS_NODE, i2c_scl_gpios);
static const struct gpio_dt_spec flash_sck =
	GPIO_DT_SPEC_GET(SUS_NODE, flash_sck_gpios);
static const struct gpio_dt_spec flash_miso =
	GPIO_DT_SPEC_GET(SUS_NODE, flash_miso_gpios);
static const struct gpio_dt_spec flash_mosi =
	GPIO_DT_SPEC_GET(SUS_NODE, flash_mosi_gpios);

static const struct device *const i2c = DEVICE_DT_GET(DT_NODELABEL(xiao_i2c));
static const struct device *const spi = DEVICE_DT_GET(DT_NODELABEL(xiao_spi));

#if DT_NODE_HAS_STATUS(DT_ALIAS(led0), okay)
static const struct gpio_dt_spec red_led =
	GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);
#define HAVE_RED_LED 1
#else
#define HAVE_RED_LED 0
#endif

#if DT_NODE_HAS_STATUS(DT_ALIAS(led1), okay)
static const struct gpio_dt_spec green_led =
	GPIO_DT_SPEC_GET(DT_ALIAS(led1), gpios);
#define HAVE_GREEN_LED 1
#else
#define HAVE_GREEN_LED 0
#endif

static int power_peripherals_early(void)
{
	if (!gpio_is_ready_dt(&vperiph_en)) {
		return 0;
	}

	(void)gpio_pin_configure_dt(&vperiph_en, GPIO_OUTPUT_ACTIVE);

	/*
	 * Bring the switched rail up before the application talks to I2C/SPI.
	 * main() waits longer before the first device transaction.
	 */
	k_busy_wait(5000);
	return 0;
}

SYS_INIT(power_peripherals_early, POST_KERNEL, 0);

static void led_set(const struct gpio_dt_spec *led, bool on)
{
	if (led == NULL || !gpio_is_ready_dt(led)) {
		return;
	}

	(void)gpio_pin_set_dt(led, on ? 1 : 0);
}

static void led_all_off(void)
{
#if HAVE_RED_LED
	led_set(&red_led, false);
#endif
#if HAVE_GREEN_LED
	led_set(&green_led, false);
#endif
}

static void configure_leds(void)
{
#if HAVE_RED_LED
	if (gpio_is_ready_dt(&red_led)) {
		(void)gpio_pin_configure_dt(&red_led, GPIO_OUTPUT_INACTIVE);
	}
#endif
#if HAVE_GREEN_LED
	if (gpio_is_ready_dt(&green_led)) {
		(void)gpio_pin_configure_dt(&green_led, GPIO_OUTPUT_INACTIVE);
	}
#endif
}

static void blink_led(const struct gpio_dt_spec *led, int count, int on_ms,
		      int off_ms)
{
	for (int i = 0; i < count; i++) {
		led_set(led, true);
		k_msleep(on_ms);
		led_set(led, false);
		k_msleep(off_ms);
	}
}

static bool configure_board_gpios(void)
{
	int err;

	if (!gpio_is_ready_dt(&vperiph_en)) {
		printk("Power: enable GPIO controller is not ready\n");
		return false;
	}

	err = gpio_pin_configure_dt(&vperiph_en, GPIO_OUTPUT_ACTIVE);
	if (err != 0) {
		printk("Power: failed to drive enable GPIO, err=%d\n", err);
		return false;
	}

	printk("Power: enable %s pin %u driven active\n",
	       vperiph_en.port->name, vperiph_en.pin);

	if (gpio_is_ready_dt(&imu_int)) {
		(void)gpio_pin_configure_dt(&imu_int, GPIO_INPUT);
	}

	if (gpio_is_ready_dt(&flash_cs)) {
		(void)gpio_pin_configure_dt(&flash_cs, GPIO_OUTPUT_INACTIVE);
	}

	return true;
}

static int raw_line_level(const struct gpio_dt_spec *line)
{
	if (!gpio_is_ready_dt(line)) {
		return -1;
	}

	return gpio_pin_get_raw(line->port, line->pin);
}

static void print_bus_levels(const char *when)
{
	printk("Lines %s: SDA=%d SCL=%d CS#=%d SCK=%d MISO=%d MOSI=%d\n",
	       when,
	       raw_line_level(&i2c_sda),
	       raw_line_level(&i2c_scl),
	       raw_line_level(&flash_cs),
	       raw_line_level(&flash_sck),
	       raw_line_level(&flash_miso),
	       raw_line_level(&flash_mosi));
}

static bool read_imu_whoami(uint8_t address, uint8_t *who_am_i)
{
	uint8_t reg = LSM6DSO_REG_WHO_AM_I;
	int err = i2c_write_read(i2c, address, &reg, sizeof(reg), who_am_i,
				 sizeof(*who_am_i));

	if (err != 0) {
		printk("IMU: no response at 0x%02x, err=%d\n", address, err);
		return false;
	}

	printk("IMU: WHO_AM_I at 0x%02x = 0x%02x\n", address, *who_am_i);
	return *who_am_i == LSM6DSO_WHO_AM_I_EXPECTED;
}

static int flash_write_command(uint8_t command)
{
	struct spi_config config = {
		.frequency = 125000U,
		.operation = SPI_WORD_SET(8) | SPI_TRANSFER_MSB,
		.slave = 0,
	};
	struct spi_buf tx_buf = {
		.buf = &command,
		.len = sizeof(command),
	};
	struct spi_buf_set tx = {
		.buffers = &tx_buf,
		.count = 1,
	};
	int err;

	gpio_pin_set_dt(&flash_cs, 1);
	k_busy_wait(2);
	err = spi_write(spi, &config, &tx);
	k_busy_wait(2);
	gpio_pin_set_dt(&flash_cs, 0);

	return err;
}

static bool read_flash_jedec_id(uint8_t id[3])
{
	struct spi_config config = {
		.frequency = 125000U,
		.operation = SPI_WORD_SET(8) | SPI_TRANSFER_MSB,
		.slave = 0,
	};
	uint8_t tx_bytes[4] = { FLASH_CMD_READ_JEDEC_ID, 0x00, 0x00, 0x00 };
	uint8_t rx_bytes[4] = { 0 };
	struct spi_buf tx_buf = {
		.buf = tx_bytes,
		.len = sizeof(tx_bytes),
	};
	struct spi_buf rx_buf = {
		.buf = rx_bytes,
		.len = sizeof(rx_bytes),
	};
	struct spi_buf_set tx = {
		.buffers = &tx_buf,
		.count = 1,
	};
	struct spi_buf_set rx = {
		.buffers = &rx_buf,
		.count = 1,
	};
	int err;

	err = flash_write_command(FLASH_CMD_RELEASE_DPD);
	if (err != 0) {
		printk("Flash: release from deep power-down failed, err=%d\n",
		       err);
		return false;
	}

	k_msleep(1);

	gpio_pin_set_dt(&flash_cs, 1);
	k_busy_wait(2);
	err = spi_transceive(spi, &config, &tx, &rx);
	k_busy_wait(2);
	gpio_pin_set_dt(&flash_cs, 0);

	if (err != 0) {
		printk("Flash: JEDEC-ID transaction failed, err=%d\n", err);
		return false;
	}

	printk("Flash: raw RX = %02x %02x %02x %02x\n",
	       rx_bytes[0], rx_bytes[1], rx_bytes[2], rx_bytes[3]);

	id[0] = rx_bytes[1];
	id[1] = rx_bytes[2];
	id[2] = rx_bytes[3];

	printk("Flash: JEDEC ID = %02x %02x %02x\n", id[0], id[1], id[2]);

	return id[0] == FLASH_EXPECTED_MANUFACTURER &&
	       id[1] == FLASH_EXPECTED_MEMORY_TYPE &&
	       id[2] == FLASH_EXPECTED_DENSITY_256MBIT;
}

int main(void)
{
	bool imu_ok = false;
	bool flash_ok = false;
	bool power_enable_ok;
	uint8_t who_am_i = 0;
	uint8_t flash_id[3] = { 0 };

	configure_leds();
	led_all_off();

	printk("\nSUS board bringup starting\n");
	printk("USB serial console: printk over CDC ACM\n");

	power_enable_ok = configure_board_gpios();
	k_msleep(100);
	print_bus_levels("idle");

	if (!power_enable_ok) {
		printk("IMU: skipped because peripheral power enable failed\n");
	} else if (!device_is_ready(i2c)) {
		printk("I2C: xiao_i2c is not ready\n");
	} else {
		imu_ok = read_imu_whoami(LSM6DSO_ADDR_LOW, &who_am_i);
		if (!imu_ok) {
			printk("IMU: trying alternate SA0-high address\n");
			imu_ok = read_imu_whoami(LSM6DSO_ADDR_HIGH, &who_am_i);
		}
		print_bus_levels("after I2C");
	}

	if (!power_enable_ok) {
		printk("Flash: skipped because peripheral power enable failed\n");
	} else if (!device_is_ready(spi)) {
		printk("SPI: xiao_spi is not ready\n");
	} else if (!gpio_is_ready_dt(&flash_cs)) {
		printk("Flash: CS GPIO is not ready\n");
	} else {
		flash_ok = read_flash_jedec_id(flash_id);
		print_bus_levels("after SPI");
	}

	printk("Bringup result: IMU=%s, flash=%s\n",
	       imu_ok ? "PASS" : "FAIL", flash_ok ? "PASS" : "FAIL");

	if (imu_ok && flash_ok) {
		printk("Expected flash for MX25L25645G: c2 20 19\n");
	} else {
		printk("Check V_periph enable pin, solder joints, and bus routing.\n");
	}

	while (true) {
		if (imu_ok && flash_ok) {
#if HAVE_GREEN_LED
			blink_led(&green_led, 1, 120, 1880);
#else
			k_msleep(2000);
#endif
			printk("Alive: IMU=0x%02x, flash=%02x %02x %02x\n",
			       who_am_i, flash_id[0], flash_id[1], flash_id[2]);
		} else {
#if HAVE_RED_LED
			blink_led(&red_led, 3, 120, 120);
			k_msleep(1240);
#else
			k_msleep(2000);
#endif
			printk("Retry by reset after checking hardware. Last: IMU=0x%02x, flash=%02x %02x %02x\n",
			       who_am_i, flash_id[0], flash_id[1], flash_id[2]);
		}
	}

	return 0;
}
