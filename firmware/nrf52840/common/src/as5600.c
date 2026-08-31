#include "as5600.h"

#include <errno.h>
#include <stdint.h>

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/i2c.h>

#define AS5600_ADDRESS 0x36U
#define AS5600_ZPOS_REGISTER 0x01U
#define AS5600_MPOS_REGISTER 0x03U
#define AS5600_MANG_REGISTER 0x05U
#define AS5600_CONF_REGISTER 0x07U
#define AS5600_STATUS_REGISTER 0x0bU
#define AS5600_ANGLE_REGISTER 0x0eU

#define AS5600_CONF_HIGH 0x06U
#define AS5600_CONF_LOW 0x20U

static const struct device *const sensor_i2c =
	DEVICE_DT_GET(DT_BUS(DT_ALIAS(imu1)));

static int as5600_write_u12(uint8_t reg, uint16_t value)
{
	const uint8_t bytes[2] = {
		(uint8_t)((value >> 8U) & 0x0fU),
		(uint8_t)value,
	};

	return i2c_burst_write(sensor_i2c, AS5600_ADDRESS, reg, bytes,
			       sizeof(bytes));
}

int sus_as5600_init(struct sus_as5600 *sensor)
{
	uint8_t status;
	const uint8_t config[2] = {
		AS5600_CONF_HIGH,
		AS5600_CONF_LOW,
	};
	int err;

	if (sensor == NULL) {
		return -EINVAL;
	}

	sensor->i2c = sensor_i2c;
	sensor->available = false;
	if (!device_is_ready(sensor_i2c)) {
		return -ENODEV;
	}

	err = i2c_reg_read_byte(sensor->i2c, AS5600_ADDRESS,
				AS5600_STATUS_REGISTER, &status);
	if (err != 0) {
		return err;
	}

	/*
	 * Match the ESP configuration: normal power, no hysteresis, digital
	 * PWM output, 4x slow filter, 6-LSB fast-filter threshold, watchdog
	 * off, and the full 12-bit angular range.
	 */
	err = i2c_burst_write(sensor->i2c, AS5600_ADDRESS,
			      AS5600_CONF_REGISTER, config,
			      sizeof(config));
	if (err == 0) {
		err = as5600_write_u12(AS5600_ZPOS_REGISTER, 0U);
	}
	if (err == 0) {
		err = as5600_write_u12(AS5600_MPOS_REGISTER, 4095U);
	}
	if (err == 0) {
		err = as5600_write_u12(AS5600_MANG_REGISTER, 4095U);
	}
	if (err != 0) {
		return err;
	}

	sensor->available = true;
	return 0;
}

int sus_as5600_read(const struct sus_as5600 *sensor, uint16_t *angle)
{
	uint8_t bytes[2];
	int err;

	if (sensor == NULL || angle == NULL || !sensor->available) {
		return -ENODEV;
	}

	err = i2c_burst_read(sensor->i2c, AS5600_ADDRESS,
			     AS5600_ANGLE_REGISTER, bytes, sizeof(bytes));
	if (err != 0) {
		return err;
	}

	*angle = (((uint16_t)bytes[0] << 8U) | bytes[1]) & 0x0fffU;
	return 0;
}
