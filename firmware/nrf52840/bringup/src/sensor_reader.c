#include "sensor_reader.h"

#include <errno.h>
#include <limits.h>
#include <string.h>

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/util.h>

static const struct device *const imu1 = DEVICE_DT_GET(DT_ALIAS(imu1));
static const struct device *const imu2 = DEVICE_DT_GET(DT_ALIAS(imu2));
static const struct device *const mmc5603 = DEVICE_DT_GET(DT_ALIAS(mmc));
static const struct device *const lis3mdl = DEVICE_DT_GET(DT_ALIAS(lis3mdl));

static int16_t clamp_i16(int32_t value)
{
	return (int16_t)CLAMP(value, INT16_MIN, INT16_MAX);
}

static int16_t radians_to_dps10(const struct sensor_value *value)
{
	int32_t ten_micro_degrees = sensor_rad_to_10udegrees(value);

	if (ten_micro_degrees >= 0) {
		ten_micro_degrees += 5000;
	} else {
		ten_micro_degrees -= 5000;
	}

	return clamp_i16(ten_micro_degrees / 10000);
}

static int read_imu(const struct device *device, struct sus_imu_sample *sample)
{
	struct sensor_value accel[3];
	struct sensor_value gyro[3];
	struct sensor_value temperature;
	int err;

	err = sensor_sample_fetch(device);
	if (err != 0) {
		return err;
	}

	err = sensor_channel_get(device, SENSOR_CHAN_ACCEL_XYZ, accel);
	if (err != 0) {
		return err;
	}

	err = sensor_channel_get(device, SENSOR_CHAN_GYRO_XYZ, gyro);
	if (err != 0) {
		return err;
	}

	err = sensor_channel_get(device, SENSOR_CHAN_DIE_TEMP, &temperature);
	if (err != 0) {
		return err;
	}

	for (size_t i = 0; i < ARRAY_SIZE(sample->accel_mg); i++) {
		sample->accel_mg[i] = clamp_i16(sensor_ms2_to_mg(&accel[i]));
		sample->gyro_dps10[i] = radians_to_dps10(&gyro[i]);
	}
	sample->temp_deci_c = (int32_t)sensor_value_to_deci(&temperature);

	return 0;
}

static int read_magnetometer(const struct device *device, int16_t field_mg[3])
{
	struct sensor_value field[3];
	int err;

	err = sensor_sample_fetch_chan(device, SENSOR_CHAN_MAGN_XYZ);
	if (err != 0) {
		return err;
	}

	err = sensor_channel_get(device, SENSOR_CHAN_MAGN_XYZ, field);
	if (err != 0) {
		return err;
	}

	for (size_t i = 0; i < ARRAY_SIZE(field); i++) {
		/* Zephyr magnetometer channels use gauss; retain milli-gauss. */
		field_mg[i] = clamp_i16((int32_t)sensor_value_to_milli(&field[i]));
	}

	return 0;
}

void sus_sensor_reader_init(struct sus_sensor_reader *reader)
{
	memset(reader, 0, sizeof(*reader));

	if (device_is_ready(imu1)) {
		reader->available |= SUS_SENSOR_IMU1;
	}
	if (device_is_ready(imu2)) {
		reader->available |= SUS_SENSOR_IMU2;
	}
	if (device_is_ready(mmc5603)) {
		reader->available |= SUS_SENSOR_MMC5603;
	}
	if (device_is_ready(lis3mdl)) {
		reader->available |= SUS_SENSOR_LIS3MDL;
	}
}

void sus_sensor_reader_read(struct sus_sensor_reader *reader,
			    struct sus_sensor_sample *sample)
{
	memset(sample, 0, sizeof(*sample));
	sample->timestamp_ms = (uint32_t)k_uptime_get();
	sample->sequence = reader->sequence++;

	if ((reader->available & SUS_SENSOR_IMU1) != 0U &&
	    read_imu(imu1, &sample->imu1) == 0) {
		sample->valid |= SUS_SENSOR_IMU1;
	}
	if ((reader->available & SUS_SENSOR_IMU2) != 0U &&
	    read_imu(imu2, &sample->imu2) == 0) {
		sample->valid |= SUS_SENSOR_IMU2;
	}
	if ((reader->available & SUS_SENSOR_MMC5603) != 0U &&
	    read_magnetometer(mmc5603, sample->mmc5603_mg) == 0) {
		sample->valid |= SUS_SENSOR_MMC5603;
	}
	if ((reader->available & SUS_SENSOR_LIS3MDL) != 0U &&
	    read_magnetometer(lis3mdl, sample->lis3mdl_mg) == 0) {
		sample->valid |= SUS_SENSOR_LIS3MDL;
	}
}
