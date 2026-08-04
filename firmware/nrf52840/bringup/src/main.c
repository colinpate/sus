#include <stdint.h>

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include "as5600.h"
#include "board_power.h"
#include "flash_smoke_test.h"
#include "sensor_reader.h"

#define SAMPLE_INTERVAL K_MSEC(200)

static void print_vector(const char *name, const int16_t value[3],
			 const char *unit)
{
	printk("%s=(%d, %d, %d) %s", name, value[0], value[1], value[2],
	       unit);
}

static void print_imu(const char *name, enum sus_sensor sensor,
		      const struct sus_sensor_sample *sample)
{
	if ((sample->valid & sensor) == 0U) {
		printk("%s=ERR", name);
		return;
	}

	const struct sus_imu_sample *imu =
		(sensor == SUS_SENSOR_IMU1) ? &sample->imu1 : &sample->imu2;
	int32_t temperature_magnitude =
		imu->temp_deci_c < 0 ? -imu->temp_deci_c : imu->temp_deci_c;

	printk("%s: ", name);
	print_vector("accel", imu->accel_mg, "mg");
	printk(" ");
	print_vector("gyro", imu->gyro_dps10, "0.1dps");
	printk(" temp=%s%d.%d C", imu->temp_deci_c < 0 ? "-" : "",
	       temperature_magnitude / 10, temperature_magnitude % 10);
}

static void print_magnetometer(const char *name, enum sus_sensor sensor,
			       const int16_t field_mg[3],
			       const struct sus_sensor_sample *sample)
{
	if ((sample->valid & sensor) == 0U) {
		printk("%s=ERR", name);
		return;
	}

	printk("%s: ", name);
	print_vector("field", field_mg, "mG");
}

static void print_angle(const struct sus_as5600 *sensor)
{
	uint16_t angle;

	if (sus_as5600_read(sensor, &angle) != 0) {
		printk("AS5600=ERR");
		return;
	}

	printk("AS5600: angle=%u raw", angle);
}

int main(void)
{
	struct sus_as5600 angle_sensor;
	struct flash_smoke_test_result flash_result;
	struct sus_sensor_reader reader;
	int angle_err;
	int flash_err;

	printk("\nSUS sensor console starting\n");

	if (!board_peripheral_power_is_ready()) {
		printk("Peripheral power rail failed to initialize\n");
		return 0;
	}

	flash_err = flash_smoke_test_run(&flash_result);
	if (flash_err != 0) {
		printk("Flash: SPI transaction failed, err=%d (FAIL)\n",
		       flash_err);
	} else {
		printk("Flash: JEDEC ID=%02x %02x %02x (%s)\n",
		       flash_result.jedec_id[0], flash_result.jedec_id[1],
		       flash_result.jedec_id[2],
		       flash_result.matches_expected_id ? "PASS" : "FAIL");
	}

	sus_sensor_reader_init(&reader);
	angle_err = sus_as5600_init(&angle_sensor);
	printk("Detected sensors: IMU1=%s IMU2=%s MMC5603=%s LIS3MDL=%s "
	       "AS5600=%s\n",
	       (reader.available & SUS_SENSOR_IMU1) != 0U ? "yes" : "no",
	       (reader.available & SUS_SENSOR_IMU2) != 0U ? "yes" : "no",
	       (reader.available & SUS_SENSOR_MMC5603) != 0U ? "yes" : "no",
	       (reader.available & SUS_SENSOR_LIS3MDL) != 0U ? "yes" : "no",
	       angle_err == 0 ? "yes" : "no");
	if (angle_err != 0) {
		printk("AS5600 initialization failed, err=%d\n", angle_err);
	}
	printk("Reading sensors every 200 ms\n");

	while (true) {
		struct sus_sensor_sample sample;

		sus_sensor_reader_read(&reader, &sample);

		printk("[%u ms] ", sample.timestamp_ms);
		print_imu("IMU1", SUS_SENSOR_IMU1, &sample);
		printk(" | ");
		print_imu("IMU2", SUS_SENSOR_IMU2, &sample);
		printk("\n           ");
		print_magnetometer("MMC5603", SUS_SENSOR_MMC5603,
				   sample.mmc5603_mg, &sample);
		printk(" | ");
		print_magnetometer("LIS3MDL", SUS_SENSOR_LIS3MDL,
				   sample.lis3mdl_mg, &sample);
		printk(" | ");
		print_angle(&angle_sensor);
		printk("\n");

		k_sleep(SAMPLE_INTERVAL);
	}

	return 0;
}
