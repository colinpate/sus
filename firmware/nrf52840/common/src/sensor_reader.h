#ifndef SUS_SENSOR_READER_H_
#define SUS_SENSOR_READER_H_

#include <stdint.h>

enum sus_sensor {
	SUS_SENSOR_IMU1 = 1U << 0,
	SUS_SENSOR_IMU2 = 1U << 1,
	SUS_SENSOR_MMC5603 = 1U << 2,
	SUS_SENSOR_LIS3MDL = 1U << 3,
};

struct sus_sensor_reader {
	uint32_t available;
	uint32_t sequence;
};

struct sus_imu_sample {
	int16_t accel_mg[3];
	int16_t gyro_dps10[3];
	int32_t temp_deci_c;
};

struct sus_sensor_sample {
	uint32_t timestamp_ms;
	uint32_t sequence;
	uint32_t valid;
	struct sus_imu_sample imu1;
	struct sus_imu_sample imu2;
	int16_t mmc5603_mg[3];
	int16_t lis3mdl_mg[3];
};

void sus_sensor_reader_init(struct sus_sensor_reader *reader);
void sus_sensor_reader_read(struct sus_sensor_reader *reader,
			    struct sus_sensor_sample *sample);

#endif /* SUS_SENSOR_READER_H_ */
