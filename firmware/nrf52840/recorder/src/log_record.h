#ifndef SUS_LOG_RECORD_H_
#define SUS_LOG_RECORD_H_

#include <stddef.h>
#include <stdint.h>

/*
 * Binary-compatible with the ESP32 "dual_mag" LogRecord. The nRF52840 is
 * little-endian, matching the host's <II... format.
 */
struct __attribute__((packed)) sus_log_record {
	uint32_t timestamp_ms;
	uint32_t sequence;
	int16_t imu1_accel_mg[3];
	int16_t imu2_accel_mg[3];
	int16_t imu1_gyro_dps10[3];
	int16_t imu2_gyro_dps10[3];
	int16_t mmc5603_mg[3];
	int16_t lis3mdl_mg[3];
	uint16_t angle;
	int32_t temp_deci_c;
};

_Static_assert(sizeof(struct sus_log_record) == 50U,
	       "ESP-compatible log record must remain 50 bytes");
_Static_assert(offsetof(struct sus_log_record, sequence) == 4U,
	       "sequence offset must match the ESP format");
_Static_assert(offsetof(struct sus_log_record, imu1_accel_mg) == 8U,
	       "IMU1 acceleration offset must match the ESP format");
_Static_assert(offsetof(struct sus_log_record, imu2_accel_mg) == 14U,
	       "IMU2 acceleration offset must match the ESP format");
_Static_assert(offsetof(struct sus_log_record, imu1_gyro_dps10) == 20U,
	       "IMU1 gyro offset must match the ESP format");
_Static_assert(offsetof(struct sus_log_record, imu2_gyro_dps10) == 26U,
	       "IMU2 gyro offset must match the ESP format");
_Static_assert(offsetof(struct sus_log_record, mmc5603_mg) == 32U,
	       "MMC5603 offset must match the ESP format");
_Static_assert(offsetof(struct sus_log_record, lis3mdl_mg) == 38U,
	       "LIS3MDL offset must match the ESP format");
_Static_assert(offsetof(struct sus_log_record, angle) == 44U,
	       "angle offset must match the ESP format");
_Static_assert(offsetof(struct sus_log_record, temp_deci_c) == 46U,
	       "temperature offset must match the ESP format");

#endif /* SUS_LOG_RECORD_H_ */
