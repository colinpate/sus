#ifndef __SENSOR_READER_H__
#define __SENSOR_READER_H__

#include <Arduino.h>
#include <Wire.h>

#ifndef IMU_SELECTION_LIS3DH
#define IMU_SELECTION_LIS3DH 0
#endif

#ifndef IMU_SELECTION_LSM6DSOX
#define IMU_SELECTION_LSM6DSOX 1
#endif

#ifndef IMU_SELECTION_LSM6DSO32
#define IMU_SELECTION_LSM6DSO32 2
#endif

#ifndef IMU1_SELECTION
#ifdef IMU_SELECTION
#define IMU1_SELECTION IMU_SELECTION
#else
#define IMU1_SELECTION IMU_SELECTION_LSM6DSOX
#endif
#endif

#ifndef IMU2_SELECTION
#ifdef IMU_SELECTION
#define IMU2_SELECTION IMU_SELECTION
#else
#define IMU2_SELECTION IMU_SELECTION_LSM6DSO32
#endif
#endif

#ifndef IMU1_ADDR
#if IMU1_SELECTION == IMU_SELECTION_LIS3DH
#define IMU1_ADDR 0x18
#else
#define IMU1_ADDR 0x6A
#endif
#endif

#ifndef IMU2_ADDR
#if IMU2_SELECTION == IMU_SELECTION_LIS3DH
#define IMU2_ADDR 0x19
#else
#define IMU2_ADDR 0x6B
#endif
#endif

#ifndef LIS3MDL_ADDR
#define LIS3MDL_ADDR 0x1C
#endif

static constexpr uint8_t MMC_ADDR = 0x30;

#pragma pack(push, 1)
struct LogRecord {
  uint32_t t_ms;
  uint32_t seq;
  int16_t lis1[3];
  int16_t lis2[3];
  int16_t gyro1_dps10[3];
  int16_t gyro2_dps10[3];
  int16_t mmc_mG[3];
  int16_t lis3mdl_mG[3];
  uint16_t angle;
  int32_t temp_C;
};
#pragma pack(pop)

static_assert(sizeof(LogRecord) == 50, "Unexpected LogRecord size");

struct SensorConnections {
  bool imu1 = false;
  bool imu2 = false;
  bool mmc = false;
  bool lis3mdl = false;
  bool angle = false;
};

SensorConnections initSensors(TwoWire &wire);
void readSensors(const SensorConnections &connections, LogRecord &record);
const char *imu1Name();
const char *imu2Name();

#endif // __SENSOR_READER_H__
