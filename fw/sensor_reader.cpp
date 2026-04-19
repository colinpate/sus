#include "sensor_reader.h"

#include <math.h>

#include <Adafruit_AS5600.h>
#include <Adafruit_LIS3MDL.h>
#include <Adafruit_MMC56x3.h>
#include <Adafruit_Sensor.h>

#if (IMU1_SELECTION == IMU_SELECTION_LIS3DH) ||                              \
    (IMU2_SELECTION == IMU_SELECTION_LIS3DH)
#include <Adafruit_LIS3DH.h>
#endif

#if (IMU1_SELECTION == IMU_SELECTION_LSM6DSOX) ||                            \
    (IMU2_SELECTION == IMU_SELECTION_LSM6DSOX)
#include <Adafruit_LSM6DSOX.h>
#endif

#if (IMU1_SELECTION == IMU_SELECTION_LSM6DSO32) ||                           \
    (IMU2_SELECTION == IMU_SELECTION_LSM6DSO32)
#include <Adafruit_LSM6DSO32.h>
#endif

#if IMU1_SELECTION == IMU_SELECTION_LIS3DH
static Adafruit_LIS3DH imu1(&Wire);
#elif IMU1_SELECTION == IMU_SELECTION_LSM6DSOX
static Adafruit_LSM6DSOX imu1;
#elif IMU1_SELECTION == IMU_SELECTION_LSM6DSO32
static Adafruit_LSM6DSO32 imu1;
#else
#error "Unsupported IMU1_SELECTION"
#endif

#if IMU2_SELECTION == IMU_SELECTION_LIS3DH
static Adafruit_LIS3DH imu2(&Wire);
#elif IMU2_SELECTION == IMU_SELECTION_LSM6DSOX
static Adafruit_LSM6DSOX imu2;
#elif IMU2_SELECTION == IMU_SELECTION_LSM6DSO32
static Adafruit_LSM6DSO32 imu2;
#else
#error "Unsupported IMU2_SELECTION"
#endif

static Adafruit_MMC5603 mmc(12345);
static Adafruit_LIS3MDL lis3mdl;
static Adafruit_AS5600 as5600;

namespace {

uint32_t sample_index = 0;
int16_t last_mmc_mG[3] = {};
int16_t last_lis3mdl_mG[3] = {};

const char *imuSelectionName(int selection) {
  switch (selection) {
    case IMU_SELECTION_LIS3DH:
      return "LIS3DH";
    case IMU_SELECTION_LSM6DSOX:
      return "LSM6DSOX";
    case IMU_SELECTION_LSM6DSO32:
      return "LSM6DSO32";
    default:
      return "UNKNOWN";
  }
}

bool initAS5600() {
  if (!as5600.begin()) {
    Serial.println("Failed to init AS5600");
    return false;
  }

  Serial.println("AS5600 found!");

  as5600.enableWatchdog(false);
  as5600.setPowerMode(AS5600_POWER_MODE_NOM);
  as5600.setHysteresis(AS5600_HYSTERESIS_OFF);
  as5600.setOutputStage(AS5600_OUTPUT_STAGE_DIGITAL_PWM);
  as5600.setSlowFilter(AS5600_SLOW_FILTER_4X);
  as5600.setFastFilterThresh(AS5600_FAST_FILTER_THRESH_6LSB);
  as5600.setZPosition(0);
  as5600.setMPosition(4095);
  as5600.setMaxAngle(4095);

  return true;
}

int16_t accelMps2ToMilliG(float value_mps2) {
  return (int16_t)lroundf((value_mps2 / SENSORS_GRAVITY_STANDARD) * 1000.0f);
}

int16_t gyroRadToDps10(float value_rads) {
  constexpr float RAD_TO_DPS10 = (180.0f / PI) * 10.0f;
  return (int16_t)lroundf(value_rads * RAD_TO_DPS10);
}

int16_t magneticUtToMilliGauss(float value_ut) {
  return (int16_t)llround((double)value_ut * 10.0);
}

#if (IMU1_SELECTION == IMU_SELECTION_LIS3DH) ||                              \
    (IMU2_SELECTION == IMU_SELECTION_LIS3DH)
bool initImu(Adafruit_LIS3DH &imu, uint8_t address, const char *label) {
  if (!imu.begin(address)) {
    Serial.printf("%s not found at 0x%02X\n", label, address);
    return false;
  }

  imu.setDataRate(LIS3DH_DATARATE_200_HZ);
  imu.setRange(LIS3DH_RANGE_16_G);
  imu.setPerformanceMode(LIS3DH_MODE_HIGH_RESOLUTION);
  return true;
}

void readImu(Adafruit_LIS3DH &imu, int16_t accel_out[3], int16_t gyro_out[3]) {
  imu.read();

  accel_out[0] = (int16_t)lroundf(imu.x_g * 1000.0f);
  accel_out[1] = (int16_t)lroundf(imu.y_g * 1000.0f);
  accel_out[2] = (int16_t)lroundf(imu.z_g * 1000.0f);

  gyro_out[0] = 0;
  gyro_out[1] = 0;
  gyro_out[2] = 0;
}
#endif

#if (IMU1_SELECTION != IMU_SELECTION_LIS3DH) ||                              \
    (IMU2_SELECTION != IMU_SELECTION_LIS3DH)
void configureLsm6dsCommon(Adafruit_LSM6DS &imu) {
  imu.setAccelDataRate(LSM6DS_RATE_208_HZ);
  imu.setGyroDataRate(LSM6DS_RATE_208_HZ);
  imu.setGyroRange(LSM6DS_GYRO_RANGE_2000_DPS);
}

#if (IMU1_SELECTION == IMU_SELECTION_LSM6DSOX) ||                            \
    (IMU2_SELECTION == IMU_SELECTION_LSM6DSOX)
bool initImu(Adafruit_LSM6DSOX &imu, uint8_t address, const char *label) {
  if (!imu.begin_I2C(address, &Wire)) {
    Serial.printf("%s not found at 0x%02X\n", label, address);
    return false;
  }

  configureLsm6dsCommon(imu);
  imu.setAccelRange(LSM6DS_ACCEL_RANGE_16_G);
  return true;
}
#endif

#if (IMU1_SELECTION == IMU_SELECTION_LSM6DSO32) ||                           \
    (IMU2_SELECTION == IMU_SELECTION_LSM6DSO32)
bool initImu(Adafruit_LSM6DSO32 &imu, uint8_t address, const char *label) {
  if (!imu.begin_I2C(address, &Wire)) {
    Serial.printf("%s not found at 0x%02X\n", label, address);
    return false;
  }

  configureLsm6dsCommon(imu);
  imu.setAccelRange(LSM6DSO32_ACCEL_RANGE_32_G);
  return true;
}
#endif

void readImu(Adafruit_LSM6DS &imu, int16_t accel_out[3], int16_t gyro_out[3],
             int32_t *temp_out) {
  sensors_event_t accel;
  sensors_event_t gyro;
  sensors_event_t temp;
  imu.getEvent(&accel, &gyro, &temp);

  accel_out[0] = accelMps2ToMilliG(accel.acceleration.x);
  accel_out[1] = accelMps2ToMilliG(accel.acceleration.y);
  accel_out[2] = accelMps2ToMilliG(accel.acceleration.z);

  gyro_out[0] = gyroRadToDps10(gyro.gyro.x);
  gyro_out[1] = gyroRadToDps10(gyro.gyro.y);
  gyro_out[2] = gyroRadToDps10(gyro.gyro.z);

  if (temp_out != nullptr) {
    *temp_out = (int32_t)lroundf(temp.temperature * 10.0f);
  }
}
#endif

bool initLis3mdl(TwoWire &wire) {
  if (!lis3mdl.begin_I2C(LIS3MDL_ADDR, &wire)) {
    Serial.printf("LIS3MDL not found at 0x%02X\n", LIS3MDL_ADDR);
    return false;
  }

  lis3mdl.setPerformanceMode(LIS3MDL_MEDIUMMODE);
  lis3mdl.setOperationMode(LIS3MDL_CONTINUOUSMODE);
  lis3mdl.setDataRate(LIS3MDL_DATARATE_155_HZ);
  lis3mdl.setRange(LIS3MDL_RANGE_4_GAUSS);
  return true;
}

void updateMmcCache() {
  sensors_event_t mag_event;
  mmc.getEvent(&mag_event);

  last_mmc_mG[0] = magneticUtToMilliGauss(mag_event.magnetic.x);
  last_mmc_mG[1] = magneticUtToMilliGauss(mag_event.magnetic.y);
  last_mmc_mG[2] = magneticUtToMilliGauss(mag_event.magnetic.z);
}

void updateLis3mdlCache() {
  sensors_event_t mag_event;
  lis3mdl.getEvent(&mag_event);

  last_lis3mdl_mG[0] = magneticUtToMilliGauss(mag_event.magnetic.x);
  last_lis3mdl_mG[1] = magneticUtToMilliGauss(mag_event.magnetic.y);
  last_lis3mdl_mG[2] = magneticUtToMilliGauss(mag_event.magnetic.z);
}

} // namespace

SensorConnections initSensors(TwoWire &wire) {
  SensorConnections connections;

  wire.setClock(400000);

  connections.imu1 = initImu(imu1, IMU1_ADDR, "IMU1");
  connections.imu2 = initImu(imu2, IMU2_ADDR, "IMU2");

  if (!mmc.begin(MMC_ADDR, &wire)) {
    Serial.println("MMC5603 not found");
  } else {
    connections.mmc = true;
    mmc.setDataRate(100);
    mmc.setContinuousMode(true);
    Wire.beginTransmission(MMC56X3_DEFAULT_ADDRESS);
    Wire.write(0x1B);   // CTRL0
    Wire.write(0x20);   // Auto_SR_en
    Wire.endTransmission();
    Wire.beginTransmission(MMC56X3_DEFAULT_ADDRESS);
    Wire.write(0x1C);   // CTRL1
    Wire.write(0x01);   // BW = 01
    Wire.endTransmission();
    updateMmcCache();
  }

  connections.lis3mdl = initLis3mdl(wire);
  if (connections.lis3mdl) {
    updateLis3mdlCache();
  }

  connections.angle = initAS5600();

  Serial.printf("Active IMU backends: IMU1=%s IMU2=%s\n", imu1Name(),
                imu2Name());
  return connections;
}

void readSensors(const SensorConnections &connections, LogRecord &record) {
  if (connections.imu1) {
#if IMU1_SELECTION == IMU_SELECTION_LIS3DH
    readImu(imu1, record.lis1, record.gyro1_dps10);
#else
    readImu(imu1, record.lis1, record.gyro1_dps10, &record.temp_C);
#endif
  }

  if (connections.imu2) {
#if IMU2_SELECTION == IMU_SELECTION_LIS3DH
    readImu(imu2, record.lis2, record.gyro2_dps10);
#else
    readImu(imu2, record.lis2, record.gyro2_dps10, nullptr);
#endif
  }

  if (connections.angle) {
    record.angle = as5600.getAngle();
  }

  const bool even_sample = (sample_index++ & 1U) == 0;

  if (connections.mmc && even_sample) {
    updateMmcCache();
  }

  if (connections.lis3mdl && !even_sample) {
    updateLis3mdlCache();
  }

  if (connections.mmc) {
    memcpy(record.mmc_mG, last_mmc_mG, sizeof(record.mmc_mG));
  }

  if (connections.lis3mdl) {
    memcpy(record.lis3mdl_mG, last_lis3mdl_mG, sizeof(record.lis3mdl_mG));
  }
}

const char *imu1Name() { return imuSelectionName(IMU1_SELECTION); }

const char *imu2Name() { return imuSelectionName(IMU2_SELECTION); }
