#ifndef __IMU_CONFIG_H__
#define __IMU_CONFIG_H__

// Select the IMU variant for each slot independently.
#define IMU_SELECTION_LIS3DH 0
#define IMU_SELECTION_LSM6DSOX 1
#define IMU_SELECTION_LSM6DSO32 2

// Preserve the legacy single-IMU override while giving each slot a shared
// project-level default when no build flag overrides are provided.
#ifndef IMU1_SELECTION
#ifdef IMU_SELECTION
#define IMU1_SELECTION IMU_SELECTION
#else
#define IMU1_SELECTION IMU_SELECTION_LSM6DSO32
#endif
#endif

#ifndef IMU2_SELECTION
#ifdef IMU_SELECTION
#define IMU2_SELECTION IMU_SELECTION
#else
#define IMU2_SELECTION IMU_SELECTION_LSM6DSO32
#endif
#endif

#endif // __IMU_CONFIG_H__
