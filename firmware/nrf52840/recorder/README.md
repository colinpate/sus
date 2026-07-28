# SUS nRF52840 recorder

This is the production-oriented companion to `../bringup`. The bring-up
application remains a simple sensor/flash console; this application records
the same packed 50-byte `dual_mag` records as the ESP32 firmware.

Board power, sensor access, the flash journal, its Zephyr storage adapter, and
the common devicetree definitions live in [`../common`](../common/README.md).
This directory contains only recorder-specific behavior and configuration.

## User flow

- Press the active-low button on XIAO `D2` to wake from System OFF.
- Recording starts automatically after boot and flash recovery.
- Hold `D2` for 0.8 seconds to stop.
- The recorder drains queued samples, writes the final partial sector, writes
  the log commit marker, powers down the sensor rail and SPI NOR, then enters
  System OFF.

The press that wakes the device is ignored by the long-press detector until
the button has first been released. `D2` is selected in `app.overlay`; change
the `record_button` GPIO there if the production board routes the button to a
different pin.

## Record and flash layout

Sampling runs at 200 Hz. A high-priority sampler places records in a 128-entry
queue so flash page-program latency does not move the sampling schedule. The
writer packs 81 records (4050 bytes) into each 4078-byte flash-log payload.
Sequence gaps expose queue drops to the host.

The binary record is compatible with `read_binary.py --format dual_mag`:

```text
uint32  t_ms
uint32  seq
int16   imu1_accel_mg[3]
int16   imu2_accel_mg[3]
int16   imu1_gyro_dps10[3]
int16   imu2_gyro_dps10[3]
int16   mmc5603_mG[3]
int16   lis3mdl_mG[3]
uint16  angle_raw
int32   temp_deciC
```

The AS5600 is read directly at I2C address `0x36` because Zephyr does not
provide a driver for it. Its runtime configuration matches the ESP firmware.
If a sensor is unavailable or a read fails, its fields are zero; the sequence
and status counters still expose timing or queue loss.

## Log-ID retention

The top 4 KiB nRF52840 RAM section is excluded from normal SRAM and retained
across System OFF. It contains a magic, format version, next log ID, inverted
copy, and CRC.

At boot:

1. Scan external flash.
2. If flash contains valid sectors, use its recovered next log ID.
3. If flash is empty and retained RAM validates, restore its next log ID.
4. Otherwise begin at zero.

The retained value is updated immediately when a new ID is consumed and again
before System OFF. Thus normal deep sleep preserves IDs even after the host
has emptied flash. A true power loss while flash is empty remains the only
case where the ID can be lost.

## Build

```sh
ZEPHYR_BASE=/opt/nordic/ncs/v3.4.0/zephyr \
ZEPHYR_TOOLCHAIN_VARIANT=zephyr \
ZEPHYR_SDK_INSTALL_DIR=/opt/nordic/ncs/toolchains/ccc010f809/opt/zephyr-sdk \
PATH=/opt/nordic/ncs/toolchains/ccc010f809/bin:$PATH \
/opt/nordic/ncs/toolchains/ccc010f809/bin/cmake \
  -S firmware/nrf52840/recorder \
  -B firmware/nrf52840/recorder/build \
  -GNinja \
  -DBOARD=xiao_ble/nrf52840 \
  -DPython3_EXECUTABLE=/opt/nordic/ncs/toolchains/ccc010f809/bin/python3 \
  -DUSER_CACHE_DIR=/Users/colin/Documents/projects/sus/.zephyr-cache \
  -DUSE_CCACHE=0

ZEPHYR_BASE=/opt/nordic/ncs/v3.4.0/zephyr \
ZEPHYR_TOOLCHAIN_VARIANT=zephyr \
ZEPHYR_SDK_INSTALL_DIR=/opt/nordic/ncs/toolchains/ccc010f809/opt/zephyr-sdk \
PATH=/opt/nordic/ncs/toolchains/ccc010f809/bin:$PATH \
/opt/nordic/ncs/toolchains/ccc010f809/bin/cmake \
  --build firmware/nrf52840/recorder/build
```

The UF2 output is `firmware/nrf52840/recorder/build/zephyr/zephyr.uf2`.
