# SUS nRF52840 recorder

This is the production-oriented companion to `../bringup`. The bring-up
application remains a simple sensor/flash console; this application records
the same packed 50-byte `dual_mag` records as the ESP32 firmware.

Board power, sensor access, the flash journal, its Zephyr storage adapter, and
the common devicetree definitions live in [`../common`](../common/README.md).
This directory contains only recorder-specific behavior and configuration.

## User flow

- Press the normally-open button from XIAO `D3` to ground to wake from
  System OFF.
- After boot and flash recovery, the recorder waits five seconds for the USB
  receiver. Recording starts automatically if no receiver connects.
- Hold `D3` for 0.8 seconds to stop.
- The recorder drains queued samples, writes the final partial sector, writes
  the log commit marker, powers down the sensor rail and SPI NOR, then enters
  System OFF.

The press that wakes the device is ignored by the long-press detector until
the button has first been released. `D3` (`P0.29`) is selected in
`app.overlay` as an active-low input with an internal pull-up; change the
`record_button` GPIO there if the production board routes the button to a
different pin.

## Status LED

The XIAO's RGB LED reports recorder state:

- White: hardware initialization or retained-checkpoint validation.
- Purple: full external-flash recovery scan.
- Blue: waiting for a USB receiver or serving an upload session.
- Green: recording with all configured sensors responding.
- Yellow: recording with an unavailable/failed sensor, dropped sample, or
  missed sampling deadline.
- Red: fatal flash, retention, upload, or recording error.

Sensor failure indication is sticky for the current recording. Fatal errors
remain red for 1.5 seconds before shutdown, and the LED is switched off before
System OFF to avoid wasting battery power.

## USB log transfer

`host/receive_logs.py` is the MVP PC receiver. It uses the board's existing USB
CDC serial port and a COBS-framed, CRC-protected binary protocol. Transfer runs
only during the boot window, before the sampling threads start, so binary data
cannot interleave with recorder status messages.

The host stores every raw sector, classifies the log from its sector CRCs,
sequence numbers, and commit marker, and extracts valid payload records into a
normal `.bin` file. Only after durable storage does it return the exact
transfer summary with permission to erase those flash sectors. See
[`host/README.md`](host/README.md) for setup and usage.

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

## Flash checkpoint retention

The top 4 KiB nRF52840 RAM section is excluded from normal SRAM and retained
across System OFF. It contains a magic, format version, clean/dirty marker,
the flash ring read/write pointers, the oldest and next log IDs, an inverted
copy of the next ID, and a CRC.

At boot:

1. If retained RAM contains a clean checkpoint, validate the write sector,
   oldest sector, and newest commit sector, then restore the ring directly.
2. If the checkpoint is dirty, invalid, or fails a boundary check, scan
   external flash and recover the ring normally.
3. If a scan finds empty flash but retained RAM is valid, preserve its next
   log ID; otherwise begin at zero.

The checkpoint is marked dirty before any upload, erase, or recording can
modify flash. It is marked clean only after a recording commit or upload
session completes successfully. Thus normal System OFF wake avoids the full
32 MiB scan, while interrupted operations safely fall back to recovery. The
retained next ID also preserves IDs after the host has emptied flash.

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

The UF2 output is
`firmware/nrf52840/recorder/build/recorder/zephyr/zephyr.uf2`.
