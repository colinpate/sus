# SUS sensor console

This Zephyr app is a minimal, modular sensor reader for the Seeed XIAO
nRF52840-based SUS board. It:

- enables `V_periph`
- initializes LSM6DSOX at `0x6A` and LSM6DSO32 at `0x6B`
- initializes MMC5603 at `0x30` and LIS3MDL at `0x1C`
- wakes the external SPI flash and checks its JEDEC ID
- reads all available sensors every 200 ms
- prints acceleration in mg, angular velocity in 0.1 dps, magnetic field in
  mG, and IMU temperature in degrees C over USB serial

This bring-up console remains intentionally simple. The separate
[`../recorder`](../recorder/README.md) application adds 200 Hz flash recording,
button-controlled System OFF, and retained log IDs.

Shared board support lives in [`../common`](../common/README.md).
`../common/src/sensor_reader.c` owns sensor discovery, sampling, and unit
conversion, while `../common/src/board_power.c` controls the switched
peripheral rail. The bring-up app keeps only its own console entry point and
`src/flash_smoke_test.c`, which reads the JEDEC ID through Zephyr's SPI NOR
driver.

The shared flash journal in `../common/src/flash.c` uses one 4 KiB sector per
data chunk and writes a separate commit sector when a log closes. A commit
stores the final sequence count and aggregate payload CRC as evidence for the
receiving host. The device does not try to classify a transferred log as
complete or corrupt. One physical sector remains reserved to keep the ring's
full and empty states unambiguous.

Initialize the production storage adapter with
`flash_zephyr_storage_init_default()`, pass `flash_zephyr_storage_ops` to
`flash_log_init()`, and then call `flash_log_scan()` before beginning a new
log. `flash_log_begin()`, `flash_log_append()`, and `flash_log_close()` form the
normal write path.

Startup recovery finds the physical ring range bounded by the oldest and
newest valid sectors. Dirty or invalid sectors in the free arc after the newest
sector are erased, including a partially programmed newest tail and remnants
of an interrupted old-log erase. Dirty or erased sectors inside the occupied
arc are preserved for transmission.

`flash_log_read_one()` snapshots the write pointer and streams raw sectors
starting at the read pointer. It stops before the first valid sector with a
different log ID, or at the write-pointer snapshot. The transfer summary
identifies the exact half-open sector range and includes a CRC over the raw
sector bytes. The host then returns:

- `FLASH_TRANSPORT_ERASE` after it has durably accepted that exact range
- `FLASH_TRANSPORT_RETRY` to retain and retransmit the range
- `FLASH_TRANSPORT_DONE` to stop without erasing the range

This also handles an incomplete newest log without a special device state:
boot recovery trims its dirty tail, the valid prefix is transmitted, and the
host detects the missing commit marker.

The XIAO board definition routes `printk()` to USB CDC ACM, so serial prints should work over USB. After flashing, open the board's USB serial port at 115200 baud. The baud rate is mostly ceremonial for USB CDC, but it keeps terminal tools happy.

Build with the same NCS install used by the blinky test:

```sh
ZEPHYR_BASE=/opt/nordic/ncs/v3.4.0/zephyr \
ZEPHYR_TOOLCHAIN_VARIANT=zephyr \
ZEPHYR_SDK_INSTALL_DIR=/opt/nordic/ncs/toolchains/ccc010f809/opt/zephyr-sdk \
PATH=/opt/nordic/ncs/toolchains/ccc010f809/bin:$PATH \
/opt/nordic/ncs/toolchains/ccc010f809/bin/cmake \
  -S firmware/nrf52840/bringup \
  -B firmware/nrf52840/bringup/build-sensors \
  -GNinja \
  -DBOARD=xiao_ble/nrf52840 \
  -DPython3_EXECUTABLE=/opt/nordic/ncs/toolchains/ccc010f809/bin/python3 \
  -DUSER_CACHE_DIR=/Users/colin/Documents/projects/sus/.zephyr-cache \
  -DUSE_CCACHE=0

ZEPHYR_BASE=/opt/nordic/ncs/v3.4.0/zephyr \
ZEPHYR_TOOLCHAIN_VARIANT=zephyr \
ZEPHYR_SDK_INSTALL_DIR=/opt/nordic/ncs/toolchains/ccc010f809/opt/zephyr-sdk \
PATH=/opt/nordic/ncs/toolchains/ccc010f809/bin:$PATH \
/opt/nordic/ncs/toolchains/ccc010f809/bin/cmake --build firmware/nrf52840/bringup/build-sensors
```

If you are using the UF2 bootloader, copy this file to the mounted XIAO bootloader volume after a successful build:

```text
firmware/nrf52840/bringup/build-sensors/zephyr/zephyr.uf2
```

The shared hardware and operating-point assumptions live in
`../common/sus_board.dtsi`; `app.overlay` includes that file:

- `D6`: `V_periph` enable
- `D7`: flash `CS#`
- `D4/D5`: I2C
- `D8/D9/D10`: SPI clock/MISO/MOSI
- external flash: MX25L25645GM2I-08G through Zephyr `jedec,spi-nor`
- SPI flash runs in mode 3 at 8 MHz and discovers geometry through runtime SFDP
- I2C runs at 400 kHz
- LSM6DSOX/LSM6DSO32 run at 208 Hz
- MMC5603 runs in 100 Hz continuous mode with automatic set/reset
- LIS3MDL runs at 155 Hz and +/-8 gauss

If `V_periph` enable is actually on `D3`, edit `../common/sus_board.dtsi`
before building:

```dts
vperiph-en-gpios = <&gpio0 29 GPIO_ACTIVE_HIGH>;
```
