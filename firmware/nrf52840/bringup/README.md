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

`src/sensor_reader.c` owns sensor discovery, sampling, and unit conversion.
`src/board_power.c` brings up the switched peripheral rail before Zephyr
initializes its sensor drivers. `src/flash_smoke_test.c` contains a read-only
SPI flash diagnostic. `src/main.c` owns the console output loop, leaving the
reader reusable by a future recorder.

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

The hardware and operating-point assumptions live in `app.overlay`:

- `D6`: `V_periph` enable
- `D7`: flash `CS#`
- `D4/D5`: I2C
- `D8/D9/D10`: SPI clock/MISO/MOSI
- I2C runs at 400 kHz
- LSM6DSOX/LSM6DSO32 run at 208 Hz
- MMC5603 runs in 100 Hz continuous mode with automatic set/reset
- LIS3MDL runs at 155 Hz and +/-8 gauss

If `V_periph` enable is actually on `D3`, edit `app.overlay` before building:

```dts
vperiph-en-gpios = <&gpio0 29 GPIO_ACTIVE_HIGH>;
```
