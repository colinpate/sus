# SUS board bringup

This Zephyr app does a minimal custom-board smoke test on the Seeed XIAO nRF52840:

- enables `V_periph`
- reads `WHO_AM_I` from the LSM6DSO32/LSM6DSOX at I2C address `0x6A`
- sends SPI flash command `0x9F` and prints the JEDEC ID
- blinks the onboard LED as a fallback status indicator

The XIAO board definition routes `printk()` to USB CDC ACM, so serial prints should work over USB. After flashing, open the board's USB serial port at 115200 baud. The baud rate is mostly ceremonial for USB CDC, but it keeps terminal tools happy.

Build with the same NCS install used by the blinky test:

```sh
ZEPHYR_BASE=/opt/nordic/ncs/v3.4.0/zephyr \
ZEPHYR_TOOLCHAIN_VARIANT=zephyr \
ZEPHYR_SDK_INSTALL_DIR=/opt/nordic/ncs/toolchains/ccc010f809/opt/zephyr-sdk \
PATH=/opt/nordic/ncs/toolchains/ccc010f809/bin:$PATH \
/opt/nordic/ncs/toolchains/ccc010f809/bin/cmake \
  -S firmware/bringup \
  -B build-sus-bringup \
  -GNinja \
  -DBOARD=xiao_ble \
  -DPython3_EXECUTABLE=/opt/nordic/ncs/toolchains/ccc010f809/bin/python3 \
  -DUSER_CACHE_DIR=/Users/colin/Documents/projects/sus/.zephyr-cache \
  -DUSE_CCACHE=0

ZEPHYR_BASE=/opt/nordic/ncs/v3.4.0/zephyr \
ZEPHYR_TOOLCHAIN_VARIANT=zephyr \
ZEPHYR_SDK_INSTALL_DIR=/opt/nordic/ncs/toolchains/ccc010f809/opt/zephyr-sdk \
PATH=/opt/nordic/ncs/toolchains/ccc010f809/bin:$PATH \
/opt/nordic/ncs/toolchains/ccc010f809/bin/cmake --build build-sus-bringup
```

If you are using the UF2 bootloader, copy this file to the mounted XIAO bootloader volume after a successful build:

```text
build-sus-bringup/zephyr/zephyr.uf2
```

Pin assumptions live in `app.overlay`:

- `D6`: `V_periph` enable
- `D7`: flash `CS#`
- `D8`: SPI `SCK`
- `D9`: SPI `MISO`
- `D10`: SPI `MOSI`
- `D4/D5`: I2C
- `D2`: IMU `INT1`

If `V_periph` enable is actually on `D3`, edit `app.overlay` before building:

```dts
vperiph-en-gpios = <&gpio0 29 GPIO_ACTIVE_HIGH>;
```
