# Shared nRF52840 firmware support

This directory contains the board support and flash journal used by the
`bringup` and `recorder` applications:

- `sus_board.dtsi` defines the peripheral rail, sensors, and external SPI NOR.
- `src/as5600.*` configures and reads the shared raw-I2C angle sensor.
- `src/board_power.*` controls the switched peripheral rail.
- `src/sensor_reader.*` initializes and samples the shared sensors.
- `src/flash.*` implements the hardware-independent flash journal.
- `src/flash_zephyr.*` adapts the journal to Zephyr's SPI NOR driver.
- `tests/flash_logic` exercises the journal with a host-side memory model.

Application-specific entry points, behavior, and configuration remain in their
respective application directories.
