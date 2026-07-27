# Flash logic host tests

These tests exercise the flash ring and log-transfer state machine without
Zephyr or a physical flash device. `fake_flash.c` provides a dense array of
erased, valid, or dirty sectors and records reads, erases, and transport
events.

Build and run directly with a host C compiler:

```sh
cc -std=c11 -Wall -Wextra -Wpedantic -Wconversion -Werror \
  -Ifirmware/nrf52840/bringup/src \
  firmware/nrf52840/bringup/src/flash.c \
  firmware/nrf52840/bringup/tests/flash_logic/fake_flash.c \
  firmware/nrf52840/bringup/tests/flash_logic/flash_log_test.c \
  -o /tmp/sus-flash-log-test

/tmp/sus-flash-log-test
```

Alternatively, configure this directory as a standalone CMake project and run
it through CTest:

```sh
cmake -S firmware/nrf52840/bringup/tests/flash_logic \
  -B /tmp/sus-flash-logic-build
cmake --build /tmp/sus-flash-logic-build
ctest --test-dir /tmp/sus-flash-logic-build --output-on-failure
```
