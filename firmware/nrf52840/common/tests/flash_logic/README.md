# Flash logic host tests

These tests exercise log writing, boot recovery, the flash ring, and raw-sector
transfer without Zephyr or a physical flash device. `fake_flash.c` provides a
dense array of erased, valid, or dirty sectors and records reads, writes,
erases, transfer ranges, and host dispositions.

The recovery cases cover dirty and erased sectors inside an occupied range,
dirty newest tails, interrupted old-log erases, wrapped logs, retries, and
exact transfer boundaries. Commit sectors are written and transmitted, but
log validation is intentionally left to the receiving host.

Build and run directly with a host C compiler:

```sh
cc -std=c11 -Wall -Wextra -Wpedantic -Wconversion -Werror \
  -Ifirmware/nrf52840/common/src \
  firmware/nrf52840/common/src/flash.c \
  firmware/nrf52840/common/tests/flash_logic/fake_flash.c \
  firmware/nrf52840/common/tests/flash_logic/flash_log_test.c \
  -o /tmp/sus-flash-log-test

/tmp/sus-flash-log-test
```

Alternatively, configure this directory as a standalone CMake project and run
it through CTest:

```sh
cmake -S firmware/nrf52840/common/tests/flash_logic \
  -B /tmp/sus-flash-logic-build
cmake --build /tmp/sus-flash-logic-build
ctest --test-dir /tmp/sus-flash-logic-build --output-on-failure
```
