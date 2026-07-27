cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(sus_board_bringup)

target_sources(app PRIVATE
	src/board_power.c
	src/flash_smoke_test.c
	src/main.c
	src/sensor_reader.c
)
