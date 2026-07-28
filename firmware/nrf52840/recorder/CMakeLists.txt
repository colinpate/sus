cmake_minimum_required(VERSION 3.20.0)

set(COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../common)
list(APPEND DTS_ROOT ${COMMON_DIR})

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(sus_recorder)

target_include_directories(app PRIVATE
	${COMMON_DIR}/src
	${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_sources(app PRIVATE
	${COMMON_DIR}/src/board_power.c
	${COMMON_DIR}/src/flash.c
	${COMMON_DIR}/src/flash_zephyr.c
	${COMMON_DIR}/src/sensor_reader.c
	src/as5600.c
	src/log_id_retention.c
	src/main.c
)

target_compile_options(app PRIVATE
	-Wextra
	-Werror
)
