cmake_minimum_required(VERSION 3.20)

project(flash_logic_tests C)

enable_testing()

add_executable(flash_log_test
	../../src/flash.c
	fake_flash.c
	flash_log_test.c
)

target_include_directories(flash_log_test PRIVATE
	../../src
)

target_compile_features(flash_log_test PRIVATE c_std_11)
target_compile_options(flash_log_test PRIVATE
	-Wall
	-Wextra
	-Wpedantic
	-Wconversion
	-Werror
)

add_test(NAME flash_log_test COMMAND flash_log_test)
