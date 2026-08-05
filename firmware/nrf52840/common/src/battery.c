#include "battery.h"

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>

#include <zephyr/devicetree.h>
#include <zephyr/drivers/adc.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/kernel.h>

#define SUS_BOARD_NODE DT_NODELABEL(sus_board)
#define BATTERY_OUTPUT_OHMS \
	DT_PROP(SUS_BOARD_NODE, battery_output_ohms)
#define BATTERY_FULL_OHMS DT_PROP(SUS_BOARD_NODE, battery_full_ohms)

static const struct adc_dt_spec battery_adc =
	ADC_DT_SPEC_GET(SUS_BOARD_NODE);
static const struct gpio_dt_spec battery_read_enable =
	GPIO_DT_SPEC_GET(SUS_BOARD_NODE, battery_enable_gpios);
static bool battery_ready;
static bool calibration_pending;

int sus_battery_init(void)
{
	int err;

	if (battery_ready) {
		return 0;
	}

	if (!gpio_is_ready_dt(&battery_read_enable) ||
	    !adc_is_ready_dt(&battery_adc)) {
		return -ENODEV;
	}

	/*
	 * GPIO_OUTPUT_ACTIVE plus GPIO_ACTIVE_LOW drives P0.14 low.  The
	 * devicetree also marks it open-drain, so this firmware cannot source a
	 * high level onto the XIAO's READ_BAT_ENABLE circuit.
	 */
	err = gpio_pin_configure_dt(&battery_read_enable, GPIO_OUTPUT_ACTIVE);
	if (err != 0) {
		return err;
	}

	k_busy_wait(DT_PROP(SUS_BOARD_NODE, battery_enable_delay_us));
	err = adc_channel_setup_dt(&battery_adc);
	if (err != 0) {
		return err;
	}

	battery_ready = true;
	calibration_pending = true;
	return 0;
}

int sus_battery_read_mv(int32_t *battery_mv)
{
	int16_t raw;
	int32_t divider_mv;
	int64_t scaled_mv;
	struct adc_sequence sequence = {
		.buffer = &raw,
		.buffer_size = sizeof(raw),
		.calibrate = calibration_pending,
	};
	int err;

	if (battery_mv == NULL) {
		return -EINVAL;
	}
	if (!battery_ready) {
		return -EACCES;
	}

	err = adc_sequence_init_dt(&battery_adc, &sequence);
	if (err != 0) {
		return err;
	}
	err = adc_read_dt(&battery_adc, &sequence);
	if (err != 0) {
		return err;
	}
	calibration_pending = false;

	divider_mv = raw;
	err = adc_raw_to_millivolts_dt(&battery_adc, &divider_mv);
	if (err != 0) {
		return err;
	}

	scaled_mv = (int64_t)divider_mv * BATTERY_FULL_OHMS;
	*battery_mv = (int32_t)((scaled_mv + BATTERY_OUTPUT_OHMS / 2) /
				BATTERY_OUTPUT_OHMS);
	return 0;
}
