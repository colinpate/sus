#include "status_led.h"

#include <errno.h>
#include <stdbool.h>

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/gpio.h>

static const struct gpio_dt_spec red_led =
	GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);
static const struct gpio_dt_spec green_led =
	GPIO_DT_SPEC_GET(DT_ALIAS(led1), gpios);
static const struct gpio_dt_spec blue_led =
	GPIO_DT_SPEC_GET(DT_ALIAS(led2), gpios);

static bool status_led_ready;

int status_led_init(void)
{
	int err;

	if (!gpio_is_ready_dt(&red_led) ||
	    !gpio_is_ready_dt(&green_led) ||
	    !gpio_is_ready_dt(&blue_led)) {
		return -ENODEV;
	}

	err = gpio_pin_configure_dt(&red_led, GPIO_OUTPUT_INACTIVE);
	if (err != 0) {
		return err;
	}
	err = gpio_pin_configure_dt(&green_led, GPIO_OUTPUT_INACTIVE);
	if (err != 0) {
		return err;
	}
	err = gpio_pin_configure_dt(&blue_led, GPIO_OUTPUT_INACTIVE);
	if (err != 0) {
		return err;
	}

	status_led_ready = true;
	return 0;
}

void status_led_set(enum status_led_color color)
{
	bool red;
	bool green;
	bool blue;

	if (!status_led_ready) {
		return;
	}

	red = color == STATUS_LED_RED || color == STATUS_LED_YELLOW ||
	      color == STATUS_LED_PURPLE || color == STATUS_LED_WHITE;
	green = color == STATUS_LED_GREEN || color == STATUS_LED_YELLOW ||
		color == STATUS_LED_WHITE;
	blue = color == STATUS_LED_BLUE || color == STATUS_LED_PURPLE ||
	       color == STATUS_LED_WHITE;

	(void)gpio_pin_set_dt(&red_led, red ? 1 : 0);
	(void)gpio_pin_set_dt(&green_led, green ? 1 : 0);
	(void)gpio_pin_set_dt(&blue_led, blue ? 1 : 0);
}
