#include "board_power.h"

#include <zephyr/devicetree.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/init.h>
#include <zephyr/kernel.h>

#define SUS_BOARD_NODE DT_NODELABEL(sus_board)

static const struct gpio_dt_spec peripheral_enable =
	GPIO_DT_SPEC_GET(SUS_BOARD_NODE, vperiph_en_gpios);
static bool peripheral_power_ready;

static int board_peripheral_power_init(void)
{
	int err;

	if (!gpio_is_ready_dt(&peripheral_enable)) {
		return 0;
	}

	err = gpio_pin_configure_dt(&peripheral_enable, GPIO_OUTPUT_ACTIVE);
	if (err != 0) {
		return 0;
	}

	/* Allow the switched sensor rail to settle before sensor driver init. */
	k_busy_wait(5000);
	peripheral_power_ready = true;
	return 0;
}

SYS_INIT(board_peripheral_power_init, POST_KERNEL, 0);

bool board_peripheral_power_is_ready(void)
{
	return peripheral_power_ready;
}
