#ifndef SUS_BOARD_POWER_H_
#define SUS_BOARD_POWER_H_

#include <stdbool.h>

bool board_peripheral_power_is_ready(void);
int board_peripheral_power_set_enabled(bool enabled);

#endif /* SUS_BOARD_POWER_H_ */
