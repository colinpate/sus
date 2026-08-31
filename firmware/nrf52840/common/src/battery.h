#ifndef SUS_BATTERY_H_
#define SUS_BATTERY_H_

#include <stdint.h>

/*
 * Enable the XIAO's battery divider and configure its ADC channel.  The
 * active-low enable remains asserted so P0.14 is never driven high while the
 * firmware is running.
 */
int sus_battery_init(void);

/* Read the divider and return the estimated battery voltage in millivolts. */
int sus_battery_read_mv(int32_t *battery_mv);

#endif /* SUS_BATTERY_H_ */
