#ifndef SUS_AS5600_H_
#define SUS_AS5600_H_

#include <stdbool.h>
#include <stdint.h>

#include <zephyr/device.h>

struct sus_as5600 {
	const struct device *i2c;
	bool available;
};

int sus_as5600_init(struct sus_as5600 *sensor);
int sus_as5600_read(const struct sus_as5600 *sensor, uint16_t *angle);

#endif /* SUS_AS5600_H_ */
