#ifndef SUS_FLASH_H_
#define SUS_FLASH_H_

#include <stdbool.h>
#include <stdint.h>

bool flash_is_empty(void);
bool flash_is_full(void);
void flash_scan(void);
uint8_t flash_read_log(void);
void read_flash(void);

#endif /* SUS_FLASH_H_ */
