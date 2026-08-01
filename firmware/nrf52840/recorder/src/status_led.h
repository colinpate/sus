#ifndef SUS_STATUS_LED_H_
#define SUS_STATUS_LED_H_

enum status_led_color {
	STATUS_LED_OFF,
	STATUS_LED_RED,
	STATUS_LED_GREEN,
	STATUS_LED_BLUE,
	STATUS_LED_YELLOW,
	STATUS_LED_PURPLE,
	STATUS_LED_WHITE,
};

int status_led_init(void);
void status_led_set(enum status_led_color color);

#endif /* SUS_STATUS_LED_H_ */
