#include <errno.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/usb/udc.h>
#include <zephyr/kernel.h>
#include <zephyr/pm/device.h>
#include <zephyr/sys/atomic.h>
#include <zephyr/sys/poweroff.h>
#include <zephyr/sys/printk.h>
#include <zephyr/sys/util.h>

#include "as5600.h"
#include "board_power.h"
#include "flash.h"
#include "flash_zephyr.h"
#include "log_id_retention.h"
#include "log_record.h"
#include "sensor_reader.h"

#define SAMPLE_INTERVAL_MS 5U
#define BUTTON_POLL_MS 20U
#define BUTTON_DEBOUNCE_MS 50U
#define BUTTON_LONG_PRESS_MS 800U
#define RECORD_QUEUE_DEPTH 128U
#define RECORDS_PER_SECTOR \
	(FLASH_LOG_PAYLOAD_BYTES / sizeof(struct sus_log_record))

BUILD_ASSERT(RECORDS_PER_SECTOR == 81U);

K_MSGQ_DEFINE(record_queue, sizeof(struct sus_log_record),
	      RECORD_QUEUE_DEPTH, 4);
K_SEM_DEFINE(sampler_stopped, 0, 1);

static const struct gpio_dt_spec record_button =
	GPIO_DT_SPEC_GET(DT_ALIAS(sw0), gpios);
static const struct device *const usb_controller =
	DEVICE_DT_GET(DT_NODELABEL(zephyr_udc0));

static struct sus_sensor_reader sensor_reader;
static struct sus_as5600 angle_sensor;
static atomic_t stop_requested;
static atomic_t dropped_records;
static atomic_t missed_deadlines;
static atomic_t angle_read_errors;
static bool record_button_ready;

static void make_record(struct sus_log_record *record)
{
	struct sus_sensor_sample sample;
	uint16_t angle;

	sus_sensor_reader_read(&sensor_reader, &sample);
	memset(record, 0, sizeof(*record));
	record->timestamp_ms = sample.timestamp_ms;
	record->sequence = sample.sequence;
	memcpy(record->imu1_accel_mg, sample.imu1.accel_mg,
	       sizeof(record->imu1_accel_mg));
	memcpy(record->imu2_accel_mg, sample.imu2.accel_mg,
	       sizeof(record->imu2_accel_mg));
	memcpy(record->imu1_gyro_dps10, sample.imu1.gyro_dps10,
	       sizeof(record->imu1_gyro_dps10));
	memcpy(record->imu2_gyro_dps10, sample.imu2.gyro_dps10,
	       sizeof(record->imu2_gyro_dps10));
	memcpy(record->mmc5603_mg, sample.mmc5603_mg,
	       sizeof(record->mmc5603_mg));
	memcpy(record->lis3mdl_mg, sample.lis3mdl_mg,
	       sizeof(record->lis3mdl_mg));
	record->temp_deci_c = sample.imu1.temp_deci_c;

	if (sus_as5600_read(&angle_sensor, &angle) != 0) {
		atomic_inc(&angle_read_errors);
	} else {
		record->angle = angle;
	}
}

static void sampler_thread(void *unused1, void *unused2, void *unused3)
{
	int64_t deadline = k_uptime_get();

	ARG_UNUSED(unused1);
	ARG_UNUSED(unused2);
	ARG_UNUSED(unused3);

	while (atomic_get(&stop_requested) == 0) {
		struct sus_log_record record;
		int64_t now;

		deadline += SAMPLE_INTERVAL_MS;
		k_sleep(K_TIMEOUT_ABS_MS(deadline));
		if (atomic_get(&stop_requested) != 0) {
			break;
		}

		make_record(&record);
		if (k_msgq_put(&record_queue, &record, K_NO_WAIT) != 0) {
			atomic_inc(&dropped_records);
		}

		now = k_uptime_get();
		if (now >= deadline + SAMPLE_INTERVAL_MS) {
			int64_t skipped =
				(now - deadline) / SAMPLE_INTERVAL_MS;

			deadline += skipped * SAMPLE_INTERVAL_MS;
			atomic_add(&missed_deadlines, (atomic_val_t)skipped);
		}
	}

	k_sem_give(&sampler_stopped);
}

static void button_thread(void *unused1, void *unused2, void *unused3)
{
	int64_t pressed_at = 0;

	ARG_UNUSED(unused1);
	ARG_UNUSED(unused2);
	ARG_UNUSED(unused3);

	/* Do not count the button press that woke the board as a stop press. */
	while (gpio_pin_get_dt(&record_button) > 0) {
		k_msleep(BUTTON_POLL_MS);
	}
	k_msleep(BUTTON_DEBOUNCE_MS);

	while (atomic_get(&stop_requested) == 0) {
		int pressed = gpio_pin_get_dt(&record_button);

		if (pressed > 0) {
			if (pressed_at == 0) {
				pressed_at = k_uptime_get();
			} else if (k_uptime_get() - pressed_at >=
				   BUTTON_LONG_PRESS_MS) {
				printk("Long press: stopping recording\n");
				atomic_set(&stop_requested, 1);
				break;
			}
		} else {
			pressed_at = 0;
		}
		k_msleep(BUTTON_POLL_MS);
	}
}

K_THREAD_DEFINE(sampler_thread_id, 3072, sampler_thread,
		NULL, NULL, NULL, 1, 0, SYS_FOREVER_MS);
K_THREAD_DEFINE(button_thread_id, 1024, button_thread,
		NULL, NULL, NULL, 2, 0, SYS_FOREVER_MS);

struct record_accumulator {
	struct sus_log_record records[RECORDS_PER_SECTOR];
	size_t count;
};

static enum flash_log_result
flush_records(struct flash_log *log, struct record_accumulator *accumulator,
	      uint32_t *written_records)
{
	enum flash_log_result result;

	if (accumulator->count == 0U) {
		return FLASH_LOG_OK;
	}

	result = flash_log_append(
		log, accumulator->records,
		accumulator->count * sizeof(accumulator->records[0]));
	if (result == FLASH_LOG_OK) {
		*written_records += (uint32_t)accumulator->count;
		accumulator->count = 0U;
	}
	return result;
}

static enum flash_log_result
queue_record(struct flash_log *log, struct record_accumulator *accumulator,
	     const struct sus_log_record *record, uint32_t *written_records)
{
	accumulator->records[accumulator->count++] = *record;
	if (accumulator->count == ARRAY_SIZE(accumulator->records)) {
		return flush_records(log, accumulator, written_records);
	}
	return FLASH_LOG_OK;
}

static void wait_for_button_release(void)
{
	while (gpio_pin_get_dt(&record_button) > 0) {
		k_msleep(BUTTON_POLL_MS);
	}
	k_msleep(BUTTON_DEBOUNCE_MS);
}

static void enter_system_off(struct flash_zephyr_storage *storage)
{
	int err;

	if (!record_button_ready) {
		printk("Cannot enter System OFF without a configured wake button\n");
		return;
	}
	wait_for_button_release();

	if (storage != NULL && storage->device != NULL) {
		err = pm_device_action_run(storage->device,
					   PM_DEVICE_ACTION_SUSPEND);
		if (err != 0 && err != -EALREADY) {
			printk("Flash deep-power-down failed: %d\n", err);
		}
		k_msleep(1);
	}

	err = board_peripheral_power_set_enabled(false);
	if (err != 0) {
		printk("Peripheral rail shutdown failed: %d\n", err);
	}

	err = gpio_pin_interrupt_configure_dt(&record_button,
					      GPIO_INT_LEVEL_ACTIVE);
	if (err != 0) {
		printk("Button wake configuration failed: %d\n", err);
		return;
	}

	printk("Entering System OFF; press D2 to record again\n");
	k_msleep(10);

	if (device_is_ready(usb_controller)) {
		(void)udc_disable(usb_controller);
	}
	sys_poweroff();
}

static int recorder_hardware_init(void)
{
	int err;

	if (!gpio_is_ready_dt(&record_button)) {
		return -ENODEV;
	}

	err = gpio_pin_configure_dt(&record_button, GPIO_INPUT);
	if (err != 0) {
		return err;
	}
	record_button_ready = true;
	if (!board_peripheral_power_is_ready()) {
		return -ENODEV;
	}

	sus_sensor_reader_init(&sensor_reader);
	err = sus_as5600_init(&angle_sensor);
	if (err != 0) {
		printk("AS5600 not available: %d; angle will be zero\n", err);
	}

	printk("Sensors: IMU1=%s IMU2=%s MMC5603=%s LIS3MDL=%s "
	       "AS5600=%s\n",
	       (sensor_reader.available & SUS_SENSOR_IMU1) != 0U ?
		       "yes" : "no",
	       (sensor_reader.available & SUS_SENSOR_IMU2) != 0U ?
		       "yes" : "no",
	       (sensor_reader.available & SUS_SENSOR_MMC5603) != 0U ?
		       "yes" : "no",
	       (sensor_reader.available & SUS_SENSOR_LIS3MDL) != 0U ?
		       "yes" : "no",
	       angle_sensor.available ? "yes" : "no");
	return 0;
}

int main(void)
{
	static struct flash_zephyr_storage storage;
	static struct flash_chunk scratch;
	static struct flash_log log;
	static struct record_accumulator accumulator;
	struct sus_log_record record;
	enum flash_log_result log_result;
	uint32_t retained_next_log_id;
	uint32_t active_log_id = 0;
	uint32_t written_records = 0;
	int64_t next_status_ms;
	bool writer_ok = true;
	int err;

	printk("\nSUS 200 Hz recorder starting\n");

	err = recorder_hardware_init();
	if (err != 0) {
		printk("Hardware initialization failed: %d\n", err);
		enter_system_off(NULL);
		return 0;
	}

	err = flash_zephyr_storage_init_default(&storage);
	if (err != 0) {
		printk("Flash initialization failed: %d\n", err);
		enter_system_off(&storage);
		return 0;
	}

	log_result = flash_log_init(
		&log, storage.sector_count, &scratch,
		&flash_zephyr_storage_ops, &storage, NULL, NULL);
	if (log_result != FLASH_LOG_OK) {
		printk("Flash log initialization failed: %d\n", log_result);
		enter_system_off(&storage);
		return 0;
	}

	printk("Scanning flash log...\n");
	log_result = flash_log_scan(&log);
	if (log_result != FLASH_LOG_OK) {
		printk("Flash scan failed: %d\n", log_result);
		enter_system_off(&storage);
		return 0;
	}

	err = log_id_retention_load(&retained_next_log_id);
	if (flash_log_is_empty(&log) && err == 0) {
		log.next_log_id = retained_next_log_id;
		printk("Restored next log ID %u from retained RAM\n",
		       retained_next_log_id);
	} else if (!flash_log_is_empty(&log)) {
		printk("Recovered next log ID %u from flash\n",
		       log.next_log_id);
	} else {
		printk("No retained log ID; starting at %u\n",
		       log.next_log_id);
	}

	err = log_id_retention_store(log.next_log_id);
	if (err != 0) {
		printk("Could not synchronize retained log ID: %d\n", err);
	}

	log_result = flash_log_begin(&log, &active_log_id);
	if (log_result != FLASH_LOG_OK) {
		printk("Could not begin log: %d\n", log_result);
		enter_system_off(&storage);
		return 0;
	}

	/* flash_log_begin() consumes an ID, so retain the new next ID now. */
	err = log_id_retention_store(log.next_log_id);
	if (err != 0) {
		printk("Could not retain next log ID %u: %d\n",
		       log.next_log_id, err);
	}

	printk("Recording log %u at 200 Hz; hold D2 for 0.8 s to stop\n",
	       active_log_id);
	k_thread_start(sampler_thread_id);
	k_thread_start(button_thread_id);
	next_status_ms = k_uptime_get() + 1000;

	while (atomic_get(&stop_requested) == 0) {
		if (k_msgq_get(&record_queue, &record, K_MSEC(100)) == 0) {
			log_result = queue_record(
				&log, &accumulator, &record,
				&written_records);
			if (log_result != FLASH_LOG_OK) {
				printk("Flash append stopped: %d\n",
				       log_result);
				writer_ok = false;
				atomic_set(&stop_requested, 1);
				break;
			}
		}

		if (k_uptime_get() >= next_status_ms) {
			printk("log=%u records=%u queued=%u dropped=%ld "
			       "missed=%ld\n",
			       active_log_id, written_records,
			       k_msgq_num_used_get(&record_queue),
			       (long)atomic_get(&dropped_records),
			       (long)atomic_get(&missed_deadlines));
			next_status_ms += 1000;
		}
	}

	k_sem_take(&sampler_stopped, K_FOREVER);
	while (k_msgq_get(&record_queue, &record, K_NO_WAIT) == 0) {
		if (writer_ok) {
			log_result = queue_record(
				&log, &accumulator, &record,
				&written_records);
			if (log_result != FLASH_LOG_OK) {
				printk("Final flash append failed: %d\n",
				       log_result);
				writer_ok = false;
			}
		}
	}

	if (writer_ok) {
		log_result = flush_records(&log, &accumulator,
					   &written_records);
		if (log_result != FLASH_LOG_OK) {
			printk("Partial-sector write failed: %d\n",
			       log_result);
			writer_ok = false;
		}
	}

	log_result = flash_log_close(&log);
	if (log_result != FLASH_LOG_OK) {
		printk("Commit write failed: %d; boot recovery will retain "
		       "the valid prefix\n", log_result);
	} else {
		printk("Closed log %u: %u records, dropped=%ld, "
		       "missed=%ld, angle_errors=%ld%s\n",
		       active_log_id, written_records,
		       (long)atomic_get(&dropped_records),
		       (long)atomic_get(&missed_deadlines),
		       (long)atomic_get(&angle_read_errors),
		       writer_ok ? "" : " (recording stopped early)");
	}

	err = log_id_retention_store(log.next_log_id);
	if (err != 0) {
		printk("Final retained log ID update failed: %d\n", err);
	}

	enter_system_off(&storage);
	return 0;
}
