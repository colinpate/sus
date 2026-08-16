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
#include "battery.h"
#include "board_power.h"
#include "flash.h"
#include "flash_serial.h"
#include "flash_zephyr.h"
#include "log_id_retention.h"
#include "log_record.h"
#include "sensor_reader.h"
#include "status_led.h"

#define SAMPLE_INTERVAL_MS 5U
#define BUTTON_POLL_MS 20U
#define BUTTON_DEBOUNCE_MS 50U
#define BUTTON_LONG_PRESS_MS 800U
#define UPLOAD_HELLO_WINDOW_MS 5000U
#define RECORD_QUEUE_DEPTH 128U
#define FLASH_SCAN_STATUS_INTERVAL_SECTORS 256U
#define FLASH_SCAN_STATUS_INTERVAL_MS 5000U
#define EXPECTED_SENSOR_MASK \
	(SUS_SENSOR_IMU1 | SUS_SENSOR_IMU2 | SUS_SENSOR_MMC5603 | \
	 SUS_SENSOR_LIS3MDL)
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
static const struct device *const serial_uart =
	DEVICE_DT_GET(DT_CHOSEN(zephyr_console));

static struct sus_sensor_reader sensor_reader;
static struct sus_as5600 angle_sensor;
static atomic_t stop_requested;
static atomic_t dropped_records;
static atomic_t missed_deadlines;
static atomic_t angle_read_errors;
static atomic_t sensor_fault_detected;
static bool record_button_ready;

struct flash_scan_status {
	int64_t started_ms;
	int64_t last_report_ms;
	uint32_t erased_sectors;
	bool erase_pending;
};

static void flash_scan_progress(void *context,
				enum flash_log_scan_phase phase,
				uint32_t completed_sectors,
				uint32_t total_sectors)
{
	struct flash_scan_status *status = context;
	int64_t now = k_uptime_get();
	uint32_t elapsed_seconds;
	uint32_t percent;
	bool report_due;

	if (phase == FLASH_LOG_SCAN_CLEANUP_ERASE) {
		status->erase_pending = true;
		if (status->erased_sectors != 0U &&
		    now - status->last_report_ms <
			    FLASH_SCAN_STATUS_INTERVAL_MS) {
			return;
		}
		elapsed_seconds = (uint32_t)((now - status->started_ms) /
					     1000);
		printk("Flash scan cleanup: starting erase %u after "
		       "%u/%u inspected, elapsed=%u s\n",
		       status->erased_sectors + 1U, completed_sectors,
		       total_sectors, elapsed_seconds);
		status->last_report_ms = now;
		return;
	}

	if (phase == FLASH_LOG_SCAN_CLEANUP && status->erase_pending) {
		status->erased_sectors++;
		status->erase_pending = false;
	}
	if (completed_sectors == 0U || total_sectors == 0U) {
		return;
	}

	report_due = completed_sectors == total_sectors ||
		     (completed_sectors %
		      FLASH_SCAN_STATUS_INTERVAL_SECTORS) == 0U ||
		     now - status->last_report_ms >=
			     FLASH_SCAN_STATUS_INTERVAL_MS;
	if (!report_due) {
		return;
	}

	elapsed_seconds = (uint32_t)((now - status->started_ms) / 1000);
	percent = (completed_sectors * 100U) / total_sectors;
	if (phase == FLASH_LOG_SCAN_DISCOVER) {
		printk("Flash scan discover: %u/%u inspected (%u%%), "
		       "elapsed=%u s\n",
		       completed_sectors, total_sectors, percent,
		       elapsed_seconds);
	} else {
		printk("Flash scan cleanup: %u/%u inspected (%u%%), "
		       "erased=%u, elapsed=%u s\n",
		       completed_sectors, total_sectors, percent,
		       status->erased_sectors, elapsed_seconds);
	}
	status->last_report_ms = now;
}

static void make_record(struct sus_log_record *record)
{
	struct sus_sensor_sample sample;
	uint16_t angle;

	sus_sensor_reader_read(&sensor_reader, &sample);
	if ((sample.valid & EXPECTED_SENSOR_MASK) != EXPECTED_SENSOR_MASK) {
		atomic_set(&sensor_fault_detected, 1);
	}
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
		atomic_set(&sensor_fault_detected, 1);
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
				atomic_set(&stop_requested, 1);
				status_led_set(STATUS_LED_OFF);
				printk("Long press: stopping recording\n");
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

	/* Hardware is quiescent; only now wait for a clean wake-button edge. */
	wait_for_button_release();

	err = gpio_pin_interrupt_configure_dt(&record_button,
					      GPIO_INT_LEVEL_ACTIVE);
	if (err != 0) {
		printk("Button wake configuration failed: %d\n", err);
		return;
	}

	printk("Entering System OFF; press D3 to record again\n");
	k_msleep(10);

	if (device_is_ready(usb_controller)) {
		(void)udc_disable(usb_controller);
	}
	status_led_set(STATUS_LED_OFF);
	sys_poweroff();
}

static void indicate_error(void)
{
	status_led_set(STATUS_LED_RED);
	/* Keep fatal errors visible before System OFF removes LED power. */
	k_msleep(1500);
}

static int retain_log_state(const struct flash_log *log, bool clean)
{
	struct flash_log_checkpoint checkpoint;

	flash_log_checkpoint_save(log, &checkpoint);
	return log_id_retention_store(&checkpoint, clean);
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
	static struct flash_serial_transport serial_transport;
	static struct record_accumulator accumulator;
	struct sus_log_record record;
	struct retained_log_state retained_state;
	struct flash_scan_status scan_status;
	enum flash_log_result log_result;
	uint32_t active_log_id = 0;
	uint32_t written_records = 0;
	int64_t next_status_ms;
	enum flash_serial_session_result upload_result;
	const struct flash_transport_ops *transport_ops = NULL;
	void *transport_context = NULL;
	bool writer_ok = true;
	bool serial_ready = false;
	bool retained_state_available = false;
	bool checkpoint_restored = false;
	bool recording_error = false;
	int32_t battery_mv;
	int err;

	printk("\nSUS 200 Hz recorder starting\n");
	err = sus_battery_init();
	if (err != 0) {
		printk("Battery monitor initialization failed: %d\n", err);
	} else {
		err = sus_battery_read_mv(&battery_mv);
		if (err != 0) {
			printk("Battery voltage read failed: %d\n", err);
		} else {
			printk("Battery: %d mV\n", battery_mv);
		}
	}
	err = status_led_init();
	if (err != 0) {
		printk("Status LED unavailable: %d\n", err);
	} else {
		status_led_set(STATUS_LED_WHITE);
	}

	err = recorder_hardware_init();
	if (err != 0) {
		printk("Hardware initialization failed: %d\n", err);
		indicate_error();
		enter_system_off(NULL);
		return 0;
	}

	err = flash_zephyr_storage_init_default(&storage);
	if (err != 0) {
		printk("Flash initialization failed: %d\n", err);
		indicate_error();
		enter_system_off(&storage);
		return 0;
	}

	err = flash_serial_transport_init(
		&serial_transport, serial_uart, storage.sector_count);
	if (err == 0) {
		transport_ops = &flash_serial_transport_ops;
		transport_context = &serial_transport;
		serial_ready = true;
	} else {
		printk("USB log upload unavailable: %d\n", err);
	}

	log_result = flash_log_init(
		&log, storage.sector_count, &scratch,
		&flash_zephyr_storage_ops, &storage, transport_ops,
		transport_context);
	if (log_result != FLASH_LOG_OK) {
		printk("Flash log initialization failed: %d\n", log_result);
		indicate_error();
		enter_system_off(&storage);
		return 0;
	}

	err = log_id_retention_load(&retained_state);
	if (err == 0) {
		retained_state_available = true;
		if (retained_state.clean) {
			log_result = flash_log_checkpoint_restore(
				&log, &retained_state.checkpoint);
			if (log_result == FLASH_LOG_OK) {
				checkpoint_restored = true;
				printk("Restored flash checkpoint: read=%u "
				       "write=%u next_log=%u\n",
				       log.read_sector, log.write_sector,
				       log.next_log_id);
			} else {
				printk("Retained flash checkpoint rejected: %d\n",
				       log_result);
			}
		} else {
			printk("Retained flash checkpoint is dirty; "
			       "full scan required\n");
		}
	} else {
		printk("Retained flash checkpoint unavailable: %d; "
		       "full scan required\n", err);
	}

	if (!checkpoint_restored) {
		status_led_set(STATUS_LED_PURPLE);
		scan_status = (struct flash_scan_status) {
			.started_ms = k_uptime_get(),
			.last_report_ms = k_uptime_get(),
		};
		printk("Scanning %u flash sectors...\n", log.sector_count);
		log_result = flash_log_scan_with_progress(
			&log, flash_scan_progress, &scan_status);
		if (log_result != FLASH_LOG_OK) {
			printk("Flash scan failed: %d\n", log_result);
			indicate_error();
			enter_system_off(&storage);
			return 0;
		}

		if (flash_log_is_empty(&log) && retained_state_available) {
			log.next_log_id =
				retained_state.checkpoint.next_log_id;
			printk("Restored next log ID %u from retained RAM\n",
			       log.next_log_id);
		} else if (!flash_log_is_empty(&log)) {
			printk("Recovered next log ID %u from flash\n",
			       log.next_log_id);
		} else {
			printk("No retained log ID; starting at %u\n",
			       log.next_log_id);
		}
	}

	/* Any flash mutation makes these pointers unsafe until re-checkpointed. */
	err = retain_log_state(&log, false);
	if (err != 0) {
		printk("Could not invalidate retained flash checkpoint: %d\n",
		       err);
		indicate_error();
		enter_system_off(&storage);
		return 0;
	}

	if (serial_ready) {
		status_led_set(STATUS_LED_BLUE);
		printk("Waiting %u ms for a USB log receiver...\n",
		       UPLOAD_HELLO_WINDOW_MS);
		upload_result = flash_serial_upload_session(
			&serial_transport, &log,
			UPLOAD_HELLO_WINDOW_MS);
		if (upload_result != FLASH_SERIAL_NO_HOST) {
			if (upload_result == FLASH_SERIAL_SESSION_COMPLETE) {
				err = retain_log_state(&log, true);
				if (err != 0) {
					printk("Post-upload checkpoint failed: %d\n",
					       err);
					indicate_error();
				}
			} else {
				indicate_error();
			}
			printk("USB upload session %s; powering down\n",
			       upload_result ==
					       FLASH_SERIAL_SESSION_COMPLETE ?
				       "complete" : "failed");
			enter_system_off(&storage);
			return 0;
		}
	}

	log_result = flash_log_begin(&log, &active_log_id);
	if (log_result != FLASH_LOG_OK) {
		printk("Could not begin log: %d\n", log_result);
		indicate_error();
		enter_system_off(&storage);
		return 0;
	}

	printk("Recording log %u at 200 Hz; hold D3 for 0.8 s to stop\n",
	       active_log_id);
	if ((sensor_reader.available & EXPECTED_SENSOR_MASK) ==
			EXPECTED_SENSOR_MASK && angle_sensor.available) {
		status_led_set(STATUS_LED_GREEN);
	} else {
		atomic_set(&sensor_fault_detected, 1);
		status_led_set(STATUS_LED_YELLOW);
	}
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
				status_led_set(STATUS_LED_RED);
				recording_error = true;
				writer_ok = false;
				atomic_set(&stop_requested, 1);
				break;
			}
		}

		if (k_uptime_get() >= next_status_ms) {
			if ((atomic_get(&sensor_fault_detected) != 0 ||
			     atomic_get(&dropped_records) != 0 ||
			     atomic_get(&missed_deadlines) != 0) && writer_ok) {
				status_led_set(STATUS_LED_YELLOW);
			}
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
				status_led_set(STATUS_LED_RED);
				recording_error = true;
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
			status_led_set(STATUS_LED_RED);
			recording_error = true;
			writer_ok = false;
		}
	}

	log_result = flash_log_close(&log);
	if (log_result != FLASH_LOG_OK) {
		status_led_set(STATUS_LED_RED);
		recording_error = true;
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

	if (log_result == FLASH_LOG_OK) {
		err = retain_log_state(&log, true);
		if (err != 0) {
			printk("Final flash checkpoint failed: %d\n", err);
			status_led_set(STATUS_LED_RED);
			recording_error = true;
		}
	}

	if (recording_error) {
		indicate_error();
	}
	enter_system_off(&storage);
	return 0;
}
