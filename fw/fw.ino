#include <Arduino.h>
#include <Wire.h>

#include <Adafruit_LIS3DH.h>
#include <Adafruit_AS5600.h>
#include <Adafruit_MMC56x3.h>
#include <Adafruit_Sensor.h>

#include "FS.h"
#include "SD_MMC.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "esp_sleep.h"
#include "driver/rtc_io.h"

#include "sensor_reader.h"


static constexpr gpio_num_t WAKE_PIN = GPIO_NUM_10;
static constexpr gpio_num_t PERIPH_EN = GPIO_NUM_45;
static constexpr uint8_t BUTTON_HOLD = 5; // 0.5s at 10Hz

static constexpr uint32_t SAMPLE_HZ = 100;

// SparkFun Thing Plus ESP32-S3 (from SparkFun schematic pin labels)
/*static constexpr int SD_CLK = 38;
static constexpr int SD_CMD = 34;
static constexpr int SD_D0  = 39;
static constexpr int SD_D1  = 40;
static constexpr int SD_D2  = 47;
static constexpr int SD_D3  = 33;*/
// static constexpr int SD_DET = 48; // optional card-detect input

// Two LIS3DH I2C addresses (set by SA0 pin on each sensor)
static constexpr uint8_t LIS1_ADDR = 0x18;
static constexpr uint8_t LIS2_ADDR = 0x19;

// MMC5603 default I2C address is 0x30 in Adafruit libs
static constexpr uint8_t MMC_ADDR  = MMC56X3_DEFAULT_ADDRESS;

#pragma pack(push, 1)
struct LogRecord {
  uint32_t t_ms; // 4B
  uint32_t seq; // 4B
  int16_t  lis1[3]; // 6B
  int16_t  lis2[3]; // 6B
  int16_t  mmc_mG[3]; // 6B mG
  uint16_t angle; // 2B
  int32_t temp_C; // 4B
};
#pragma pack(pop)

static_assert(sizeof(LogRecord) == 32, "Unexpected LogRecord size");

static Adafruit_LIS3DH lis1(&Wire);
static Adafruit_LIS3DH lis2(&Wire);
static Adafruit_MMC5603 mmc(12345); // sensor ID (arbitrary)
static Adafruit_AS5600 as5600;

static SemaphoreHandle_t sdMutex;
static SemaphoreHandle_t i2cMutex;
static QueueHandle_t logQ;

static volatile uint32_t dropped = 0;

static File f;

static uint32_t now_ms() {
  // esp_timer_get_time() returns microseconds since boot on ESP32 Arduino
  return (uint32_t)(esp_timer_get_time() / 1000ULL);
}

static String make_log_filename(fs::FS &fs) {
  // Find first unused /logNNN.bin
  for (int i = 0; i < 1000; i++) {
    char name[16];
    snprintf(name, sizeof(name), "/log%03d.bin", i);
    if (!fs.exists(name)) return String(name);
  }
  return String("/log999.bin");
}

bool initAS5600(){
  if (!as5600.begin()){
    Serial.println("Failed to init AS5600");
    return false;
  }

  Serial.println("AS5600 found!");

  as5600.enableWatchdog(false);
  // Normal (high) power mode
  as5600.setPowerMode(AS5600_POWER_MODE_NOM);
  // No Hysteresis
  as5600.setHysteresis(AS5600_HYSTERESIS_OFF);

  // analog output
  as5600.setOutputStage(AS5600_OUTPUT_STAGE_ANALOG_FULL);

  // setup filters
  as5600.setSlowFilter(AS5600_SLOW_FILTER_16X);
  as5600.setFastFilterThresh(AS5600_FAST_FILTER_THRESH_SLOW_ONLY);

  // Reset position settings to defaults
  as5600.setZPosition(0);
  as5600.setMPosition(4095);
  as5600.setMaxAngle(4095);

  return true;
}

void sensorTask(void *param) {
  (void)param;

  bool lis1_conn = false;
  bool lis2_conn = false;
  bool mmc_conn = false;
  bool as_conn = false;

  // Configure sensors
  xSemaphoreTake(i2cMutex, portMAX_DELAY);

  // Faster I2C helps at 200 Hz with 3 sensors
  Wire.setClock(200000); // 1 MHz (ESP32-S3 supports fast I2C; drop to 400k if needed)

  if (!lis1.begin(LIS1_ADDR)) {
    Serial.println("LIS1 not found");
  } else {
    lis1_conn = true;
    lis1.setDataRate(LIS3DH_DATARATE_200_HZ);
    lis1.setRange(LIS3DH_RANGE_16_G);
    lis1.setPerformanceMode(LIS3DH_MODE_HIGH_RESOLUTION);
  }
  if (!lis2.begin(LIS2_ADDR)) {
    Serial.println("LIS2 not found");
  } else {
    lis2_conn = true;
    lis2.setDataRate(LIS3DH_DATARATE_200_HZ);
    lis2.setRange(LIS3DH_RANGE_16_G);
    lis2.setPerformanceMode(LIS3DH_MODE_HIGH_RESOLUTION);
  }

  if (!mmc.begin(MMC_ADDR, &Wire)) {
    Serial.println("MMC5603 not found");
  } else {
    mmc_conn = true;
    mmc.setDataRate(200); // in Hz, from 1-255 or 1000
    mmc.setContinuousMode(true);
  }

  // Set up the AS5600 (angle sensor)
  as_conn = initAS5600();

  xSemaphoreGive(i2cMutex);

  uint32_t seq = 0;

  // Use vTaskDelayUntil for stable timing
  TickType_t lastWake = xTaskGetTickCount();
  const TickType_t periodTicks = pdMS_TO_TICKS(1000 / SAMPLE_HZ); // 5ms at 200Hz

  Serial.println("Starting sensor reading");

  while (true) {
    // If you want closer-to-exact 200Hz than tick resolution, you can do a micros-based delay loop.
    vTaskDelayUntil(&lastWake, periodTicks);

    LogRecord r{};
    r.t_ms = now_ms();
    r.seq  = seq++;

    sensors_event_t magEvent;

    xSemaphoreTake(i2cMutex, portMAX_DELAY);

    // LIS3DH: read() updates lis.x/y/z (raw int16 counts)
    if (lis1_conn){
      lis1.read();

      r.lis1[0] = (int16_t)round(lis1.x_g * 1000.0f);
      r.lis1[1] = (int16_t)round(lis1.y_g * 1000.0f);
      r.lis1[2] = (int16_t)round(lis1.z_g * 1000.0f);
    }

    if (lis2_conn){
      lis2.read();

      r.lis2[0] = (int16_t)round(lis2.x_g * 1000.0f);
      r.lis2[1] = (int16_t)round(lis2.y_g * 1000.0f);
      r.lis2[2] = (int16_t)round(lis2.z_g * 1000.0f);
    }

    // read AS5600 angle
    if (as_conn){
      uint16_t rawAngle = as5600.getRawAngle();
      r.angle = rawAngle;
    }

    if (mmc_conn){
      // MMC5603: getEvent returns microtesla floats in event.magnetic.{x,y,z}
      // Store as int16 milliGauss to preserve precision and keep records fixed-size.
      mmc.getEvent(&magEvent);

      float temp_c = mmc.readTemperature();


      r.mmc_mG[0] = (int16_t)llround((double)magEvent.magnetic.x * 10.0); // uT -> mG
      r.mmc_mG[1] = (int16_t)llround((double)magEvent.magnetic.y * 10.0);
      r.mmc_mG[2] = (int16_t)llround((double)magEvent.magnetic.z * 10.0);

      r.temp_C = (int32_t)round(temp_c * 10.0);
    }
    
    xSemaphoreGive(i2cMutex);

    if (xQueueSend(logQ, &r, 0) != pdTRUE) {
      dropped++;
    }
  }
}


void buttonTask(void *param) {
  (void)param;
  // set up the sleepy pin, keep an eye on it, then put us to sleep if its hit

  // wait for the wake pin to be released
  while (digitalRead(WAKE_PIN) == LOW){
    vTaskDelay(pdMS_TO_TICKS(100));
  }

  uint8_t button_held_count = 0;
  while (1){
    vTaskDelay(pdMS_TO_TICKS(100));
    if (digitalRead(WAKE_PIN) == LOW){
      button_held_count++;
      if (button_held_count >= BUTTON_HOLD){
        goToSleep();
      }
    } else {
      button_held_count = 0;
    }
  }
}


void goToSleep() {
  xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(200));
  xSemaphoreTake(sdMutex, pdMS_TO_TICKS(200));
  f.close();
  SD_MMC.end();

  digitalWrite(PERIPH_EN, 0); // switch off periph/i2c Vreg

  // wait until button is released so we dont instawake
  while (digitalRead(WAKE_PIN) == LOW){
    vTaskDelay(pdMS_TO_TICKS(10));
  }

  // debounce
  vTaskDelay(pdMS_TO_TICKS(50));

  // Configure wake on button press (pin pulled LOW)
  rtc_gpio_pullup_en(WAKE_PIN);
  rtc_gpio_pulldown_dis(WAKE_PIN);
  esp_sleep_enable_ext0_wakeup(WAKE_PIN, 0);
  esp_deep_sleep_start();
}


void writerTask(void *param) {
  (void)param;
  
  xSemaphoreTake(sdMutex, portMAX_DELAY);

  // On ESP32-S3 you often need to set SDMMC pins explicitly. :contentReference[oaicite:4]{index=4}
  if (!SD_MMC.setPins(SDIO_CLK, SDIO_CMD, SDIO0, SDIO1, SDIO2, SDIO3)) {
    Serial.println("SD_MMC.setPins failed");
    while (true) vTaskDelay(pdMS_TO_TICKS(1000));
  }

  // begin(mountpoint, oneBitMode=false) -> 4-bit mode
  if (!SD_MMC.begin("/sdcard", /*oneBitMode=*/false)) {
    Serial.println("SD_MMC mount failed");
    while (true) vTaskDelay(pdMS_TO_TICKS(1000));
  }

  String filename = make_log_filename(SD_MMC);
  Serial.printf("Logging to %s\n", filename.c_str());

  f = SD_MMC.open(filename.c_str(), FILE_WRITE);
  if (!f) {
    Serial.println("Failed to open log file");
    while (true) vTaskDelay(pdMS_TO_TICKS(1000));
  }

  xSemaphoreGive(sdMutex);

  // Simple block accumulator (~512B) to reduce small writes
  uint8_t block[512];
  size_t  used = 0;

  uint32_t lastFlushMs = now_ms();

  while (true) {
    LogRecord r;
    if (xQueueReceive(logQ, &r, pdMS_TO_TICKS(250)) == pdTRUE) {
      // Append record to block
      if (used + sizeof(r) > sizeof(block)) {
        // write full block
        xSemaphoreTake(sdMutex, portMAX_DELAY);
        f.write(block, used);
        xSemaphoreGive(sdMutex);
        used = 0;
      }
      memcpy(block + used, &r, sizeof(r));
      used += sizeof(r);
    }

    uint32_t t = now_ms();

    // Flush about once per second (tune as you like)
    if ((t - lastFlushMs) >= 1000) {
      xSemaphoreTake(sdMutex, portMAX_DELAY);
      if (used) {
        f.write(block, used);
        used = 0;
      }
      f.flush();
      xSemaphoreGive(sdMutex);
      lastFlushMs = t;

      Serial.printf("dropped=%lu, size=%llu bytes\n",
                    (unsigned long)dropped,
                    (unsigned long long)f.size());
    }
  }
}

void setup() {
  //rtc_gpio_deinit(WAKE_PIN);
  pinMode(WAKE_PIN, INPUT_PULLUP);
  pinMode(PERIPH_EN, OUTPUT);

  Serial.begin(115200);
  Serial.printf("Reset reason = %d\n", (int)esp_reset_reason());

  digitalWrite(PERIPH_EN, 1); // switch on periph/i2c Vreg
  delay(200);

  Wire.begin(); // uses board default SDA/SCL (SparkFun Qwiic is ESP8=SDA, ESP9=SCL per schematic) :contentReference[oaicite:5]{index=5}

  sdMutex = xSemaphoreCreateMutex();
  i2cMutex = xSemaphoreCreateMutex();

  // Queue holds LogRecord copies. Depth ~400 => ~2 seconds at 200Hz.
  logQ = xQueueCreate(/*queue_len=*/400, sizeof(LogRecord));
  if (!logQ || !i2cMutex) {
    Serial.println("Failed to create RTOS objects");
    while (true) delay(1000);
  }

  // Task stack sizes: bump if you add WiFi/networking + parsing
  xTaskCreatePinnedToCore(sensorTask, "sensor", 4096, nullptr, 2, nullptr, 1);
  xTaskCreatePinnedToCore(writerTask, "writer", 4096, nullptr, 1, nullptr, 0);
  xTaskCreate(buttonTask, "power", 4096, nullptr, 0, nullptr);
}

void loop() {
  // nothing; work happens in tasks
  vTaskDelay(pdMS_TO_TICKS(1000));
  Serial.println("Runnning all good!");
}

