#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include "log_server.h"
#include "wifi_config.h"
#include "SD_MMC.h"

WebServer server(80);
// simple flag so we can stop logging before download
static volatile bool loggingActive = true;
static constexpr size_t kDownloadChunkSize = 2048;
// WebServer handles one request at a time, so a shared buffer avoids large
// per-request stack allocations in the web task.
static uint8_t downloadBuf[kDownloadChunkSize];

void startWifiAndServer() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.print("WiFi connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("WiFi OK, IP = ");
  Serial.println(WiFi.localIP());

  // mDNS so you can use http://sus-logger.local/ (often works on Mac/iOS)
  if (MDNS.begin("sus-logger")) {
    Serial.println("mDNS: http://sus-logger.local/");
  }

  server.on("/", HTTP_GET, []() {
    server.send(200, "text/plain",
      "OK\n"
      "GET /files\n"
      "GET /download?name=/log000.bin\n"
      "POST /stop\n");
  });

  // List files as JSON (minimal)
  server.on("/files", HTTP_GET, []() {
    //xSemaphoreTake(sdMutex, portMAX_DELAY);

    File root = SD_MMC.open("/");
    String out = "[";
    bool first = true;

    for (File f = root.openNextFile(); f; f = root.openNextFile()) {
      if (!f.isDirectory()) {
        String name = String("/") + String(f.name()); // name() often returns without leading /
        if (name.endsWith(".bin")) {
          if (!first) out += ",";
          first = false;
          out += "\"" + name + "\"";
        }
      }
      f.close();
    }
    root.close();

    //xSemaphoreGive(sdMutex);
    out += "]";
    server.send(200, "application/json", out);
  });

  // Stop logging (close file) so downloads are consistent
  server.on("/stop", HTTP_POST, []() {
    loggingActive = false;

    // Give writer a moment to notice and close (simple approach)
    delay(50);

    server.send(200, "text/plain", "logging stopped\n");
  });

  // Stream download
  server.on("/download", HTTP_GET, []() {
    if (!server.hasArg("name")) {
      server.send(400, "text/plain", "missing ?name=\n");
      return;
    }

    String name = server.arg("name");
    if (!name.startsWith("/")) name = "/" + name;

    // Basic safety: only allow .bin from root
    if (!name.endsWith(".bin") || name.indexOf("..") >= 0) {
      server.send(400, "text/plain", "bad name\n");
      return;
    }

    //xSemaphoreTake(sdMutex, portMAX_DELAY);
    File df = SD_MMC.open(name.c_str(), FILE_READ);
    if (!df) {
      //xSemaphoreGive(sdMutex);
      server.send(404, "text/plain", "not found\n");
      return;
    }

    // Send headers + stream file
    server.setContentLength(df.size());
    server.sendHeader("Content-Type", "application/octet-stream");
    server.sendHeader("Content-Disposition", String("attachment; filename=\"") + name.substring(1) + "\"");
    server.send(200);

    WiFiClient client = server.client();
    while (df.available() && client.connected()) {
      size_t n = df.read(downloadBuf, sizeof(downloadBuf));
      if (n == 0) break;
      client.write(downloadBuf, n);
      // yield to keep WiFi stack happy
      delay(0);
    }

    df.close();
    //xSemaphoreGive(sdMutex);
  });

  server.begin();
  Serial.println("HTTP server started");
}
