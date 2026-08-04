#ifndef __LOG_SERVER_H__
#define __LOG_SERVER_H__

#include "freertos/semphr.h"
#include <WebServer.h>

extern WebServer server;

void startWifiAndServer();

#endif // __LOG_SERVER_H__