#ifndef SUS_LOG_ID_RETENTION_H_
#define SUS_LOG_ID_RETENTION_H_

#include <stdint.h>

int log_id_retention_load(uint32_t *next_log_id);
int log_id_retention_store(uint32_t next_log_id);

#endif /* SUS_LOG_ID_RETENTION_H_ */
