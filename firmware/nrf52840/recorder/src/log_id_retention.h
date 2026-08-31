#ifndef SUS_LOG_ID_RETENTION_H_
#define SUS_LOG_ID_RETENTION_H_

#include <stdbool.h>

#include "flash.h"

struct retained_log_state {
	struct flash_log_checkpoint checkpoint;
	bool clean;
};

int log_id_retention_load(struct retained_log_state *state);
int log_id_retention_store(const struct flash_log_checkpoint *checkpoint,
			   bool clean);

#endif /* SUS_LOG_ID_RETENTION_H_ */
