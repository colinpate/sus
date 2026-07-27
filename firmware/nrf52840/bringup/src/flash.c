#include <stdint.h>
#include <stdbool.h>
#include "flash.h"

#define SECTOR_BYTES 4096
#define CHUNK_METADATA_BYTES 18
#define PAYLOAD_BYTES (SECTOR_BYTES - CHUNK_METADATA_BYTES)
#define FLASH_START_ADDR 0
#define FLASH_END_ADDR (1 << 18)
#define SECTOR_ADDR_INCR SECTOR_BYTES

#define SERIAL_LOG_ACK 1
#define SERIAL_LOG_ERR 2
#define SERIAL_DONE 3
#define SERIAL_MAX_RETRIES 3

#define SECTOR_ERASED 1
#define SECTOR_CRC_FAIL 2

#define LOG_READ_INVALID 1
#define LOG_READ_SERIAL_ERR 2
#define LOG_READ_SERIAL_DONE 3
#define LOG_READ_EMPTY 4

uint32_t flash_write_addr;
uint32_t flash_read_addr;
uint32_t flash_log_id;

struct flash_chunk {
    uint32_t magic;
    uint32_t log_id;
    uint32_t sequence;
    uint16_t payload_length;
    uint8_t  payload[PAYLOAD_BYTES];
    uint32_t crc;
};

uint32_t flash_next_sector_addr(uint32_t addr){
    if ((addr + SECTOR_ADDR_INCR) < FLASH_END_ADDR) {
        return addr + SECTOR_ADDR_INCR;
    } else {
        return FLASH_START_ADDR;
    }
}

/*
 * Keep one sector unused so read_addr == write_addr is unambiguously empty.
 * write_addr points to the next free sector and read_addr points to the oldest
 * unread sector.
 */
bool flash_is_empty(void){
    return flash_write_addr == flash_read_addr;
}

bool flash_is_full(void){
    return flash_next_sector_addr(flash_write_addr) == flash_read_addr;
}

void update_log_crc(uint32_t* crc, struct flash_chunk* chunk){

}

void flash_read_sector(uint32_t addr, struct flash_chunk* chunk){

}

void flash_erase_sectors(uint32_t start_addr, uint32_t end_addr){
    // erase the half-open range [start_addr, end_addr)
    // wrap around if start > end
}

uint8_t check_chunk(struct flash_chunk* chunk){
    // if erased, return SECTOR_ERASED
    // if not erased but crc fail, return SECTOR_CRC_FAIL
    return 0;
}

void flash_scan(void){
    uint32_t lowest_seq = UINT32_MAX;
    uint32_t lowest_id = UINT32_MAX;
    uint32_t highest_seq = 0;
    uint32_t highest_id = 0;
    struct flash_chunk curr_chunk;

    // Presume empty
    flash_read_addr = FLASH_START_ADDR;
    flash_write_addr = flash_read_addr;
    bool valid_found = false;
    flash_log_id = 0;

    for (
        uint32_t search_addr = FLASH_START_ADDR; 
        search_addr < FLASH_END_ADDR; 
        search_addr += SECTOR_ADDR_INCR
    ){
        flash_read_sector(search_addr, &curr_chunk);
        uint8_t check_result = check_chunk(&curr_chunk);
        if (check_result == 0){
            // Valid CRC
            if (!valid_found){
                lowest_seq = curr_chunk.sequence;
                highest_seq = curr_chunk.sequence;
                lowest_id = curr_chunk.log_id;
                highest_id = curr_chunk.log_id;
                flash_read_addr = search_addr;
                flash_write_addr = flash_next_sector_addr(search_addr);
                valid_found = true;
            } else {
                if (curr_chunk.log_id < lowest_id){
                    lowest_id = curr_chunk.log_id;
                    lowest_seq = curr_chunk.sequence;
                    flash_read_addr = search_addr;
                } else if (curr_chunk.log_id == lowest_id){
                    if (curr_chunk.sequence < lowest_seq){
                        lowest_seq = curr_chunk.sequence;
                        flash_read_addr = search_addr;
                    }
                }

                if (curr_chunk.log_id > highest_id){
                    highest_id = curr_chunk.log_id;
                    highest_seq = curr_chunk.sequence;
                    flash_write_addr = flash_next_sector_addr(search_addr);
                } else if (curr_chunk.log_id == highest_id){
                    if (curr_chunk.sequence > highest_seq){
                        highest_seq = curr_chunk.sequence;
                        flash_write_addr = flash_next_sector_addr(search_addr);
                    }
                }
            }
        }
    }
    flash_log_id = valid_found ? highest_id + 1 : 0;
}

uint8_t flash_read_log(void){
    struct flash_chunk chunk;
    uint32_t addr = flash_read_addr;
    uint32_t log_crc = 0;
    uint32_t read_log_id = 0;
    bool first_read = true;

    if (flash_is_empty()){
        return LOG_READ_EMPTY;
    }

    while (addr != flash_write_addr){
        flash_read_sector(addr, &chunk);
        uint8_t check_result = check_chunk(&chunk);

        if (check_result != 0){
            return LOG_READ_INVALID;
        }

        if (first_read){
            read_log_id = chunk.log_id;
            serial_send_log_start(read_log_id);
            first_read = false;
        } else if (chunk.log_id != read_log_id){
            // addr still points to the first sector of the next log
            break;
        }

        serial_send_chunk(&chunk);
        update_log_crc(&log_crc, &chunk);
        addr = flash_next_sector_addr(addr);
    }

    uint8_t ret = serial_send_log_end(read_log_id, log_crc);
    if (ret == SERIAL_LOG_ACK){
        // receiver acknowledged
        // Erase previous log sectors and set read addr to start of next log
        flash_erase_sectors(flash_read_addr, addr);
        flash_read_addr = addr;
    } else if (ret == SERIAL_LOG_ERR) {
        // resend same log
        return LOG_READ_SERIAL_ERR;
    } else if (ret == SERIAL_DONE){
        return LOG_READ_SERIAL_DONE;
    }
    return 0;
}

void read_flash(void){
    uint8_t num_retries = 0;
    if (!flash_is_empty()){
        while (true){
            uint8_t ret = flash_read_log();
            if (ret == LOG_READ_SERIAL_ERR){
                if (num_retries >= SERIAL_MAX_RETRIES){
                    break;
                } else {
                    num_retries += 1;
                }
            } else if (
                (ret == LOG_READ_SERIAL_DONE) ||
                (ret == LOG_READ_INVALID) ||
                (ret == LOG_READ_EMPTY)
            ){
                break;
            } else if (flash_is_empty()){
                break;
            } else {
                // Give each log its own retry budget.
                num_retries = 0;
            }
        }
    }
}
