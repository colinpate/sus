# SUS flash receiver MVP

This utility receives the recorder's raw 4 KiB flash sectors over its USB CDC
serial port. It verifies the transport CRC, classifies each log, saves it
durably, and then authorizes the recorder to erase that exact sector range.

The recorder waits five seconds for a receiver handshake during boot, before
sampling starts. If no handshake arrives, it records normally. Because upload
and recording never run together, the MVP can safely share the existing USB
console port.

## Setup and use

Create a virtual environment and install pySerial:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install \
  -r firmware/nrf52840/recorder/host/requirements.txt
```

Connect or wake the recorder, then start the receiver. The PC waits up to
90 seconds at a low HELLO rate, so it can run while the firmware is still
scanning flash and will catch the five-second post-scan upload window:

```sh
python3 firmware/nrf52840/recorder/host/receive_logs.py \
  --port /dev/cu.usbmodemXXXX \
  --output received_logs
```

On Linux the port will normally be `/dev/ttyACM0`. On macOS, list candidates
with `ls /dev/cu.usbmodem*`.

By default, every durably saved log receives an `ERASE` disposition. To test a
single transfer without removing it from flash, add `--keep`. The device then
retains the range and powers down after that transfer.

Each transfer produces:

- `*.flash.bin`: all raw sectors exactly as received
- `*.bin`: concatenated data payloads for valid or cleanly incomplete logs

The extracted `.bin` file is directly compatible with the repository's
`read_binary.py --format dual_mag` command. Corrupt logs retain only the raw
file.

## MVP protocol

Frames use COBS encoding, a zero-byte delimiter, fixed little-endian fields,
and CRC-32:

```text
uint32  magic       = 0x50535553
uint8   version     = 1
uint8   message
uint16  payload_length
uint32  transfer_token
uint8   payload[payload_length]
uint32  frame_crc
```

The decoded frame is COBS-encoded and surrounded by zero bytes. This lets the
receiver discard startup console text or a damaged frame at the next
delimiter.

The stop-and-wait exchange is:

```text
PC                 recorder
HELLO          ->
               <- INFO
READ_NEXT      ->
               <- BEGIN
               <- SECTOR
SECTOR_ACK     ->
                 ...one ACK per sector...
               <- END
ERASE/RETRY/
DONE           ->
```

`ERASE`, `RETRY`, and `DONE` echo the transfer token and the complete `END`
summary. The recorder accepts an erase only when the log ID, half-open sector
range, sector count, and raw CRC all match.

## Tests

The tests cover framing, damaged-frame resynchronization, valid/incomplete/
corrupt classification, and a complete simulated receive-save-erase session:

```sh
python3 -m unittest discover \
  -s firmware/nrf52840/recorder/host/tests -v
```
