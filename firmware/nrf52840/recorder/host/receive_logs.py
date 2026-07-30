#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import struct
import tempfile
import time
import zlib
from pathlib import Path

from sus_protocol import (
    BEGIN_PAYLOAD,
    DISPOSITION_PAYLOAD,
    INFO_PAYLOAD,
    SECTOR_BYTES,
    SECTOR_PREFIX,
    Disposition,
    Frame,
    Message,
    SerialProtocol,
    TransferSummary,
    classify_log,
)

FRAME_TIMEOUT_SECONDS = 12.0
HANDSHAKE_TIMEOUT_SECONDS = 90.0
HELLO_RETRY_SECONDS = 2.0


def wait_for_message(
    protocol: SerialProtocol,
    expected: set[Message],
    timeout: float = FRAME_TIMEOUT_SECONDS,
) -> Frame:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = protocol.receive(max(0.01, deadline - time.monotonic()))
        if frame.message in expected:
            return frame
    raise TimeoutError(f"timed out waiting for {sorted(expected)}")


def handshake(
    protocol: SerialProtocol,
    timeout: float = HANDSHAKE_TIMEOUT_SECONDS,
) -> tuple[int, int, bool]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        protocol.send(Message.HELLO)
        try:
            frame = wait_for_message(
                protocol,
                {Message.INFO},
                timeout=min(HELLO_RETRY_SECONDS, deadline - time.monotonic()),
            )
        except TimeoutError:
            continue
        if frame.token != 0 or len(frame.payload) != INFO_PAYLOAD.size:
            raise RuntimeError("device returned malformed INFO")
        sector_bytes, sector_count, empty = INFO_PAYLOAD.unpack(frame.payload)
        if sector_bytes != SECTOR_BYTES:
            raise RuntimeError(f"device sector size is {sector_bytes}, expected {SECTOR_BYTES}")
        return sector_bytes, sector_count, bool(empty)
    raise TimeoutError("device did not answer HELLO; reset it and retry")


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def persist_transfer(
    partial_path: Path,
    output_directory: Path,
    summary: TransferSummary,
) -> tuple[Path, Path | None, str, str]:
    raw = partial_path.read_bytes()
    classification = classify_log(raw, summary.log_id)
    stem = (
        f"log-{summary.log_id:010d}"
        f"-s{summary.start_sector:08x}"
        f"-e{summary.end_sector:08x}"
        f"-{summary.raw_crc:08x}"
        f"-{classification.status}"
    )
    raw_path = output_directory / f"{stem}.flash.bin"
    payload_path: Path | None = None

    os.replace(partial_path, raw_path)
    if classification.status != "corrupt":
        payload_path = output_directory / f"{stem}.bin"
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output_directory, prefix=f".{stem}-", delete=False
        ) as payload_file:
            payload_file.write(classification.payload)
            payload_file.flush()
            os.fsync(payload_file.fileno())
            payload_temp = Path(payload_file.name)
        os.replace(payload_temp, payload_path)
    fsync_directory(output_directory)
    return raw_path, payload_path, classification.status, classification.reason


def send_disposition(
    protocol: SerialProtocol,
    token: int,
    disposition: Disposition,
    summary: TransferSummary,
) -> None:
    protocol.send(
        Message.DISPOSITION,
        token,
        DISPOSITION_PAYLOAD.pack(
            int(disposition),
            summary.log_id,
            summary.start_sector,
            summary.end_sector,
            summary.sector_count,
            summary.raw_crc,
        ),
    )


def receive_transfer(
    protocol: SerialProtocol,
    begin: Frame,
    output_directory: Path,
    erase: bool,
) -> tuple[Path, str]:
    if begin.token == 0 or len(begin.payload) != BEGIN_PAYLOAD.size:
        raise RuntimeError("device returned malformed BEGIN")
    log_id, start_sector = BEGIN_PAYLOAD.unpack(begin.payload)
    expected_ordinal = 0
    raw_crc = 0

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=output_directory,
        prefix=f".log-{log_id:010d}-",
        suffix=".partial",
        delete=False,
    ) as partial_file:
        partial_path = Path(partial_file.name)
        try:
            while True:
                frame = wait_for_message(
                    protocol, {Message.SECTOR, Message.END, Message.ERROR}
                )
                if frame.message is Message.ERROR:
                    error_code = (
                        struct.unpack("<I", frame.payload)[0]
                        if len(frame.payload) == 4
                        else -1
                    )
                    raise RuntimeError(f"device reported flash error {error_code}")
                if frame.token != begin.token:
                    continue
                if frame.message is Message.SECTOR:
                    expected_length = SECTOR_PREFIX.size + SECTOR_BYTES
                    if len(frame.payload) != expected_length:
                        raise RuntimeError("device returned malformed SECTOR")
                    ordinal, _physical_sector = SECTOR_PREFIX.unpack_from(frame.payload)
                    if ordinal != expected_ordinal:
                        raise RuntimeError(
                            f"expected sector ordinal {expected_ordinal}, got {ordinal}"
                        )
                    raw_sector = frame.payload[SECTOR_PREFIX.size :]
                    partial_file.write(raw_sector)
                    raw_crc = zlib.crc32(raw_sector, raw_crc)
                    protocol.send(
                        Message.SECTOR_ACK,
                        begin.token,
                        struct.pack("<I", ordinal),
                    )
                    expected_ordinal += 1
                    continue

                summary = TransferSummary.unpack(frame.payload)
                if summary.log_id != log_id or summary.start_sector != start_sector:
                    raise RuntimeError("END does not match BEGIN")
                if summary.sector_count != expected_ordinal:
                    raise RuntimeError("END sector count does not match received data")
                if summary.raw_crc != raw_crc:
                    raise RuntimeError("END raw CRC does not match received data")
                partial_file.flush()
                os.fsync(partial_file.fileno())
                break
        except Exception:
            partial_file.close()
            partial_path.unlink(missing_ok=True)
            raise

    try:
        raw_path, payload_path, status, reason = persist_transfer(
            partial_path, output_directory, summary
        )
    except Exception:
        send_disposition(protocol, begin.token, Disposition.RETRY, summary)
        raise

    print(
        f"log {log_id}: {status}, {summary.sector_count} sectors; "
        f"{reason}\n  raw: {raw_path}"
    )
    if payload_path is not None:
        print(f"  payload: {payload_path}")

    send_disposition(
        protocol,
        begin.token,
        Disposition.ERASE if erase else Disposition.DONE,
        summary,
    )
    return raw_path, status


def receive_all(
    protocol: SerialProtocol,
    output_directory: Path,
    erase: bool = True,
) -> list[tuple[Path, str]]:
    output_directory.mkdir(parents=True, exist_ok=True)
    sector_bytes, sector_count, initially_empty = handshake(protocol)
    print(
        f"connected: protocol sector={sector_bytes} bytes, "
        f"flash={sector_count} sectors, empty={initially_empty}"
    )

    received: list[tuple[Path, str]] = []
    while True:
        protocol.send(Message.READ_NEXT)
        frame = wait_for_message(
            protocol, {Message.BEGIN, Message.EMPTY, Message.ERROR}
        )
        if frame.message is Message.EMPTY:
            protocol.send(Message.SESSION_DONE)
            return received
        if frame.message is Message.ERROR:
            raise RuntimeError("device reported an upload error")
        received.append(receive_transfer(protocol, frame, output_directory, erase))
        if not erase:
            return received


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive raw SUS flash logs over USB CDC serial."
    )
    parser.add_argument("--port", required=True, help="serial device path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("received_logs"),
        help="output directory (default: received_logs)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="receive one log but leave its flash sectors unerased",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import serial
    except ImportError:
        print("pyserial is required; install host/requirements.txt")
        return 2

    try:
        with serial.Serial(
            args.port,
            baudrate=115200,
            timeout=0.1,
            write_timeout=2.0,
        ) as serial_port:
            serial_port.reset_input_buffer()
            protocol = SerialProtocol(serial_port)
            received = receive_all(
                protocol, args.output.resolve(), erase=not args.keep
            )
    except (OSError, RuntimeError, TimeoutError, serial.SerialException) as error:
        print(f"receive failed: {error}")
        return 1

    print(f"session complete: received {len(received)} log(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
