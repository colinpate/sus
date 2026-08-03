from __future__ import annotations

import dataclasses
import enum
import struct
import zlib
from collections import deque
from time import monotonic
from typing import BinaryIO

PROTOCOL_MAGIC = 0x50535553
PROTOCOL_VERSION = 1
SECTOR_BYTES = 4096
DATA_MAGIC = 0x44535553
COMMIT_MAGIC = 0x45535553
PAYLOAD_BYTES = 4078
MAX_ENCODED_FRAME_BYTES = 8192

FRAME_HEADER = struct.Struct("<IBBHI")
FRAME_CRC = struct.Struct("<I")
CHUNK_HEADER = struct.Struct("<IIIH")
CHUNK_CRC = struct.Struct("<I")
INFO_PAYLOAD = struct.Struct("<III")
BEGIN_PAYLOAD = struct.Struct("<II")
SECTOR_PREFIX = struct.Struct("<II")
SUMMARY_PAYLOAD = struct.Struct("<IIIII")
DISPOSITION_PAYLOAD = struct.Struct("<IIIIII")


class Message(enum.IntEnum):
    HELLO = 1
    INFO = 2
    READ_NEXT = 3
    BEGIN = 4
    SECTOR = 5
    SECTOR_ACK = 6
    END = 7
    DISPOSITION = 8
    EMPTY = 9
    SESSION_DONE = 10
    ERROR = 11
    ERASE_COMPLETE = 12


class Disposition(enum.IntEnum):
    ERASE = 1
    RETRY = 2
    DONE = 3


@dataclasses.dataclass(frozen=True)
class Frame:
    message: Message
    token: int
    payload: bytes = b""


@dataclasses.dataclass(frozen=True)
class TransferSummary:
    log_id: int
    start_sector: int
    end_sector: int
    sector_count: int
    raw_crc: int

    @classmethod
    def unpack(cls, payload: bytes) -> "TransferSummary":
        if len(payload) != SUMMARY_PAYLOAD.size:
            raise ValueError("invalid transfer summary length")
        return cls(*SUMMARY_PAYLOAD.unpack(payload))

    def pack(self) -> bytes:
        return SUMMARY_PAYLOAD.pack(
            self.log_id,
            self.start_sector,
            self.end_sector,
            self.sector_count,
            self.raw_crc,
        )


@dataclasses.dataclass(frozen=True)
class LogClassification:
    status: str
    reason: str
    payload: bytes
    data_sector_count: int


def cobs_encode(data: bytes) -> bytes:
    output = bytearray([0])
    code_index = 0
    code = 1

    for byte in data:
        if byte == 0:
            output[code_index] = code
            code_index = len(output)
            output.append(0)
            code = 1
            continue
        output.append(byte)
        code += 1
        if code == 0xFF:
            output[code_index] = code
            code_index = len(output)
            output.append(0)
            code = 1

    output[code_index] = code
    return bytes(output)


def cobs_decode(data: bytes) -> bytes:
    output = bytearray()
    index = 0

    while index < len(data):
        code = data[index]
        index += 1
        if code == 0:
            raise ValueError("zero byte inside COBS frame")
        end = index + code - 1
        if end > len(data):
            raise ValueError("truncated COBS frame")
        output.extend(data[index:end])
        index = end
        if code != 0xFF and index < len(data):
            output.append(0)

    return bytes(output)


def encode_frame(frame: Frame) -> bytes:
    header = FRAME_HEADER.pack(
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        int(frame.message),
        len(frame.payload),
        frame.token,
    )
    decoded = header + frame.payload
    decoded += FRAME_CRC.pack(zlib.crc32(decoded))
    return b"\x00" + cobs_encode(decoded) + b"\x00"


def decode_frame(encoded: bytes) -> Frame:
    decoded = cobs_decode(encoded)
    minimum_length = FRAME_HEADER.size + FRAME_CRC.size
    if len(decoded) < minimum_length:
        raise ValueError("frame is too short")

    magic, version, message, payload_length, token = FRAME_HEADER.unpack_from(decoded)
    if magic != PROTOCOL_MAGIC:
        raise ValueError("frame magic does not match")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version {version}")
    if len(decoded) != FRAME_HEADER.size + payload_length + FRAME_CRC.size:
        raise ValueError("frame payload length does not match")

    expected_crc = FRAME_CRC.unpack_from(decoded, len(decoded) - FRAME_CRC.size)[0]
    actual_crc = zlib.crc32(decoded[:-FRAME_CRC.size])
    if actual_crc != expected_crc:
        raise ValueError("frame CRC does not match")

    try:
        message_type = Message(message)
    except ValueError as error:
        raise ValueError(f"unknown message type {message}") from error
    payload = decoded[FRAME_HEADER.size:-FRAME_CRC.size]
    return Frame(message_type, token, payload)


class FrameStream:
    def __init__(self) -> None:
        self._encoded = bytearray()
        self._overflow = False

    def feed(self, data: bytes) -> list[Frame]:
        frames: list[Frame] = []
        for byte in data:
            if byte != 0:
                if len(self._encoded) < MAX_ENCODED_FRAME_BYTES:
                    self._encoded.append(byte)
                else:
                    self._overflow = True
                continue
            if self._overflow:
                self._encoded.clear()
                self._overflow = False
                continue
            if not self._encoded:
                continue
            try:
                frames.append(decode_frame(bytes(self._encoded)))
            except ValueError:
                # Startup printk text or a damaged frame is discarded at
                # the next delimiter.
                pass
            self._encoded.clear()
        return frames


class SerialProtocol:
    def __init__(self, serial_port: BinaryIO) -> None:
        self.serial = serial_port
        self.stream = FrameStream()
        self.pending: deque[Frame] = deque()

    def send(self, message: Message, token: int = 0, payload: bytes = b"") -> None:
        wire = encode_frame(Frame(message, token, payload))
        written = self.serial.write(wire)
        if written is not None and written != len(wire):
            raise OSError(f"short serial write: {written} of {len(wire)} bytes")
        flush = getattr(self.serial, "flush", None)
        if flush is not None:
            flush()

    def receive(self, timeout: float) -> Frame:
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            if self.pending:
                return self.pending.popleft()
            waiting = int(getattr(self.serial, "in_waiting", 0))
            data = self.serial.read(min(512, waiting) if waiting > 0 else 1)
            if data:
                self.pending.extend(self.stream.feed(data))
        raise TimeoutError("timed out waiting for a protocol frame")


def classify_log(raw: bytes, expected_log_id: int) -> LogClassification:
    if not raw or len(raw) % SECTOR_BYTES != 0:
        return LogClassification(
            "corrupt", "raw length is not a non-empty sector multiple", b"", 0
        )

    payload_parts: list[bytes] = []
    expected_sequence = 0
    payload_crc = 0
    saw_commit = False

    for sector_index in range(0, len(raw), SECTOR_BYTES):
        sector = raw[sector_index : sector_index + SECTOR_BYTES]
        magic, log_id, sequence, payload_length = CHUNK_HEADER.unpack_from(sector)
        if magic == DATA_MAGIC:
            maximum_payload = PAYLOAD_BYTES
        elif magic == COMMIT_MAGIC:
            maximum_payload = 4
        else:
            return LogClassification(
                "corrupt",
                f"sector {sector_index // SECTOR_BYTES} has invalid magic",
                b"",
                len(payload_parts),
            )

        if payload_length > maximum_payload or (
            magic == COMMIT_MAGIC and payload_length != 4
        ):
            return LogClassification(
                "corrupt",
                f"sector {sector_index // SECTOR_BYTES} has invalid payload length",
                b"",
                len(payload_parts),
            )
        stored_crc = CHUNK_CRC.unpack_from(sector, SECTOR_BYTES - CHUNK_CRC.size)[0]
        actual_crc = zlib.crc32(sector[: CHUNK_HEADER.size + payload_length])
        if stored_crc != actual_crc:
            return LogClassification(
                "corrupt",
                f"sector {sector_index // SECTOR_BYTES} CRC does not match",
                b"",
                len(payload_parts),
            )
        if log_id != expected_log_id:
            return LogClassification(
                "corrupt",
                f"sector {sector_index // SECTOR_BYTES} has log ID {log_id}",
                b"",
                len(payload_parts),
            )
        if sequence != expected_sequence:
            return LogClassification(
                "corrupt",
                f"expected sequence {expected_sequence}, found {sequence}",
                b"",
                len(payload_parts),
            )

        payload = sector[CHUNK_HEADER.size : CHUNK_HEADER.size + payload_length]
        if magic == DATA_MAGIC:
            if saw_commit:
                return LogClassification(
                    "corrupt", "data appears after the commit sector", b"", len(payload_parts)
                )
            payload_parts.append(payload)
            payload_crc = zlib.crc32(payload, payload_crc)
            expected_sequence += 1
            continue

        if saw_commit or sector_index + SECTOR_BYTES != len(raw):
            return LogClassification(
                "corrupt", "commit is duplicated or is not the final sector", b"", len(payload_parts)
            )
        committed_crc = struct.unpack_from("<I", payload)[0]
        if committed_crc != payload_crc:
            return LogClassification(
                "corrupt", "commit payload CRC does not match", b"", len(payload_parts)
            )
        saw_commit = True

    extracted = b"".join(payload_parts)
    if saw_commit:
        return LogClassification(
            "valid", "commit and payload CRCs match", extracted, len(payload_parts)
        )
    return LogClassification(
        "incomplete",
        "valid data prefix has no commit sector",
        extracted,
        len(payload_parts),
    )
