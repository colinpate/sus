from __future__ import annotations

import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

HOST_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOST_DIRECTORY))

from sus_protocol import (  # noqa: E402
    CHUNK_HEADER,
    COMMIT_MAGIC,
    DATA_MAGIC,
    DISPOSITION_PAYLOAD,
    INFO_PAYLOAD,
    MAX_ENCODED_FRAME_BYTES,
    PAYLOAD_BYTES,
    SECTOR_BYTES,
    SECTOR_PREFIX,
    SUMMARY_PAYLOAD,
    Disposition,
    Frame,
    FrameStream,
    Message,
    SerialProtocol,
    classify_log,
    decode_frame,
    encode_frame,
)
from receive_logs import receive_all, wait_for_serial_port  # noqa: E402


def make_sector(magic: int, log_id: int, sequence: int, payload: bytes) -> bytes:
    sector = bytearray(b"\xff" * SECTOR_BYTES)
    CHUNK_HEADER.pack_into(sector, 0, magic, log_id, sequence, len(payload))
    sector[CHUNK_HEADER.size : CHUNK_HEADER.size + len(payload)] = payload
    struct.pack_into(
        "<I",
        sector,
        SECTOR_BYTES - 4,
        zlib.crc32(sector[: CHUNK_HEADER.size + len(payload)]),
    )
    return bytes(sector)


def make_complete_log(log_id: int, payloads: list[bytes]) -> bytes:
    sectors = [
        make_sector(DATA_MAGIC, log_id, sequence, payload)
        for sequence, payload in enumerate(payloads)
    ]
    aggregate_crc = 0
    for payload in payloads:
        aggregate_crc = zlib.crc32(payload, aggregate_crc)
    sectors.append(
        make_sector(
            COMMIT_MAGIC,
            log_id,
            len(payloads),
            struct.pack("<I", aggregate_crc),
        )
    )
    return b"".join(sectors)


class PortWaitTests(unittest.TestCase):
    def test_waits_until_serial_port_appears(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            port = Path(temporary_directory) / "cu.usbmodem-test"

            def create_port(_delay: float) -> None:
                port.touch()

            with mock.patch(
                "receive_logs.time.sleep", side_effect=create_port
            ) as sleep_mock, mock.patch("builtins.print"):
                wait_for_serial_port(str(port))

            sleep_mock.assert_called_once_with(0.1)


class FramingTests(unittest.TestCase):
    def test_round_trip_zero_heavy_sector_frame(self) -> None:
        payload = bytes(range(256)) * 16
        frame = Frame(Message.SECTOR, 0x12345678, payload)
        wire = encode_frame(frame)

        self.assertEqual(wire[0], 0)
        self.assertEqual(wire[-1], 0)
        self.assertNotIn(0, wire[1:-1])
        self.assertEqual(decode_frame(wire[1:-1]), frame)

    def test_stream_discards_console_text_and_bad_crc(self) -> None:
        valid = encode_frame(Frame(Message.INFO, 0, b"info"))
        damaged = bytearray(encode_frame(Frame(Message.INFO, 0, b"bad")))
        damaged[-3] ^= 1
        stream = FrameStream()

        frames = stream.feed(b"startup printk text" + bytes(damaged) + valid)

        self.assertEqual(frames, [Frame(Message.INFO, 0, b"info")])

    def test_stream_recovers_after_oversized_garbage(self) -> None:
        valid = Frame(Message.EMPTY, 0)
        stream = FrameStream()

        frames = stream.feed(
            b"x" * (MAX_ENCODED_FRAME_BYTES + 1)
            + b"\x00"
            + encode_frame(valid)
        )

        self.assertEqual(frames, [valid])


class BufferedSerial:
    def __init__(self, wire: bytes) -> None:
        self.output = bytearray(wire)
        self.read_sizes: list[int] = []

    @property
    def in_waiting(self) -> int:
        return len(self.output)

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        result = bytes(self.output[:size])
        del self.output[:size]
        return result


class SerialProtocolTests(unittest.TestCase):
    def test_receive_reads_only_buffered_tail_bytes(self) -> None:
        expected = Frame(Message.SECTOR, 0x12345678, b"\xff" * SECTOR_BYTES)
        serial = BufferedSerial(encode_frame(expected))

        actual = SerialProtocol(serial).receive(1.0)

        self.assertEqual(actual, expected)
        self.assertEqual(serial.read_sizes[:-1], [512] * 8)
        self.assertGreater(serial.read_sizes[-1], 0)
        self.assertLess(serial.read_sizes[-1], 512)


class ClassificationTests(unittest.TestCase):
    def test_complete_log(self) -> None:
        raw = make_complete_log(7, [b"abc", b"\x00" * PAYLOAD_BYTES])

        result = classify_log(raw, 7)

        self.assertEqual(result.status, "valid")
        self.assertEqual(result.payload, b"abc" + b"\x00" * PAYLOAD_BYTES)
        self.assertEqual(result.data_sector_count, 2)

    def test_valid_prefix_without_commit_is_incomplete(self) -> None:
        raw = make_sector(DATA_MAGIC, 8, 0, b"partial")

        result = classify_log(raw, 8)

        self.assertEqual(result.status, "incomplete")
        self.assertEqual(result.payload, b"partial")

    def test_dirty_sector_is_corrupt(self) -> None:
        raw = bytearray(make_complete_log(9, [b"payload"]))
        raw[CHUNK_HEADER.size] ^= 1

        result = classify_log(bytes(raw), 9)

        self.assertEqual(result.status, "corrupt")
        self.assertIn("CRC", result.reason)

    def test_commit_payload_crc_mismatch_is_corrupt(self) -> None:
        data = make_sector(DATA_MAGIC, 10, 0, b"payload")
        bad_commit = make_sector(COMMIT_MAGIC, 10, 1, struct.pack("<I", 0))

        result = classify_log(data + bad_commit, 10)

        self.assertEqual(result.status, "corrupt")
        self.assertIn("payload CRC", result.reason)


class FakeDeviceSerial:
    def __init__(self, raw_log: bytes, log_id: int = 12) -> None:
        self.raw_log = raw_log
        self.log_id = log_id
        self.token = 0xA5A5
        self.output = bytearray()
        self.input_stream = FrameStream()
        self.next_ordinal = 0
        self.disposition: Disposition | None = None
        self.disposition_values: tuple[int, ...] | None = None
        self.erase_complete_count = 0
        self.session_done = False

    @property
    def sectors(self) -> list[bytes]:
        return [
            self.raw_log[offset : offset + SECTOR_BYTES]
            for offset in range(0, len(self.raw_log), SECTOR_BYTES)
        ]

    @property
    def in_waiting(self) -> int:
        return len(self.output)

    def queue(self, frame: Frame) -> None:
        self.output.extend(encode_frame(frame))

    def queue_sector_or_end(self) -> None:
        if self.next_ordinal < len(self.sectors):
            payload = SECTOR_PREFIX.pack(self.next_ordinal, self.next_ordinal)
            payload += self.sectors[self.next_ordinal]
            self.queue(Frame(Message.SECTOR, self.token, payload))
            return
        summary = SUMMARY_PAYLOAD.pack(
            self.log_id,
            0,
            len(self.sectors),
            len(self.sectors),
            zlib.crc32(self.raw_log),
        )
        self.queue(Frame(Message.END, self.token, summary))

    def write(self, data: bytes) -> int:
        for frame in self.input_stream.feed(data):
            if frame.message is Message.HELLO:
                self.queue(
                    Frame(
                        Message.INFO,
                        0,
                        INFO_PAYLOAD.pack(SECTOR_BYTES, 8192, 0),
                    )
                )
            elif frame.message is Message.READ_NEXT:
                if self.disposition is None:
                    self.queue(
                        Frame(
                            Message.BEGIN,
                            self.token,
                            struct.pack("<II", self.log_id, 0),
                        )
                    )
                    self.queue_sector_or_end()
                else:
                    self.queue(Frame(Message.EMPTY, 0))
            elif frame.message is Message.SECTOR_ACK:
                ordinal = struct.unpack("<I", frame.payload)[0]
                self.assert_ordinal(ordinal)
                self.next_ordinal += 1
                self.queue_sector_or_end()
            elif frame.message is Message.DISPOSITION:
                values = DISPOSITION_PAYLOAD.unpack(frame.payload)
                self.disposition_values = values
                self.disposition = Disposition(values[0])
                if self.disposition is Disposition.ERASE:
                    self.queue(Frame(Message.ERASE_COMPLETE, frame.token))
                    self.erase_complete_count += 1
            elif frame.message is Message.SESSION_DONE:
                self.session_done = True
        return len(data)

    def assert_ordinal(self, ordinal: int) -> None:
        if ordinal != self.next_ordinal:
            raise AssertionError(
                f"expected ACK {self.next_ordinal}, received {ordinal}"
            )

    def read(self, size: int) -> bytes:
        result = bytes(self.output[:size])
        del self.output[:size]
        return result


class ReceiverIntegrationTests(unittest.TestCase):
    def test_receive_persist_classify_and_erase(self) -> None:
        raw = make_complete_log(12, [b"sensor records"])
        fake = FakeDeviceSerial(raw)

        with tempfile.TemporaryDirectory() as temporary_directory:
            received = receive_all(
                SerialProtocol(fake), Path(temporary_directory), erase=True
            )
            files = sorted(Path(temporary_directory).iterdir())

            self.assertEqual(len(received), 1)
            self.assertEqual(received[0][1], "valid")
            self.assertEqual(len(files), 2)
            self.assertEqual(
                next(path for path in files if path.name.endswith(".flash.bin")).read_bytes(),
                raw,
            )
            self.assertEqual(
                next(
                    path
                    for path in files
                    if path.name.endswith(".bin")
                    and not path.name.endswith(".flash.bin")
                ).read_bytes(),
                b"sensor records",
            )
        self.assertEqual(fake.disposition, Disposition.ERASE)
        self.assertEqual(
            fake.disposition_values,
            (
                int(Disposition.ERASE),
                12,
                0,
                2,
                2,
                zlib.crc32(raw),
            ),
        )
        self.assertTrue(fake.session_done)
        self.assertEqual(fake.erase_complete_count, 1)


if __name__ == "__main__":
    unittest.main()
