from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from read_binary import DUAL_MAG_STRUCT, convert, detect_format


class ReadBinaryTests(unittest.TestCase):
    def test_convert_returns_validation_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source = tmp_path / "log.bin"
            output = tmp_path / "log.csv"
            records = [
                DUAL_MAG_STRUCT.pack(1000, 10, *([0] * 18), 2048, 250),
                DUAL_MAG_STRUCT.pack(1010, 11, *([0] * 18), 2048, 251),
                DUAL_MAG_STRUCT.pack(1020, 13, *([0] * 18), 2048, 252),
            ]
            source.write_bytes(b"".join(records))

            result = convert(source, output, fmt="dual_mag")

            self.assertEqual(result.record_format, "dual_mag")
            self.assertEqual(result.records, 3)
            self.assertEqual(result.duration_s, 0.02)
            self.assertEqual(result.sequence_gaps, 1)
            self.assertTrue(result.source_sha256)
            self.assertTrue(result.output_sha256)
            self.assertTrue(output.read_text(encoding="utf-8").splitlines()[0].startswith("t_ms,t_s,seq"))

    def test_ambiguous_format_requires_explicit_choice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "ambiguous.bin"
            source.write_bytes(bytes(800))  # Divisible by both 32-byte legacy and 50-byte dual-mag records.

            with self.assertRaisesRegex(ValueError, "Ambiguous record format"):
                detect_format(source)


if __name__ == "__main__":
    unittest.main()
