#!/usr/bin/env python3
import argparse
import csv
from dataclasses import dataclass
import hashlib
import os
import struct
from pathlib import Path
from typing import Dict, Iterator, Tuple

LEGACY_STRUCT = struct.Struct("<II" + "hhh" + "hhh" + "hhh" + "H" + "i")
IMU_GYRO_STRUCT = struct.Struct("<II" + "hhh" + "hhh" + "hhh" + "hhh" + "hhh" + "H" + "i")
DUAL_MAG_STRUCT = struct.Struct("<II" + "hhh" + "hhh" + "hhh" + "hhh" + "hhh" + "hhh" + "H" + "i")

FORMATS: Dict[str, Dict[str, object]] = {
    "legacy": {
        "size": LEGACY_STRUCT.size,
        "struct": LEGACY_STRUCT,
        "header": [
            "t_ms",
            "seq",
            "lis1_x", "lis1_y", "lis1_z",
            "lis2_x", "lis2_y", "lis2_z",
            "mmc_mG_x", "mmc_mG_y", "mmc_mG_z",
            "angle_raw",
            "temp_deciC",
            "temp_C",
        ],
    },
    "imu_gyro": {
        "size": IMU_GYRO_STRUCT.size,
        "struct": IMU_GYRO_STRUCT,
        "header": [
            "t_ms",
            "seq",
            "lis1_x", "lis1_y", "lis1_z",
            "lis2_x", "lis2_y", "lis2_z",
            "gyro1_dps10_x", "gyro1_dps10_y", "gyro1_dps10_z",
            "gyro2_dps10_x", "gyro2_dps10_y", "gyro2_dps10_z",
            "mmc_mG_x", "mmc_mG_y", "mmc_mG_z",
            "angle_raw",
            "temp_deciC",
            "temp_C",
        ],
    },
    "dual_mag": {
        "size": DUAL_MAG_STRUCT.size,
        "struct": DUAL_MAG_STRUCT,
        "header": [
            "t_ms",
            "seq",
            "lis1_x", "lis1_y", "lis1_z",
            "lis2_x", "lis2_y", "lis2_z",
            "gyro1_dps10_x", "gyro1_dps10_y", "gyro1_dps10_z",
            "gyro2_dps10_x", "gyro2_dps10_y", "gyro2_dps10_z",
            "mmc_mG_x", "mmc_mG_y", "mmc_mG_z",
            "lis3mdl_mG_x", "lis3mdl_mG_y", "lis3mdl_mG_z",
            "angle_raw",
            "temp_deciC",
            "temp_C",
        ],
    },
}


@dataclass(frozen=True)
class ConversionResult:
    record_format: str
    records: int
    duration_s: float
    sequence_gaps: int
    source_sha256: str
    output_sha256: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detect_formats(path: str | Path) -> list[str]:
    size = os.path.getsize(path)
    return [fmt for fmt in ("dual_mag", "imu_gyro", "legacy") if size % FORMATS[fmt]["size"] == 0]


def detect_format(path: str | Path) -> str:
    matches = detect_formats(path)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous record format for {path}: file size matches {', '.join(matches)}; "
            "select a hardware profile or pass --format"
        )
    size = os.path.getsize(path)
    raise ValueError(
        f"Could not determine record format for {path}: size {size} is not a multiple "
        f"of {LEGACY_STRUCT.size}, {IMU_GYRO_STRUCT.size}, or {DUAL_MAG_STRUCT.size} bytes"
    )


def iter_records(path: str | Path, fmt: str) -> Iterator[Tuple]:
    record_size = FORMATS[fmt]["size"]
    struct_def = FORMATS[fmt]["struct"]
    with open(path, "rb") as f:
        idx = 0
        while True:
            chunk = f.read(record_size)
            if not chunk:
                return
            if len(chunk) != record_size:
                raise ValueError(
                    f"File ended with a partial record: got {len(chunk)} bytes at record {idx}"
                )
            idx += 1
            yield struct_def.unpack(chunk)


def convert(
    bin_path: str | Path,
    csv_path: str | Path,
    add_seconds: bool = True,
    fmt: str | None = None,
) -> ConversionResult:
    fmt = fmt or detect_format(bin_path)
    if fmt not in FORMATS:
        raise ValueError(f"Unknown record format {fmt!r}")
    size = os.path.getsize(bin_path)
    record_size = int(FORMATS[fmt]["size"])
    if size % record_size != 0:
        raise ValueError(f"{bin_path} size {size} is not a multiple of {fmt} record size {record_size}")

    # Optionally include t_s column computed from t_ms
    out_header = list(FORMATS[fmt]["header"])
    if add_seconds:
        out_header.insert(1, "t_s")

    record_count = 0
    first_t_ms: int | None = None
    last_t_ms: int | None = None
    previous_seq: int | None = None
    sequence_gaps = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as out_f:
        w = csv.writer(out_f)
        w.writerow(out_header)

        for rec in iter_records(bin_path, fmt):
            t_ms = int(rec[0])
            seq = int(rec[1])
            if first_t_ms is None:
                first_t_ms = t_ms
            if previous_seq is not None and seq != ((previous_seq + 1) & 0xFFFFFFFF):
                sequence_gaps += 1
            previous_seq = seq
            last_t_ms = t_ms
            record_count += 1
            if fmt == "legacy":
                (
                    t_ms, seq,
                    lis1_x, lis1_y, lis1_z,
                    lis2_x, lis2_y, lis2_z,
                    mmc_x, mmc_y, mmc_z,
                    angle_raw,
                    temp_deciC,
                ) = rec

                row = [
                    t_ms,
                    seq,
                    lis1_x, lis1_y, lis1_z,
                    lis2_x, lis2_y, lis2_z,
                    mmc_x, mmc_y, mmc_z,
                    angle_raw,
                    temp_deciC,
                    f"{temp_deciC / 10.0:.1f}",
                ]
            elif fmt == "imu_gyro":
                (
                    t_ms, seq,
                    lis1_x, lis1_y, lis1_z,
                    lis2_x, lis2_y, lis2_z,
                    gyro1_x, gyro1_y, gyro1_z,
                    gyro2_x, gyro2_y, gyro2_z,
                    mmc_x, mmc_y, mmc_z,
                    angle_raw,
                    temp_deciC,
                ) = rec

                row = [
                    t_ms,
                    seq,
                    lis1_x, lis1_y, lis1_z,
                    lis2_x, lis2_y, lis2_z,
                    gyro1_x, gyro1_y, gyro1_z,
                    gyro2_x, gyro2_y, gyro2_z,
                    mmc_x, mmc_y, mmc_z,
                    angle_raw,
                    temp_deciC,
                    f"{temp_deciC / 10.0:.1f}",
                ]
            else:
                (
                    t_ms, seq,
                    lis1_x, lis1_y, lis1_z,
                    lis2_x, lis2_y, lis2_z,
                    gyro1_x, gyro1_y, gyro1_z,
                    gyro2_x, gyro2_y, gyro2_z,
                    mmc_x, mmc_y, mmc_z,
                    lis3mdl_x, lis3mdl_y, lis3mdl_z,
                    angle_raw,
                    temp_deciC,
                ) = rec

                row = [
                    t_ms,
                    seq,
                    lis1_x, lis1_y, lis1_z,
                    lis2_x, lis2_y, lis2_z,
                    gyro1_x, gyro1_y, gyro1_z,
                    gyro2_x, gyro2_y, gyro2_z,
                    mmc_x, mmc_y, mmc_z,
                    lis3mdl_x, lis3mdl_y, lis3mdl_z,
                    angle_raw,
                    temp_deciC,
                    f"{temp_deciC / 10.0:.1f}",
                ]

            if add_seconds:
                # insert after t_ms
                row.insert(1, f"{t_ms / 1000.0:.3f}")

            w.writerow(row)

    duration_s = 0.0 if first_t_ms is None or last_t_ms is None else (last_t_ms - first_t_ms) / 1000.0
    return ConversionResult(
        record_format=fmt,
        records=record_count,
        duration_s=duration_s,
        sequence_gaps=sequence_gaps,
        source_sha256=sha256_file(bin_path),
        output_sha256=sha256_file(csv_path),
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert a binary LogRecord file to CSV. Use tools/logs.py ingest for normal imports."
    )
    p.add_argument("input", help="Input .bin file (logNNN.bin)")
    p.add_argument("-o", "--output", help="Output .csv path (default: input name with .csv)")
    p.add_argument("--no-seconds", action="store_true", help="Do not add computed t_s column")
    p.add_argument(
        "--format",
        choices=sorted(FORMATS.keys()),
        help="Override record format detection",
    )
    args = p.parse_args()

    bin_path = args.input
    if args.output:
        csv_path = args.output
    else:
        base, _ = os.path.splitext(bin_path)
        csv_path = base + ".csv"

    result = convert(bin_path, csv_path, add_seconds=not args.no_seconds, fmt=args.format)
    print(
        f"Wrote {csv_path}: format={result.record_format}, records={result.records}, "
        f"duration={result.duration_s:.1f}s, sequence_gaps={result.sequence_gaps}"
    )


if __name__ == "__main__":
    main()
