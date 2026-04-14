#!/usr/bin/env python3
import argparse
import csv
import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

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


def get_metadata_path(log_path: str) -> Path:
    return Path(log_path).with_suffix(".meta.json")


def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not isinstance(metadata, dict):
        raise ValueError(f"Expected top-level object in {metadata_path}")

    return metadata


def write_metadata(
    log_path: str,
    hypotenuse: float | None = None,
    top_adjacent: float | None = None,
) -> Path | None:
    if hypotenuse is None and top_adjacent is None:
        return None

    metadata_path = get_metadata_path(log_path)
    metadata = load_metadata(metadata_path)

    steps = metadata.get("steps")
    if not isinstance(steps, dict):
        steps = {}
        metadata["steps"] = steps

    angle_to_travel = steps.get("angle_to_travel")
    if not isinstance(angle_to_travel, dict):
        angle_to_travel = {}
        steps["angle_to_travel"] = angle_to_travel

    if hypotenuse is not None:
        angle_to_travel["hypotenuse"] = hypotenuse
    if top_adjacent is not None:
        angle_to_travel["top_adjacent"] = top_adjacent

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    return metadata_path

def detect_format(path: str) -> str:
    size = os.path.getsize(path)
    for fmt in ("dual_mag", "imu_gyro", "legacy"):
        if size % FORMATS[fmt]["size"] == 0:
            return fmt
    raise ValueError(
        f"Could not determine record format for {path}: size {size} is not a multiple "
        f"of {LEGACY_STRUCT.size}, {IMU_GYRO_STRUCT.size}, or {DUAL_MAG_STRUCT.size} bytes"
    )

def iter_records(path: str, fmt: str) -> Iterator[Tuple]:
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

def convert(bin_path: str, csv_path: str, add_seconds: bool = True, fmt: str | None = None) -> None:
    fmt = fmt or detect_format(bin_path)

    # Optionally include t_s column computed from t_ms
    out_header = list(FORMATS[fmt]["header"])
    if add_seconds:
        out_header.insert(1, "t_s")

    with open(csv_path, "w", newline="") as out_f:
        w = csv.writer(out_f)
        w.writerow(out_header)

        for rec in iter_records(bin_path, fmt):
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

def main() -> None:
    p = argparse.ArgumentParser(description="Convert ESP32 binary LogRecord file to CSV")
    p.add_argument("input", help="Input .bin file (logNNN.bin)")
    p.add_argument("-o", "--output", help="Output .csv path (default: input name with .csv)")
    p.add_argument("--no-seconds", action="store_true", help="Do not add computed t_s column")
    p.add_argument(
        "--hypotenuse",
        type=float,
        help="Write steps.angle_to_travel.hypotenuse to the input log's .meta.json file",
    )
    p.add_argument(
        "--top-adjacent",
        type=float,
        help="Write steps.angle_to_travel.top_adjacent to the input log's .meta.json file",
    )
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

    convert(bin_path, csv_path, add_seconds=not args.no_seconds, fmt=args.format)
    print(f"Wrote: {csv_path}")

    metadata_path = write_metadata(
        bin_path,
        hypotenuse=args.hypotenuse,
        top_adjacent=args.top_adjacent,
    )
    if metadata_path is not None:
        print(f"Wrote: {metadata_path}")

if __name__ == "__main__":
    main()
