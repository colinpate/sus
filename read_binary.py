#!/usr/bin/env python3
import argparse
import csv
import os
import struct
from typing import Iterator, Tuple

# LogRecord packed size and format (little-endian)
# C struct (packed):
# uint32_t t_ms;
# uint32_t seq;
# int16_t  lis1[3];
# int16_t  lis2[3];
# int16_t  mmc_mG[3];
# uint16_t angle;
# int32_t  temp_C;   // stored as deci-degC (x10)
#
# Total: 4+4 + 6+6+6 + 2 + 4 = 32 bytes
RECORD_SIZE = 32
STRUCT_FMT = "<II" + "hhh" + "hhh" + "hhh" + "H" + "i"
STRUCT = struct.Struct(STRUCT_FMT)

Header = [
    "t_ms",
    "seq",
    "lis1_x", "lis1_y", "lis1_z",
    "lis2_x", "lis2_y", "lis2_z",
    "mmc_mG_x", "mmc_mG_y", "mmc_mG_z",
    "angle_raw",
    "temp_deciC",
    "temp_C",
]

def iter_records(path: str) -> Iterator[Tuple]:
    with open(path, "rb") as f:
        idx = 0
        while True:
            chunk = f.read(RECORD_SIZE)
            if not chunk:
                return
            if len(chunk) != RECORD_SIZE:
                raise ValueError(
                    f"File ended with a partial record: got {len(chunk)} bytes at record {idx}"
                )
            idx += 1
            yield STRUCT.unpack(chunk)

def convert(bin_path: str, csv_path: str, add_seconds: bool = True) -> None:
    # Optionally include t_s column computed from t_ms
    out_header = Header.copy()
    if add_seconds:
        out_header.insert(1, "t_s")

    with open(csv_path, "w", newline="") as out_f:
        w = csv.writer(out_f)
        w.writerow(out_header)

        for rec in iter_records(bin_path):
            (
                t_ms, seq,
                lis1_x, lis1_y, lis1_z,
                lis2_x, lis2_y, lis2_z,
                mmc_x, mmc_y, mmc_z,
                angle_raw,
                temp_deciC,
            ) = rec

            temp_C = temp_deciC / 10.0

            row = [
                t_ms,
                seq,
                lis1_x, lis1_y, lis1_z,
                lis2_x, lis2_y, lis2_z,
                mmc_x, mmc_y, mmc_z,
                angle_raw,
                temp_deciC,
                f"{temp_C:.1f}",
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
    args = p.parse_args()

    bin_path = args.input
    if args.output:
        csv_path = args.output
    else:
        base, _ = os.path.splitext(bin_path)
        csv_path = base + ".csv"

    convert(bin_path, csv_path, add_seconds=not args.no_seconds)
    print(f"Wrote: {csv_path}")

if __name__ == "__main__":
    main()
