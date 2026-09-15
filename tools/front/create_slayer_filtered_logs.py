#!/usr/bin/env python3
"""Create the seven contiguous Slayer paper logs and their provenance manifest."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONVERTED = ROOT / "logs" / "converted"
MANIFEST_PATH = ROOT / "logs" / "slayer-filtered-manifest.json"
SELECTION_RULE = (
    "Join the maximum-count 60-active-second seed windows within each parent log, "
    "then maximize total retained non-overlapping wall time subject to each output "
    "remaining contiguous, containing its seeds, and having <20% raw core-sensor "
    "zero-output dropout."
)
CORE_SENSOR_COLUMNS = (
    ("lis1_x", "lis1_y", "lis1_z"),
    ("lis2_x", "lis2_y", "lis2_z"),
    ("gyro1_dps10_x", "gyro1_dps10_y", "gyro1_dps10_z"),
    ("gyro2_dps10_x", "gyro2_dps10_y", "gyro2_dps10_z"),
    ("mmc_mG_x", "mmc_mG_y", "mmc_mG_z"),
)

# Bounds are half-open indices on the 100 Hz pipeline timeline. The converted
# source CSVs are 200 Hz, so each output copies rows [2*start, 2*stop).
CHUNKS = (
    ("log-0145-filtered-c01", "log-0145", 0, 39029, 348.37),
    ("log-0147-filtered-c01", "log-0147", 674, 21512, 81.56),
    ("log-0147-filtered-c02", "log-0147", 23132, 61371, 319.60),
    ("log-0151-filtered-c01", "log-0151", 6573, 16564, 86.59),
    ("log-0152-filtered-c01", "log-0152", 0, 16176, 149.14),
    ("log-0152-filtered-c02", "log-0152", 31185, 41401, 92.55),
    ("log-0155-filtered-c01", "log-0155", 0, 14580, 133.28),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_rows(source: Path, destination: Path, raw_start: int, raw_stop: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8", newline="") as src, destination.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        header = src.readline()
        if not header:
            raise ValueError(f"Empty source CSV: {source}")
        dst.write(header)
        for row_index, line in enumerate(src):
            if row_index >= raw_stop:
                break
            if row_index >= raw_start:
                dst.write(line)


def inspect_output(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No data rows: {path}")
    bad = [
        any(all(float(row[name]) == 0.0 for name in group) for group in CORE_SENSOR_COLUMNS)
        for row in rows
    ]
    return {
        "records": len(rows),
        "start_s": float(rows[0]["t_s"]),
        "stop_s": float(rows[-1]["t_s"]) + 0.005,
        "duration_s": len(rows) / 200.0,
        "raw_dropout_fraction_200hz": sum(bad) / len(rows),
        "selection_dropout_fraction_100hz": sum(bad[::2]) / len(bad[::2]),
        "first_sequence": int(rows[0]["seq"]),
        "last_sequence": int(rows[-1]["seq"]),
    }


def main() -> None:
    manifest_rows = []
    source_hashes: dict[str, str] = {}
    for output_id, parent_id, start, stop, active_s in CHUNKS:
        source = CONVERTED / f"{parent_id}.csv"
        destination = CONVERTED / f"{output_id}.csv"
        raw_start, raw_stop = 2 * start, 2 * stop
        copy_rows(source, destination, raw_start, raw_stop)
        observed = inspect_output(destination)
        with source.open("r", encoding="utf-8", newline="") as handle:
            source_records = sum(1 for _ in handle) - 1
        expected_records = max(0, min(raw_stop, source_records) - raw_start)
        if observed["records"] != expected_records:
            raise ValueError(
                f"{output_id}: expected {expected_records} records, got {observed['records']}"
            )
        source_hashes.setdefault(parent_id, sha256(source))
        manifest_rows.append(
            {
                "log_id": output_id,
                "parent_log": parent_id,
                "source_csv": str(source.relative_to(ROOT)),
                "output_csv": str(destination.relative_to(ROOT)),
                "source_sha256": source_hashes[parent_id],
                "output_sha256": sha256(destination),
                "source_start_index_100hz": start,
                "source_stop_index_100hz": stop,
                "source_start_row_200hz": raw_start,
                "source_stop_row_200hz": raw_stop,
                "active_s_at_selection": active_s,
                **observed,
            }
        )

    manifest = {
        "schema_version": 1,
        "set": "slayer-filtered",
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "selection_rule": SELECTION_RULE,
        "dropout_definition": (
            "A raw 200 Hz record is bad when any core sensor triad (LIS1, LIS2, "
            "gyro1, gyro2, or primary MMC magnetometer) is exactly [0, 0, 0]."
        ),
        "outputs": manifest_rows,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest_rows)} logs and {MANIFEST_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
