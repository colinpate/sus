#!/usr/bin/env python3
"""Run the chunk pipeline on the seven maximally grown Slayer ranges."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
BASE_SCRIPT = Path(__file__).with_name("run_chunk_pipeline.py")
OUT = ROOT / "reports" / "slayer_chunk_pipeline_grown_20pct"

module_spec = importlib.util.spec_from_file_location("slayer_chunk_base", BASE_SCRIPT)
if module_spec is None or module_spec.loader is None:
    raise RuntimeError(f"Could not import {BASE_SCRIPT}")
base = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = base
module_spec.loader.exec_module(base)


RANGES_BY_LENGTH = {
    39029: [(0, 39029)],
    61371: [(674, 21512), (23132, 61371)],
    16564: [(6573, 16564)],
    41401: [(0, 16176), (31185, 41401)],
    14580: [(0, 14580)],
}


def fixed_grown_chunks(active: np.ndarray, dropout: np.ndarray, time_s: np.ndarray):
    ranges = RANGES_BY_LENGTH[len(active)]
    sample_dt = float(np.median(np.diff(time_s)))
    chunks = []
    for ordinal, (start, stop) in enumerate(ranges, start=1):
        chunks.append(
            base.ChunkSpec(
                chunk_id=f"{ordinal:02d}",
                log_id="",
                start=start,
                stop=stop,
                start_s=float(time_s[start]),
                stop_s=float(time_s[stop - 1] + sample_dt),
                wall_s=float((stop - start) * sample_dt),
                active_s=float(np.count_nonzero(active[start:stop]) * sample_dt),
                dropout_fraction=float(np.mean(dropout[start:stop])),
            )
        )
    return chunks


if __name__ == "__main__":
    base.OUT = OUT
    base.find_maximum_chunks = fixed_grown_chunks
    base.main()
    report_path = OUT / "report.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace(
        "# Slayer 60-active-second chunk pipeline experiment",
        "# Slayer maximally grown 20%-dropout chunk pipeline experiment",
    ).replace(
        "Each chunk contains at least 60 seconds of activity and less than 20% raw core-sensor zero-output dropout.",
        "These are the seven non-overlapping ranges obtained by joining and maximally growing the original 15 seed chunks while preserving at least 60 seconds of activity and less than 20% raw core-sensor zero-output dropout per range.",
    )
    report_path.write_text(report, encoding="utf-8")
