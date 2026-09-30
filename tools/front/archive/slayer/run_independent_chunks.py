#!/usr/bin/env python3
"""Run each grown Slayer range through the complete front pipeline independently."""

from __future__ import annotations

import contextlib
import csv
from dataclasses import replace
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sus-mpl-cache")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BACKEND = ROOT / "backend"
OUT = ROOT / "reports" / "slayer_chunk_pipeline_independent_20pct"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import pipeline
from log_registry import resolve_log


BASE_SCRIPT = Path(__file__).with_name("run_chunk_pipeline.py")
module_spec = importlib.util.spec_from_file_location("slayer_chunk_metrics", BASE_SCRIPT)
if module_spec is None or module_spec.loader is None:
    raise RuntimeError(f"Could not import {BASE_SCRIPT}")
metrics = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = metrics
module_spec.loader.exec_module(metrics)


RANGES = {
    "log-0145": [(0, 39029)],
    "log-0147": [(674, 21512), (23132, 61371)],
    "log-0151": [(6573, 16564)],
    "log-0152": [(0, 16176), (31185, 41401)],
    "log-0155": [(0, 14580)],
}
CORE_SENSOR_COLUMNS = (
    ("lis1_x", "lis1_y", "lis1_z"),
    ("lis2_x", "lis2_y", "lis2_z"),
    ("gyro1_dps10_x", "gyro1_dps10_y", "gyro1_dps10_z"),
    ("gyro2_dps10_x", "gyro2_dps10_y", "gyro2_dps10_z"),
    ("mmc_mG_x", "mmc_mG_y", "mmc_mG_z"),
)


def flatten(value: Any, dtype: Any = float) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1)


def series(cache: np.lib.npyio.NpzFile, key: str) -> tuple[np.ndarray, np.ndarray]:
    return flatten(cache[f"{key}__t"]), np.asarray(cache[f"{key}__x"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_independent_log(
    parent_id: str,
    chunk_id: str,
    raw_chunk: pd.DataFrame,
    dropout: np.ndarray,
    work_root: Path,
) -> tuple[dict[str, Any], dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]]:
    chunk_csv = work_root / "input" / f"{chunk_id}.csv"
    chunk_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_chunk.to_csv(chunk_csv, index=False)

    parent = resolve_log(parent_id)
    independent = replace(
        parent,
        log_id=chunk_id,
        csv_path=chunk_csv,
        source_path=None,
        sets=("slayer-independent-chunks",),
    )
    original_resolve = pipeline.resolve_log
    original_parse = pipeline.parse_args
    original_cwd = Path.cwd()
    try:
        pipeline.resolve_log = lambda _: independent
        pipeline.parse_args = lambda: SimpleNamespace(log_filename=chunk_id)
        os.chdir(work_root)
        pipeline.main()
    finally:
        os.chdir(original_cwd)
        pipeline.resolve_log = original_resolve
        pipeline.parse_args = original_parse

    cache_path = work_root / "backend" / "run_artifacts" / chunk_id / "cache" / "all.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        time_s, truth_values = series(cache, "travel")
        truth = flatten(truth_values)
        active = flatten(cache["active_mask"], bool)
        accel_prediction = flatten(series(cache, "accel/lpf/proj")[1])
        mag_prediction = flatten(series(cache, "travel/mag_model/adj")[1])
        fusion1 = flatten(series(cache, "travel/fusion1")[1])
        fusion2 = flatten(series(cache, "travel/solved")[1])
        baseline = flatten(cache["mag_baseline"])
        reference = flatten(cache["mag_travel_ref_point"])
        model_offset = flatten(cache["mag_model_offset_mm"])
        zv_points = flatten(cache["mag_zv_points"], int)
        scatter = np.asarray(cache["fusion_scatter_points"])

    if len(dropout) != len(truth):
        raise ValueError(f"Dropout length mismatch: {len(dropout)} vs {len(truth)}")
    velocity = np.gradient(truth, time_s, edge_order=2)
    accel_truth = np.gradient(velocity, time_s, edge_order=2) / 1000.0
    evaluation = active & (truth > 0) & np.isfinite(truth)
    masks = {"unmasked": evaluation, "masked": evaluation & ~dropout}

    sample_dt = float(np.median(np.diff(time_s)))
    row: dict[str, Any] = {
        "chunk": chunk_id,
        "log": parent_id,
        "start_s": float(time_s[0]),
        "stop_s": float(time_s[-1] + sample_dt),
        "wall_s": float(len(time_s) * sample_dt),
        "active_s": float(np.count_nonzero(active) * sample_dt),
        "dropout_pct": float(np.mean(dropout) * 100.0),
        "zv_points": len(zv_points),
        "training_points": int(scatter.shape[0]),
        "mag_baseline": float(baseline[0]),
        "mag_reference": float(reference[0]),
        "mag_model_offset_mm": float(model_offset[0]),
    }
    payload: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    predictions = {"mag": mag_prediction, "fusion1": fusion1, "fusion2": fusion2}
    for mask_name, mask in masks.items():
        accel_mask = np.abs(accel_truth) > 0.5
        if mask_name == "masked":
            accel_mask &= ~dropout
        accel_stats = metrics.metric_stats(accel_prediction, accel_truth, accel_mask)
        metrics.prefix_stats(row, f"accel_{mask_name}", accel_stats)
        finite_accel = accel_mask & np.isfinite(accel_prediction) & np.isfinite(accel_truth)
        payload[("accel", mask_name)] = (
            accel_prediction[finite_accel] - accel_truth[finite_accel],
            np.full(np.count_nonzero(finite_accel), chunk_id, dtype=object),
        )
        for stage, prediction in predictions.items():
            stage_stats = metrics.metric_stats(prediction, truth, mask)
            metrics.prefix_stats(row, f"{stage}_{mask_name}", stage_stats)
            finite = mask & np.isfinite(prediction) & np.isfinite(truth)
            payload[(stage, mask_name)] = (
                prediction[finite] - truth[finite],
                np.full(np.count_nonzero(finite), chunk_id, dtype=object),
            )
    return row, payload


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    logs_dir = OUT / "logs"
    logs_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    residuals = {
        (stage, mask): []
        for stage in ("accel", "mag", "fusion1", "fusion2")
        for mask in ("unmasked", "masked")
    }
    total = sum(len(value) for value in RANGES.values())
    with tempfile.TemporaryDirectory(prefix="slayer-independent-") as temporary:
        work_root = Path(temporary)
        run_number = 0
        for parent_id, ranges in RANGES.items():
            raw = pd.read_csv(ROOT / "logs" / "converted" / f"{parent_id}.csv")
            raw_bad = np.logical_or.reduce(
                [np.all(raw[list(columns)].to_numpy(float) == 0, axis=1) for columns in CORE_SENSOR_COLUMNS]
            )
            for ordinal, (start, stop) in enumerate(ranges, start=1):
                run_number += 1
                chunk_id = f"{parent_id}-independent-c{ordinal:02d}"
                print(f"[{run_number}/{total}] {chunk_id}", flush=True)
                raw_start, raw_stop = 2 * start, 2 * stop
                raw_chunk = raw.iloc[raw_start:raw_stop].copy()
                dropout = raw_bad[raw_start:raw_stop:2]
                capture = io.StringIO()
                try:
                    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                        row, payload = run_independent_log(
                            parent_id, chunk_id, raw_chunk, dropout, work_root
                        )
                    rows.append(row)
                    for key, (errors, _) in payload.items():
                        residuals[key].append(errors)
                except Exception as error:
                    failures.append(
                        {"chunk": chunk_id, "error": f"{type(error).__name__}: {error}"}
                    )
                    capture.write(f"\n{type(error).__name__}: {error}\n")
                (logs_dir / f"{chunk_id}.txt").write_text(capture.getvalue(), encoding="utf-8")

    write_csv(OUT / "per_chunk_metrics.csv", rows)
    write_csv(OUT / "failures.csv", failures)
    aggregates = metrics.aggregate_rows(rows, residuals) if rows else []
    write_csv(OUT / "aggregate_metrics.csv", aggregates)
    print(f"Completed {len(rows)} chunks; {len(failures)} failures")


if __name__ == "__main__":
    main()
