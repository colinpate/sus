#!/usr/bin/env python3
"""Run the downstream front pipeline independently on selected Slayer windows.

Sensor preprocessing and mounting calibration come from each parent log's full
pipeline cache. The magnetic model and both travel solvers are fit/run anew on
each chunk. Dropout masks are used only for evaluation and chunk selection.
"""

from __future__ import annotations

import contextlib
from copy import deepcopy
import csv
import io
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sus-mpl-cache")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BACKEND = ROOT / "backend"
OUT = ROOT / "reports" / "slayer_chunk_pipeline_60s_20pct"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from classes.log_config import attach_log_config  # noqa: E402
from classes.time_series import TimeSeries  # noqa: E402
from fusion import GetMagToTravelModel  # noqa: E402
from log_registry import resolve_log  # noqa: E402
from mag_nuisance import (  # noqa: E402
    MagNuisanceFullRateCorrection,
    MagNuisanceTravelCorrection,
)
from travel_solver import TravelSolver  # noqa: E402


ACTIVE_SECONDS = 60.0
MAX_DROPOUT_FRACTION = 0.20
SLAYER_LOGS = ["log-0145", "log-0147", "log-0151", "log-0152", "log-0155"]
CORE_SENSOR_COLUMNS = (
    ("lis1_x", "lis1_y", "lis1_z"),
    ("lis2_x", "lis2_y", "lis2_z"),
    ("gyro1_dps10_x", "gyro1_dps10_y", "gyro1_dps10_z"),
    ("gyro2_dps10_x", "gyro2_dps10_y", "gyro2_dps10_z"),
    ("mmc_mG_x", "mmc_mG_y", "mmc_mG_z"),
)


@dataclass(frozen=True)
class ChunkSpec:
    chunk_id: str
    log_id: str
    start: int
    stop: int
    start_s: float
    stop_s: float
    wall_s: float
    active_s: float
    dropout_fraction: float


def flatten(value: np.ndarray, dtype: Any = float) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1)


def cache_series(cache: np.lib.npyio.NpzFile, key: str) -> tuple[np.ndarray, np.ndarray]:
    return flatten(cache[f"{key}__t"]), np.asarray(cache[f"{key}__x"])


def core_dropout_mask(log_id: str) -> np.ndarray:
    path = ROOT / "logs" / "converted" / f"{log_id}.csv"
    frame = pd.read_csv(path, usecols=[column for group in CORE_SENSOR_COLUMNS for column in group])
    raw_bad = np.logical_or.reduce(
        [np.all(frame[list(columns)].to_numpy(float) == 0, axis=1) for columns in CORE_SENSOR_COLUMNS]
    )
    return raw_bad[::2]


def find_maximum_chunks(
    active: np.ndarray,
    dropout: np.ndarray,
    time_s: np.ndarray,
) -> list[ChunkSpec]:
    """Greedily select earliest-finishing valid intervals (optimal interval scheduling)."""
    sample_dt = float(np.median(np.diff(time_s)))
    active_samples_required = int(np.ceil(ACTIVE_SECONDS / sample_dt - 1e-9))
    active_prefix = np.r_[0, np.cumsum(np.asarray(active, dtype=bool).astype(int))]
    dropout_prefix = np.r_[0, np.cumsum(np.asarray(dropout, dtype=bool).astype(int))]
    score = dropout_prefix - MAX_DROPOUT_FRACTION * np.arange(len(active_prefix))

    intervals: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(active):
        add_index = cursor
        best_score = -np.inf
        best_start: int | None = None
        found: tuple[int, int] | None = None
        for stop in range(cursor + 1, len(active) + 1):
            max_start_activity = active_prefix[stop] - active_samples_required
            while add_index < stop and active_prefix[add_index] <= max_start_activity:
                if score[add_index] > best_score:
                    best_score = float(score[add_index])
                    best_start = add_index
                add_index += 1
            if best_start is not None and best_score > score[stop] + 1e-10:
                found = (best_start, stop)
                break
        if found is None:
            break
        intervals.append(found)
        cursor = found[1]

    specs = []
    for ordinal, (start, stop) in enumerate(intervals, start=1):
        specs.append(
            ChunkSpec(
                chunk_id=f"{ordinal:02d}",
                log_id="",
                start=start,
                stop=stop,
                start_s=float(time_s[start]),
                stop_s=float(time_s[stop - 1] + sample_dt),
                wall_s=float((stop - start) * sample_dt),
                active_s=float((active_prefix[stop] - active_prefix[start]) * sample_dt),
                dropout_fraction=float(
                    (dropout_prefix[stop] - dropout_prefix[start]) / (stop - start)
                ),
            )
        )
    return specs


def make_series(
    cache: np.lib.npyio.NpzFile,
    key: str,
    sample_slice: slice,
    *,
    units: str,
    frame: str,
) -> TimeSeries:
    time_s, values = cache_series(cache, key)
    sliced_time = time_s[sample_slice]
    fs_hz = 1.0 / float(np.median(np.diff(sliced_time)))
    return TimeSeries(
        t=sliced_time,
        x=values[sample_slice],
        units=units,
        frame=frame,
        meta={"fs_hz": fs_hz, "source_log": sample_slice},
    )


def metric_stats(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    prediction = flatten(prediction)
    truth = flatten(truth)
    mask = flatten(mask, bool) & np.isfinite(prediction) & np.isfinite(truth)
    if not np.any(mask):
        return {"samples": 0, "rmse": np.nan, "mae": np.nan, "me": np.nan, "centered_rmse": np.nan}
    error = prediction[mask] - truth[mask]
    centered_error = error - np.mean(error)
    return {
        "samples": int(np.sum(mask)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "me": float(np.mean(error)),
        "centered_rmse": float(np.sqrt(np.mean(centered_error**2))),
    }


def prefix_stats(row: dict[str, Any], prefix: str, stats: dict[str, Any]) -> None:
    for key, value in stats.items():
        row[f"{prefix}_{key}"] = value


def run_chunk(
    spec: ChunkSpec,
    cache: np.lib.npyio.NpzFile,
    dropout_full: np.ndarray,
    config: dict[str, Any],
    parent_accel_truth: np.ndarray,
) -> tuple[dict[str, Any], dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]]:
    sample_slice = slice(spec.start, spec.stop)
    ws: dict[str, Any] = {}
    attach_log_config(ws, config)
    ws["accel/lpf/proj"] = make_series(cache, "accel/lpf/proj", sample_slice, units="m/s^2", frame="lis1")
    ws["accel/lpfhp/proj"] = make_series(cache, "accel/lpfhp/proj", sample_slice, units="m/s^2", frame="lis1")
    ws["travel"] = make_series(cache, "travel", sample_slice, units="mm", frame="sensor")
    ws["mag/lpf"] = make_series(cache, "mag/lpf", sample_slice, units="milli-Gauss", frame="mag")
    ws["gyro/lpf/gyro1"] = make_series(cache, "gyro/lpf/gyro1", sample_slice, units="degrees/s", frame="gyro1")
    ws["mag/norm/corr/lpf"] = make_series(cache, "mag/norm/corr/lpf", sample_slice, units="milli-Gauss", frame="travel")
    ws["mag/norm/bad_mask"] = make_series(cache, "mag/norm/bad_mask", sample_slice, units="bool", frame="mag")
    ws["active_mask"] = flatten(cache["active_mask"], bool)[sample_slice]
    ws["boring_mask"] = flatten(cache["boring_mask"], bool)[sample_slice]
    ws["mag_baseline"] = flatten(cache["mag_baseline"])
    ws["mag_travel_ref_point"] = flatten(cache["mag_travel_ref_point"])
    all_zv = flatten(cache["mag_zv_points"], int)
    ws["mag_zv_points"] = all_zv[(all_zv >= spec.start) & (all_zv < spec.stop)] - spec.start

    mag_model = GetMagToTravelModel(
        name="mag_to_travel_model",
        inputs=(
            "mag/norm/corr/lpf",
            "accel/lpfhp/proj",
            "travel",
            "mag/norm/bad_mask",
            "mag_zv_points",
            "mag_travel_ref_point",
            "mag_baseline",
        ),
        outputs=(
            "travel/mag_model",
            "travel/mag_model/adj",
            "fusion_scatter_points",
            "mag_model_coeffs",
            "mag_model_offset_mm",
        ),
        train_with_mask=False,
        ref_neg_fallback_max_pct=1.0,
        ref_max_out_of_range_pct=1.0,
    )
    mag_model.run(ws)

    TravelSolver(
        name="travel_solver",
        inputs=(
            "accel/lpfhp/proj",
            "mag/norm/corr/lpf",
            "travel/mag_model/adj",
            "mag_zv_points",
            "mag_baseline",
        ),
        outputs=("travel/fusion1",),
        verbose=0,
    ).run(ws)

    MagNuisanceTravelCorrection(
        name="mag_nuisance_correction",
        inputs=(
            "mag/lpf",
            "gyro/lpf/gyro1",
            "mag/norm/corr/lpf",
            "mag_model_coeffs",
            "mag_model_offset_mm",
            "travel/fusion1",
        ),
        outputs=(
            "travel/solved/mag_nuisance/10hz",
            "mag/nuisance/body/10hz",
            "mag/nuisance/world/10hz",
            "mag/nuisance/xyz_path",
            "mag/nuisance/summary",
        ),
    ).run(ws)
    MagNuisanceFullRateCorrection(
        name="mag_nuisance_full_rate",
        inputs=(
            "mag/lpf",
            "gyro/lpf/gyro1",
            "travel/fusion1",
            "travel/mag_model/adj",
            "travel/solved/mag_nuisance/10hz",
            "mag/nuisance/body/10hz",
            "mag/nuisance/world/10hz",
            "mag/nuisance/xyz_path",
        ),
        outputs=(
            "travel/solved/mag_nuisance/delta_lifted",
            "travel/mag_nuisance/corrected",
            "mag/nuisance/corrected/norm",
        ),
    ).run(ws)
    TravelSolver(
        name="travel_solver_mag_nuisance",
        inputs=(
            "accel/lpfhp/proj",
            "mag/norm/corr/lpf",
            "travel/mag_nuisance/corrected",
            "mag_zv_points",
            "mag_baseline",
        ),
        outputs=("travel/solved",),
        verbose=0,
    ).run(ws)

    dropout = dropout_full[sample_slice]
    truth = ws["travel"].x[:, 0]
    evaluation = ws["active_mask"] & (truth > 0) & np.isfinite(truth)
    masks = {"unmasked": evaluation, "masked": evaluation & ~dropout}

    accel_prediction = ws["accel/lpf/proj"].x[:, 0]
    accel_truth = parent_accel_truth[sample_slice]
    accel_base_mask = np.abs(accel_truth) > 0.5

    row: dict[str, Any] = {
        "chunk": spec.chunk_id,
        "log": spec.log_id,
        "start_s": spec.start_s,
        "stop_s": spec.stop_s,
        "wall_s": spec.wall_s,
        "active_s": spec.active_s,
        "dropout_pct": spec.dropout_fraction * 100.0,
        "zv_points": len(ws["mag_zv_points"]),
        "training_chunks": len(mag_model.chunks),
        "mag_model_offset_mm": float(flatten(ws["mag_model_offset_mm"])[0]),
    }
    residual_payload: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for mask_name, mask in masks.items():
        accel_mask = accel_base_mask if mask_name == "unmasked" else accel_base_mask & ~dropout
        prefix_stats(
            row,
            f"accel_{mask_name}",
            metric_stats(accel_prediction, accel_truth, accel_mask),
        )
        finite_accel_mask = accel_mask & np.isfinite(accel_prediction) & np.isfinite(accel_truth)
        residual_payload[("accel", mask_name)] = (
            accel_prediction[finite_accel_mask] - accel_truth[finite_accel_mask],
            np.array([spec.chunk_id] * int(np.sum(finite_accel_mask)), dtype=object),
        )
        for stage, key in (
            ("mag", "travel/mag_model/adj"),
            ("fusion1", "travel/fusion1"),
            ("fusion2", "travel/solved"),
        ):
            prediction = ws[key].x[:, 0]
            stats = metric_stats(prediction, truth, mask)
            prefix_stats(row, f"{stage}_{mask_name}", stats)
            finite_mask = mask & np.isfinite(prediction) & np.isfinite(truth)
            residual_payload[(stage, mask_name)] = (
                prediction[finite_mask] - truth[finite_mask],
                np.array([spec.chunk_id] * int(np.sum(finite_mask)), dtype=object),
            )
    return row, residual_payload


def aggregate_rows(
    rows: list[dict[str, Any]],
    residuals: dict[tuple[str, str], list[np.ndarray]],
) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for stage in ("accel", "mag", "fusion1", "fusion2"):
        for mask_name in ("unmasked", "masked"):
            prefix = f"{stage}_{mask_name}"
            chunk_rmse = np.asarray([row[f"{prefix}_rmse"] for row in rows], dtype=float)
            chunk_centered = np.asarray([row[f"{prefix}_centered_rmse"] for row in rows], dtype=float)
            log_rmse = []
            log_centered = []
            for log_id in sorted({str(row["log"]) for row in rows}):
                log_rows = [row for row in rows if row["log"] == log_id]
                log_rmse.append(np.mean([row[f"{prefix}_rmse"] for row in log_rows]))
                log_centered.append(
                    np.mean([row[f"{prefix}_centered_rmse"] for row in log_rows])
                )
            errors = np.concatenate(residuals[(stage, mask_name)])
            centered_errors = []
            for error in residuals[(stage, mask_name)]:
                centered_errors.append(error - np.mean(error))
            centered_errors_array = np.concatenate(centered_errors)
            aggregates.append(
                {
                    "stage": stage,
                    "units": "m/s^2" if stage == "accel" else "mm",
                    "evaluation": mask_name,
                    "chunks": len(rows),
                    "samples": len(errors),
                    "macro_mean_rmse": float(np.mean(chunk_rmse)),
                    "macro_mean_centered_rmse": float(np.mean(chunk_centered)),
                    "log_balanced_mean_rmse": float(np.mean(log_rmse)),
                    "log_balanced_mean_centered_rmse": float(np.mean(log_centered)),
                    "pooled_rmse": float(np.sqrt(np.mean(errors**2))),
                    "pooled_mae": float(np.mean(np.abs(errors))),
                    "pooled_me": float(np.mean(errors)),
                    "pooled_chunk_centered_rmse": float(
                        np.sqrt(np.mean(centered_errors_array**2))
                    ),
                }
            )
    return aggregates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 2) -> str:
    return "—" if not np.isfinite(float(value)) else f"{float(value):.{digits}f}"


def write_report(rows: list[dict[str, Any]], aggregates: list[dict[str, Any]], failures: list[dict[str, str]]) -> None:
    aggregate_by_key = {
        (str(row["stage"]), str(row["evaluation"])): row for row in aggregates
    }
    lines = [
        "# Slayer 60-active-second chunk pipeline experiment",
        "",
        "Each chunk contains at least 60 seconds of activity and less than 20% raw core-sensor zero-output dropout. "
        "Full-log preprocessing, mounting calibration, magnetic baseline, and absolute reference are reused. "
        "The magnetic model, first travel solver, nuisance correction, and second travel solver run independently per chunk. "
        "Dropout masks are not used during training or solving; they are applied only to the masked evaluation columns.",
        "",
        "## Aggregate results",
        "",
        "| Stage | Units | Evaluation | Chunk-mean RMSE | Chunk-mean centered RMSE | Log-balanced RMSE | Pooled RMSE | Pooled centered RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['stage']} | {row['units']} | {row['evaluation']} | {fmt(row['macro_mean_rmse'])} | "
            f"{fmt(row['macro_mean_centered_rmse'])} | {fmt(row['log_balanced_mean_rmse'])} | "
            f"{fmt(row['pooled_rmse'])} | {fmt(row['pooled_chunk_centered_rmse'])} |"
        )
    fusion1_all = aggregate_by_key[("fusion1", "unmasked")]
    fusion1_masked = aggregate_by_key[("fusion1", "masked")]
    fusion2_all = aggregate_by_key[("fusion2", "unmasked")]
    fusion2_masked = aggregate_by_key[("fusion2", "masked")]
    accel_all = aggregate_by_key[("accel", "unmasked")]
    accel_masked = aggregate_by_key[("accel", "masked")]
    lines.extend(
        [
            "",
            "## Main observations",
            "",
            f"- Dropout-only evaluation masking reduces pooled acceleration RMSE from "
            f"{accel_all['pooled_rmse']:.2f} to {accel_masked['pooled_rmse']:.2f} m/s².",
            f"- First-fusion pooled chunk-centered RMSE changes from "
            f"{fusion1_all['pooled_chunk_centered_rmse']:.2f} mm unmasked to "
            f"{fusion1_masked['pooled_chunk_centered_rmse']:.2f} mm masked.",
            f"- Final-fusion pooled chunk-centered RMSE changes from "
            f"{fusion2_all['pooled_chunk_centered_rmse']:.2f} mm unmasked to "
            f"{fusion2_masked['pooled_chunk_centered_rmse']:.2f} mm masked.",
            f"- Final fusion improves on first fusion in aggregate in both evaluations: "
            f"{fusion1_all['pooled_chunk_centered_rmse']:.2f} to "
            f"{fusion2_all['pooled_chunk_centered_rmse']:.2f} mm unmasked and "
            f"{fusion1_masked['pooled_chunk_centered_rmse']:.2f} to "
            f"{fusion2_masked['pooled_chunk_centered_rmse']:.2f} mm masked.",
            "",
            "## Per-chunk results",
            "",
            "Travel columns are centered RMSE in millimetres. Acceleration columns are RMSE in m/s² over samples where "
            "the encoder-derived acceleration magnitude exceeds 0.5 m/s². Travel evaluation uses active samples with "
            "positive encoder travel. Masked columns additionally exclude exact core-sensor zero-output samples; no "
            "safety halo is applied. Values displayed as 20.00% are strictly below the 20% selection threshold before rounding.",
            "",
            "| Chunk | Log | Range (s) | Dropout | Train chunks | Accel all | Accel masked | F1 all | F1 masked | F2 all | F2 masked |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['chunk']} | {row['log']} | {row['start_s']:.2f}–{row['stop_s']:.2f} | "
            f"{row['dropout_pct']:.2f}% | {row['training_chunks']} | "
            f"{fmt(row['accel_unmasked_rmse'])} | {fmt(row['accel_masked_rmse'])} | "
            f"{fmt(row['fusion1_unmasked_centered_rmse'])} | {fmt(row['fusion1_masked_centered_rmse'])} | "
            f"{fmt(row['fusion2_unmasked_centered_rmse'])} | {fmt(row['fusion2_masked_centered_rmse'])} |"
        )
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append(f"- `{failure['chunk']}`: {failure['error']}")
    lines.extend(
        [
            "",
            "Detailed RMSE, centered RMSE, MAE, mean error, and sample counts for magnetic-only, first-fusion, and "
            "second-fusion outputs are in `per_chunk_metrics.csv`.",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    specs: list[ChunkSpec] = []
    cache_data: dict[str, np.lib.npyio.NpzFile] = {}
    dropout_masks: dict[str, np.ndarray] = {}
    configs: dict[str, dict[str, Any]] = {}
    accel_truth: dict[str, np.ndarray] = {}

    for log_id in SLAYER_LOGS:
        cache_path = ROOT / "backend" / "run_artifacts" / log_id / "cache" / "all.npz"
        cache = np.load(cache_path, allow_pickle=False)
        cache_data[log_id] = cache
        dropout = core_dropout_mask(log_id)
        dropout_masks[log_id] = dropout
        time_s = flatten(cache["travel__t"])
        active = flatten(cache["active_mask"], bool)
        if len(dropout) != len(active):
            raise ValueError(f"Dropout mask length mismatch for {log_id}: {len(dropout)} vs {len(active)}")
        log_specs = find_maximum_chunks(active, dropout, time_s)
        for ordinal, spec in enumerate(log_specs, start=1):
            specs.append(
                ChunkSpec(
                    chunk_id=f"{log_id}-c{ordinal:02d}",
                    log_id=log_id,
                    start=spec.start,
                    stop=spec.stop,
                    start_s=spec.start_s,
                    stop_s=spec.stop_s,
                    wall_s=spec.wall_s,
                    active_s=spec.active_s,
                    dropout_fraction=spec.dropout_fraction,
                )
            )
        configs[log_id] = deepcopy(resolve_log(log_id).processing_config)
        configs[log_id].setdefault("steps", {}).setdefault(
            "mag_to_travel_model", {}
        )["ref_max_offset_delta_mm"] = None
        travel_time, travel_values = cache_series(cache, "travel")
        travel = flatten(travel_values)
        velocity = np.gradient(travel, travel_time, edge_order=2)
        accel_truth[log_id] = np.gradient(velocity, travel_time, edge_order=2) / 1000.0

    spec_rows = [
        {
            "chunk": spec.chunk_id,
            "log": spec.log_id,
            "start_index": spec.start,
            "stop_index": spec.stop,
            "start_s": spec.start_s,
            "stop_s": spec.stop_s,
            "wall_s": spec.wall_s,
            "active_s": spec.active_s,
            "dropout_pct": spec.dropout_fraction * 100.0,
        }
        for spec in specs
    ]
    write_csv(OUT / "chunk_specs.csv", spec_rows)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    residuals: dict[tuple[str, str], list[np.ndarray]] = {
        (stage, mask): []
        for stage in ("accel", "mag", "fusion1", "fusion2")
        for mask in ("unmasked", "masked")
    }
    logs_dir = OUT / "logs"
    logs_dir.mkdir(exist_ok=True)
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] {spec.chunk_id}", flush=True)
        capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                row, payload = run_chunk(
                    spec,
                    cache_data[spec.log_id],
                    dropout_masks[spec.log_id],
                    configs[spec.log_id],
                    accel_truth[spec.log_id],
                )
            rows.append(row)
            for key, (errors, _) in payload.items():
                residuals[key].append(errors)
        except Exception as error:
            failures.append({"chunk": spec.chunk_id, "error": f"{type(error).__name__}: {error}"})
            capture.write(f"\n{type(error).__name__}: {error}\n")
        (logs_dir / f"{spec.chunk_id}.txt").write_text(capture.getvalue(), encoding="utf-8")

    for cache in cache_data.values():
        cache.close()

    write_csv(OUT / "per_chunk_metrics.csv", rows)
    write_csv(OUT / "failures.csv", failures)
    aggregates = aggregate_rows(rows, residuals) if rows else []
    write_csv(OUT / "aggregate_metrics.csv", aggregates)
    write_report(rows, aggregates, failures)
    print(f"Completed {len(rows)} chunks; {len(failures)} failures")


if __name__ == "__main__":
    main()
