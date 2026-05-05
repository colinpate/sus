from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import dataclass, field
from datetime import datetime
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np

Row = dict[str, object]
Columns = list[tuple[str, str]]
NpzFile = np.lib.npyio.NpzFile
MetricKey = tuple[str, str, str, str]

DEFAULT_LOGS = [
    "log022",
    "log029",
    "log030",
    "log031",
    "log038",
    "log056_ccdh",
    "log060_upperpred",
    "log078",
    "log079",
    "log080",
    "log085",
    "log088",
    "log091",
    "log096",
    "log098",
    "log099",
    "log103",
    "log104",
    "log106",
    "log107",
    "log109",
    "log110",
    "log112",
]
NEW_LOGS = [
    "log103",
    "log104",
    "log106",
    "log107",
    "log109",
    "log110",
    "log112",
]

COMPARISONS = (
    ("travel/mag_model", "travel"),
    ("travel/mag_model/adj", "travel"),
    ("travel/solved", "travel"),
)

DEFAULT_CACHE_ROOT = Path("backend/run_artifacts")
METRICS_FILENAME = "metrics.csv"
MANIFEST_FILENAME = "manifest.json"
REPORT_TEXT_FILENAME = "report.txt"
COMPARE_EPSILON = 1e-12
ANGLE_ERROR_HALO_S = 0.08
MIN_MAG_ANCHOR_MG = 500.0

TRAVEL_BIN_MIN_MM = 0.0
TRAVEL_BIN_MAX_MM = 150.0
TRAVEL_BIN_COUNT = 5
TRAVEL_BIN_MIN_POINTS = 100

LOW_TRAVEL_MAX_MM = 30.0
HIGH_TRAVEL_MIN_MM = 100.0
HIGH_ACCEL_PERCENTILE = 80.0
LOW_MAG_PERCENTILE = 20.0
HIGH_MAG_PERCENTILE = 80.0
HIGH_MAG_THRESH = 25000

POOLED_FEATURES = (
    ("travel", "travel"),
    ("mag", "mag"),
    ("accel_hp_abs", "|accel_hp|"),
    ("dmag_abs", "|dmag|"),
    ("dtravel_abs", "|dtravel|"),
)


@dataclass(frozen=True)
class TravelBin:
    key: str
    label: str
    start_mm: float
    stop_mm: float
    include_stop: bool = False

    def mask(self, travel_mm: np.ndarray) -> np.ndarray:
        upper_mask = travel_mm <= self.stop_mm if self.include_stop else travel_mm < self.stop_mm
        return (travel_mm >= self.start_mm) & upper_mask


@dataclass(frozen=True)
class ErrorStats:
    rmse: float
    mae: float
    mean_error: float


@dataclass(frozen=True)
class LogSummary:
    summary: Row
    comparison_rows: dict[str, Row]


@dataclass(frozen=True)
class Diagnostics:
    stage: Row
    condition_rmse: Row
    condition_occurrence: Row
    solver_delta: Row
    pooled_features: dict[str, np.ndarray]
    binned_summary: Row
    binned_occurrence: Row
    mag_adj_bins: Row
    solved_bins: Row


@dataclass
class AggregatedReport:
    summary_rows: list[Row] = field(default_factory=list)
    error_rows: dict[str, list[Row]] = field(
        default_factory=lambda: {pred_key: [] for pred_key, _ in COMPARISONS}
    )
    diagnostic_stage_rows: list[Row] = field(default_factory=list)
    diagnostic_condition_rows: list[Row] = field(default_factory=list)
    diagnostic_condition_ratio_rows: list[Row] = field(default_factory=list)
    diagnostic_delta_rows: list[Row] = field(default_factory=list)
    diagnostic_binned_summary_rows: list[Row] = field(default_factory=list)
    diagnostic_binned_occurrence_rows: list[Row] = field(default_factory=list)
    diagnostic_mag_adj_bin_rows: list[Row] = field(default_factory=list)
    diagnostic_solved_bin_rows: list[Row] = field(default_factory=list)
    pooled_features: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {key: [] for key, _ in POOLED_FEATURES} | {"solved_abs_err": []}
    )
    failures: list[tuple[str, Exception]] = field(default_factory=list)

    def add_log_summary(self, summary: LogSummary) -> None:
        self.summary_rows.append(summary.summary)
        for pred_key, _ in COMPARISONS:
            self.error_rows[pred_key].append(summary.comparison_rows[pred_key])

    def add_diagnostics(self, diagnostics: Diagnostics) -> None:
        self.diagnostic_stage_rows.append(diagnostics.stage)
        self.diagnostic_condition_rows.append(diagnostics.condition_rmse)
        self.diagnostic_condition_ratio_rows.append(diagnostics.condition_occurrence)
        self.diagnostic_delta_rows.append(diagnostics.solver_delta)
        self.diagnostic_binned_summary_rows.append(diagnostics.binned_summary)
        self.diagnostic_binned_occurrence_rows.append(diagnostics.binned_occurrence)
        self.diagnostic_mag_adj_bin_rows.append(diagnostics.mag_adj_bins)
        self.diagnostic_solved_bin_rows.append(diagnostics.solved_bins)
        for key, values in diagnostics.pooled_features.items():
            self.pooled_features[key].append(values)

    @property
    def has_diagnostics(self) -> bool:
        return bool(self.diagnostic_stage_rows)


def flatten_1d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr.reshape(-1)


def cache_path_for(log_name: str, cache_root: Path) -> Path:
    return cache_root / log_name / "cache" / "all.npz"


def load_cache(log_name: str, cache_root: Path) -> NpzFile:
    cache_path = cache_path_for(log_name, cache_root)
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    return np.load(cache_path)


def bool_1d(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).astype(bool).reshape(-1)


def require_same_shape(context: str, **arrays: np.ndarray) -> None:
    shapes = {name: np.asarray(values).shape for name, values in arrays.items()}
    if len(set(shapes.values())) > 1:
        shape_text = ", ".join(f"{name}={shape}" for name, shape in shapes.items())
        raise ValueError(f"{context}: arrays do not align ({shape_text})")


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        raise ValueError("Need at least one array to build a finite mask")

    mask = np.ones_like(arrays[0], dtype=bool)
    for arr in arrays:
        require_same_shape("finite mask", reference=arrays[0], array=arr)
        mask &= np.isfinite(arr)
    return mask


def infer_dt_seconds(time_s: np.ndarray) -> float:
    time_s = flatten_1d(time_s)
    if len(time_s) < 2:
        return 0.0

    diffs = np.diff(time_s)
    finite_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(finite_diffs) == 0:
        return 0.0
    return float(np.median(finite_diffs))


def project_mask_to_timeline(
    source_t: np.ndarray,
    source_mask: np.ndarray,
    target_t: np.ndarray,
    *,
    halo_s: float = ANGLE_ERROR_HALO_S,
) -> np.ndarray:
    source_t = flatten_1d(source_t)
    source_mask = bool_1d(source_mask)
    target_t = flatten_1d(target_t)

    require_same_shape("angle bad mask", source_t=source_t, source_mask=source_mask)
    if len(target_t) == 0 or not np.any(source_mask):
        return np.zeros(len(target_t), dtype=bool)

    bad_idx = np.flatnonzero(source_mask)
    split_idx = np.where(np.diff(bad_idx) > 1)[0]
    run_starts = np.r_[bad_idx[0], bad_idx[split_idx + 1]]
    run_ends = np.r_[bad_idx[split_idx], bad_idx[-1]]

    projected = np.zeros(len(target_t), dtype=bool)
    for start_idx, end_idx in zip(run_starts, run_ends):
        start_t = source_t[start_idx] - halo_s
        end_t = source_t[end_idx] + halo_s
        start = np.searchsorted(target_t, start_t, side="left")
        end = np.searchsorted(target_t, end_t, side="right")
        projected[start:end] = True
    return projected


def build_angle_bad_mask(cache: NpzFile, target_t: np.ndarray) -> np.ndarray:
    target_t = flatten_1d(target_t)
    if "angle/bad_mask__x" not in cache or "angle/bad_mask__t" not in cache:
        return np.zeros(len(target_t), dtype=bool)

    return project_mask_to_timeline(
        cache["angle/bad_mask__t"],
        bool_1d(cache["angle/bad_mask__x"]),
        target_t,
    )


def error_vector(x: np.ndarray, gt: np.ndarray, *, center: bool = False) -> np.ndarray:
    x = flatten_1d(x)
    gt = flatten_1d(gt)
    require_same_shape("error inputs", x=x, gt=gt)

    if center:
        x = x - np.mean(x)
        gt = gt - np.mean(gt)
    return x - gt


def summarize_error(
    x: np.ndarray,
    gt: np.ndarray,
    *,
    center: bool = False,
    threshold: float | None = None,
) -> ErrorStats:
    x = flatten_1d(x)
    gt = flatten_1d(gt)
    require_same_shape("error inputs", x=x, gt=gt)

    if threshold is not None:
        mask = np.abs(gt) > threshold
        x = x[mask]
        gt = gt[mask]

    err = error_vector(x, gt, center=center)
    return ErrorStats(
        rmse=float(np.sqrt(np.mean(err**2))),
        mae=float(np.mean(np.abs(err))),
        mean_error=float(np.mean(err)),
    )


def get_error_stats(
    x: np.ndarray,
    gt: np.ndarray,
    center: bool = False,
    thresh: float | None = None,
) -> tuple[float, float, float]:
    stats = summarize_error(x, gt, center=center, threshold=thresh)
    return stats.rmse, stats.mae, stats.mean_error


def get_error_vector(x: np.ndarray, gt: np.ndarray, center: bool = False) -> np.ndarray:
    return error_vector(x, gt, center=center)


def build_mask(
    cache: NpzFile,
    pred_key: str,
    gt_key: str,
    error_threshold: float | None = None,
) -> np.ndarray:
    pred = flatten_1d(cache[f"{pred_key}__x"])
    gt = flatten_1d(cache[f"{gt_key}__x"])
    boring_mask = bool_1d(cache["boring_mask"])
    gt_time_s = flatten_1d(cache[f"{gt_key}__t"])

    require_same_shape(
        f"cache arrays for {pred_key} vs {gt_key}",
        pred=pred,
        gt=gt,
        boring_mask=boring_mask,
        gt_time_s=gt_time_s,
    )

    mask = boring_mask & finite_mask(pred, gt) & ~build_angle_bad_mask(cache, gt_time_s)
    if error_threshold is not None:
        mask &= np.abs(gt) > error_threshold
    return mask


def summarize_log(
    log_name: str,
    cache_root: Path,
    center_errors: bool,
    error_threshold: float | None,
) -> tuple[Row, dict[str, Row]]:
    summary = summarize_log_cache(log_name, cache_root, center_errors, error_threshold)
    return summary.summary, summary.comparison_rows


def summarize_log_cache(
    log_name: str,
    cache_root: Path,
    center_errors: bool,
    error_threshold: float | None,
) -> LogSummary:
    cache = load_cache(log_name, cache_root)
    time_s = flatten_1d(cache["travel__t"])
    boring_mask = bool_1d(cache["boring_mask"])
    require_same_shape(f"{log_name}: duration arrays", time_s=time_s, boring_mask=boring_mask)

    dt_s = infer_dt_seconds(time_s)
    total_seconds = len(time_s) * dt_s
    boring_seconds = int(np.sum(boring_mask)) * dt_s

    summary_row: Row = {
        "log": log_name,
        "samples": len(time_s),
        "dt_ms": dt_s * 1000.0,
        "total_s": total_seconds,
        "boring_s": boring_seconds,
        "boring_pct": percentage(boring_seconds, total_seconds),
    }

    comparison_rows: dict[str, Row] = {}
    for pred_key, gt_key in COMPARISONS:
        pred = flatten_1d(cache[f"{pred_key}__x"])
        gt = flatten_1d(cache[f"{gt_key}__x"])
        mask = build_mask(cache, pred_key, gt_key, error_threshold=error_threshold)

        masked_pred = pred[mask]
        masked_gt = gt[mask]
        if len(masked_pred) == 0:
            raise ValueError(f"{log_name}: no finite boring-mask samples for {pred_key} vs {gt_key}")

        stats = summarize_error(masked_pred, masked_gt, center=center_errors)
        binned = summarize_binned_rmse(
            error_vector(masked_pred, masked_gt, center=center_errors),
            masked_gt,
        )
        comparison_rows[pred_key] = {
            "log": log_name,
            "t": int(len(masked_pred) / 100),
            "rmse": stats.rmse,
            "bin_rmse": binned["bin_rmse"],
            "mae": stats.mae,
            "me": stats.mean_error,
            "rms_travel": float(np.std(masked_gt)),
        }

    return LogSummary(summary=summary_row, comparison_rows=comparison_rows)


def dense_index_mask(length: int, indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    idx = np.asarray(indices, dtype=int).reshape(-1)
    idx = idx[(idx >= 0) & (idx < length)]
    mask[idx] = True
    return mask


def make_masked_error(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, center_errors: bool) -> np.ndarray:
    masked_pred = flatten_1d(pred)[mask]
    masked_gt = flatten_1d(gt)[mask]
    return error_vector(masked_pred, masked_gt, center=center_errors)


def masked_rmse(err: np.ndarray, cond: np.ndarray) -> float:
    err = flatten_1d(err)
    cond = bool_1d(cond)
    if cond.shape != err.shape or not np.any(cond):
        return float("nan")
    return float(np.sqrt(np.mean(err[cond] ** 2)))


def masked_ratio_pct(cond: np.ndarray) -> float:
    cond = bool_1d(cond)
    if len(cond) == 0:
        return float("nan")
    return 100.0 * float(np.mean(cond))


def maybe_percentile(values: np.ndarray, q: float) -> float:
    values = flatten_1d(values)
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(values, q))


def percentage(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole > 0 else float("nan")


def format_bin_edge(value: float) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def format_threshold_label(prefix: str, value: float) -> str:
    return f"{prefix}{format_bin_edge(value)}"


def make_travel_bins() -> tuple[TravelBin, ...]:
    edges = np.linspace(TRAVEL_BIN_MIN_MM, TRAVEL_BIN_MAX_MM, TRAVEL_BIN_COUNT + 1, dtype=float)
    bins: list[TravelBin] = []
    for idx, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        bins.append(
            TravelBin(
                key=f"bin{idx}",
                label=f"{format_bin_edge(start)}-{format_bin_edge(stop)}",
                start_mm=float(start),
                stop_mm=float(stop),
                include_stop=idx == (len(edges) - 2),
            )
        )
    return tuple(bins)


def get_travel_bin_specs() -> list[tuple[str, str, float, float, bool]]:
    return [
        (bin_spec.key, bin_spec.label, bin_spec.start_mm, bin_spec.stop_mm, bin_spec.include_stop)
        for bin_spec in make_travel_bins()
    ]


def travel_bin_columns(metric_suffix: str) -> list[tuple[str, str]]:
    return [(bin_spec.key + f"_{metric_suffix}", bin_spec.label) for bin_spec in make_travel_bins()]


def build_travel_bin_masks(travel: np.ndarray) -> list[np.ndarray]:
    travel = flatten_1d(travel)
    return [bin_spec.mask(travel) for bin_spec in make_travel_bins()]


def summarize_binned_rmse(err: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    err = flatten_1d(err)
    gt = flatten_1d(gt)
    require_same_shape("binned RMSE inputs", err=err, gt=gt)

    bin_masks = build_travel_bin_masks(gt)
    in_range_mask = np.zeros_like(gt, dtype=bool)
    eligible_mask = np.zeros_like(gt, dtype=bool)
    row: dict[str, float] = {}
    eligible_bin_mses: list[float] = []

    for bin_spec, mask in zip(make_travel_bins(), bin_masks):
        count = int(np.sum(mask))
        in_range_mask |= mask
        row[f"{bin_spec.key}_rmse"] = masked_rmse(err, mask)
        row[f"{bin_spec.key}_n"] = float(count)
        if count >= TRAVEL_BIN_MIN_POINTS:
            eligible_mask |= mask
            eligible_bin_mses.append(float(np.mean(err[mask] ** 2)))

    in_range_n = int(np.sum(in_range_mask))
    for bin_spec in make_travel_bins():
        count = int(row[f"{bin_spec.key}_n"])
        row[f"{bin_spec.key}_pct"] = percentage(count, in_range_n)

    row["bin_rmse"] = float(np.sqrt(np.mean(eligible_bin_mses))) if eligible_bin_mses else float("nan")
    row["eligible_bins"] = float(len(eligible_bin_mses))
    row["eligible_n"] = float(np.sum(eligible_mask))
    row["in_range_n"] = float(in_range_n)
    row["in_range_pct"] = percentage(in_range_n, len(gt))
    return row


def diagnostic_rows(
    log_name: str,
    cache_root: Path,
    center_errors: bool,
) -> tuple[Row, Row, Row, Row, dict[str, np.ndarray], Row, Row, Row, Row]:
    diagnostics = summarize_diagnostics(log_name, cache_root, center_errors)
    return (
        diagnostics.stage,
        diagnostics.condition_rmse,
        diagnostics.condition_occurrence,
        diagnostics.solver_delta,
        diagnostics.pooled_features,
        diagnostics.binned_summary,
        diagnostics.binned_occurrence,
        diagnostics.mag_adj_bins,
        diagnostics.solved_bins,
    )


def summarize_diagnostics(log_name: str, cache_root: Path, center_errors: bool) -> Diagnostics:
    cache = load_cache(log_name, cache_root)

    boring_mask = bool_1d(cache["boring_mask"])
    travel = flatten_1d(cache["travel__x"])
    mag = flatten_1d(cache["mag/proj/corr/lpf__x"])
    accel_hp_abs = np.abs(flatten_1d(cache["accel/lpfhp/proj__x"]))
    bad_mag_mask = bool_1d(cache["mag/proj/bad_mask__x"])
    zv_mask = dense_index_mask(len(travel), cache["mag_zv_points"])
    mag_baseline = float(flatten_1d(cache["mag_baseline"])[0])
    mag_anchor_thresh = max(MIN_MAG_ANCHOR_MG, mag_baseline)

    require_same_shape(
        f"{log_name}: diagnostic arrays",
        travel=travel,
        boring_mask=boring_mask,
        mag=mag,
        accel_hp_abs=accel_hp_abs,
        bad_mag_mask=bad_mag_mask,
        zv_mask=zv_mask,
    )

    angle_bad_mask = build_angle_bad_mask(cache, cache["travel__t"])
    mask = boring_mask & finite_mask(travel, mag, accel_hp_abs) & ~angle_bad_mask
    if not np.any(mask):
        raise ValueError(f"{log_name}: no finite diagnostic samples on boring_mask")

    masked_travel = travel[mask]
    masked_mag = mag[mask]
    masked_accel_hp_abs = accel_hp_abs[mask]
    masked_bad_mag = bad_mag_mask[mask]
    masked_zv = zv_mask[mask]

    mag_model_err = make_masked_error(cache["travel/mag_model__x"], travel, mask, center_errors)
    mag_adj_err = make_masked_error(cache["travel/mag_model/adj__x"], travel, mask, center_errors)
    solved_err = make_masked_error(cache["travel/solved__x"], travel, mask, center_errors)

    mag_adj_binned = summarize_binned_rmse(mag_adj_err, masked_travel)
    solved_binned = summarize_binned_rmse(solved_err, masked_travel)
    conditions = build_diagnostic_conditions(
        masked_travel,
        masked_mag,
        masked_accel_hp_abs,
        masked_bad_mag,
        masked_zv,
        mag_anchor_thresh,
    )

    all_samples = np.ones_like(masked_travel, dtype=bool)
    mag_adj_rmse = masked_rmse(mag_adj_err, all_samples)
    solved_rmse = masked_rmse(solved_err, all_samples)

    stage_row: Row = {
        "log": log_name,
        "n": int(np.sum(mask)),
        "mag_model_rmse": masked_rmse(mag_model_err, all_samples),
        "mag_adj_rmse": mag_adj_rmse,
        "solved_rmse": solved_rmse,
        "solver_delta": solved_rmse - mag_adj_rmse,
        "mag_anchor_pct": masked_ratio_pct(conditions["anchor_on"]),
        "mag_abs_anchor_pct": masked_ratio_pct(conditions["anchor_abs"]),
        "bad_mag_pct": masked_ratio_pct(masked_bad_mag),
        "zvs_per_1k": 1000.0 * float(np.mean(masked_zv)),
    }

    condition_row = condition_metric_row(log_name, solved_err, conditions)
    condition_ratio_row = condition_occurrence_row(log_name, conditions)
    solver_delta_row = solver_delta_metric_row(log_name, solved_err, mag_adj_err, conditions, stage_row["solver_delta"])

    pooled = {
        "travel": masked_travel,
        "mag": masked_mag,
        "accel_hp_abs": masked_accel_hp_abs,
        "dmag_abs": np.abs(np.gradient(mag))[mask],
        "dtravel_abs": np.abs(np.gradient(travel))[mask],
        "solved_abs_err": np.abs(solved_err),
    }

    binned_summary_row: Row = {
        "log": log_name,
        "range_pct": solved_binned["in_range_pct"],
        "eligible_bins": int(solved_binned["eligible_bins"]),
        "mag_adj_bin_rmse": mag_adj_binned["bin_rmse"],
        "solved_bin_rmse": solved_binned["bin_rmse"],
    }
    binned_occurrence_row: Row = {"log": log_name}
    mag_adj_bin_row: Row = {"log": log_name}
    solved_bin_row: Row = {"log": log_name}
    for bin_spec in make_travel_bins():
        binned_occurrence_row[f"{bin_spec.key}_pct"] = solved_binned[f"{bin_spec.key}_pct"]
        mag_adj_bin_row[f"{bin_spec.key}_rmse"] = mag_adj_binned[f"{bin_spec.key}_rmse"]
        solved_bin_row[f"{bin_spec.key}_rmse"] = solved_binned[f"{bin_spec.key}_rmse"]

    return Diagnostics(
        stage=stage_row,
        condition_rmse=condition_row,
        condition_occurrence=condition_ratio_row,
        solver_delta=solver_delta_row,
        pooled_features=pooled,
        binned_summary=binned_summary_row,
        binned_occurrence=binned_occurrence_row,
        mag_adj_bins=mag_adj_bin_row,
        solved_bins=solved_bin_row,
    )


def build_diagnostic_conditions(
    travel: np.ndarray,
    mag: np.ndarray,
    accel_hp_abs: np.ndarray,
    bad_mag: np.ndarray,
    zv: np.ndarray,
    mag_anchor_thresh: float,
) -> dict[str, np.ndarray]:
    return {
        "low_trav": travel < LOW_TRAVEL_MAX_MM,
        "high_trav": travel > HIGH_TRAVEL_MIN_MM,
        "high_acc": accel_hp_abs > maybe_percentile(accel_hp_abs, HIGH_ACCEL_PERCENTILE),
        "low_mag": mag < maybe_percentile(mag, LOW_MAG_PERCENTILE),
        #"high_mag": mag > maybe_percentile(mag, HIGH_MAG_PERCENTILE),
        "high_mag": mag > HIGH_MAG_THRESH,
        "bad_mag": bad_mag,
        "zv": zv,
        "anchor_on": mag > mag_anchor_thresh,
        "anchor_off": mag <= mag_anchor_thresh,
        "anchor_abs": np.abs(mag) > mag_anchor_thresh,
    }


def condition_metric_row(log_name: str, err: np.ndarray, conditions: dict[str, np.ndarray]) -> Row:
    return {
        "log": log_name,
        "low_trav": masked_rmse(err, conditions["low_trav"]),
        "high_trav": masked_rmse(err, conditions["high_trav"]),
        "high_acc": masked_rmse(err, conditions["high_acc"]),
        "low_mag": masked_rmse(err, conditions["low_mag"]),
        "high_mag": masked_rmse(err, conditions["high_mag"]),
        "bad_mag": masked_rmse(err, conditions["bad_mag"]),
        "zv": masked_rmse(err, conditions["zv"]),
    }


def condition_occurrence_row(log_name: str, conditions: dict[str, np.ndarray]) -> Row:
    return {
        "log": log_name,
        "low_trav": masked_ratio_pct(conditions["low_trav"]),
        "high_trav": masked_ratio_pct(conditions["high_trav"]),
        "high_acc": masked_ratio_pct(conditions["high_acc"]),
        "low_mag": masked_ratio_pct(conditions["low_mag"]),
        "high_mag": masked_ratio_pct(conditions["high_mag"]),
        "bad_mag": masked_ratio_pct(conditions["bad_mag"]),
        "zv": masked_ratio_pct(conditions["zv"]),
    }


def solver_delta_metric_row(
    log_name: str,
    solved_err: np.ndarray,
    mag_adj_err: np.ndarray,
    conditions: dict[str, np.ndarray],
    all_delta: object,
) -> Row:
    return {
        "log": log_name,
        "all_d": all_delta,
        "low_trav_d": rmse_delta(solved_err, mag_adj_err, conditions["low_trav"]),
        "high_trav_d": rmse_delta(solved_err, mag_adj_err, conditions["high_trav"]),
        "anchor_off_d": rmse_delta(solved_err, mag_adj_err, conditions["anchor_off"]),
        "anchor_on_d": rmse_delta(solved_err, mag_adj_err, conditions["anchor_on"]),
        "bad_mag_d": rmse_delta(solved_err, mag_adj_err, conditions["bad_mag"]),
        "zv_d": rmse_delta(solved_err, mag_adj_err, conditions["zv"]),
    }


def rmse_delta(left_err: np.ndarray, right_err: np.ndarray, cond: np.ndarray) -> float:
    return masked_rmse(left_err, cond) - masked_rmse(right_err, cond)


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = flatten_1d(x)
    y = flatten_1d(y)
    require_same_shape("correlation inputs", x=x, y=y)

    finite = np.isfinite(x) & np.isfinite(y)
    if np.sum(finite) < 2:
        return float("nan")

    x = x[finite]
    y = y[finite]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def format_value(value: object) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "nan"
        return f"{float(value):.2f}"
    return str(value)


def resolve_sort_key(columns: Columns, rows: list[Row], sort_key: str) -> str:
    if not rows:
        return sort_key
    if sort_key in rows[0]:
        return sort_key

    label_to_key = {label: key for key, label in columns}
    resolved = label_to_key.get(sort_key)
    if resolved is not None and resolved in rows[0]:
        return resolved

    return "log" if "log" in rows[0] else columns[0][0]


def sort_value(row: Row, sort_key: str) -> tuple[int, object]:
    value = row.get(sort_key, row.get("log", ""))
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        return (1, 0.0) if not np.isfinite(value) else (0, value)
    return (0, str(value))


def print_table(
    title: str,
    columns: Columns,
    rows: Iterable[Row],
    sort_key: str = "n",
    *,
    reverse: bool = False,
) -> None:
    rows = list(rows)
    sort_key = resolve_sort_key(columns, rows, sort_key)
    rows.sort(key=lambda row: sort_value(row, sort_key), reverse=reverse)

    formatted_rows = [{key: format_value(row.get(key, "")) for key, _ in columns} for row in rows]
    widths = [
        max(len(label), max((len(row[key]) for row in formatted_rows), default=0))
        for key, label in columns
    ]

    print(title)
    print(format_table_line(columns, widths, labels=True))
    print("  ".join("-" * width for width in widths))
    for row in formatted_rows:
        print(format_table_line(columns, widths, row=row))
    print()


def format_table_line(
    columns: Columns,
    widths: list[int],
    *,
    labels: bool = False,
    row: dict[str, str] | None = None,
) -> str:
    cells: list[str] = []
    for (key, label), width in zip(columns, widths):
        if labels:
            value = label
        else:
            if row is None:
                raise ValueError("row is required when formatting table values")
            value = row[key]
        left_aligned = {"section", "log", "feature", "item", "metric"}
        cells.append(value.ljust(width) if key in left_aligned else value.rjust(width))
    return "  ".join(cells)


def collect_report(
    log_names: Iterable[str],
    cache_root: Path,
    *,
    center_errors: bool,
    error_threshold: float | None,
    include_diagnostics: bool,
) -> AggregatedReport:
    report = AggregatedReport()

    for log_name in log_names:
        try:
            log_summary = summarize_log_cache(log_name, cache_root, center_errors, error_threshold)
        except Exception as exc:  # Keep the report useful even if a cache is missing or malformed.
            report.failures.append((log_name, exc))
            continue

        report.add_log_summary(log_summary)
        if not include_diagnostics:
            continue

        try:
            diagnostics = summarize_diagnostics(log_name, cache_root, center_errors)
        except Exception as exc:  # Keep the main stats usable even if diagnostics fail.
            report.failures.append((f"{log_name} (diagnostics)", exc))
        else:
            report.add_diagnostics(diagnostics)

    return report


def print_cache_summary(report: AggregatedReport) -> None:
    print_table(
        title="Cache duration summary",
        columns=[
            ("log", "log"),
            ("samples", "samples"),
            ("dt_ms", "dt_ms"),
            ("total_s", "total_s"),
            ("boring_s", "boring_s"),
            ("boring_pct", "boring_%"),
        ],
        rows=report.summary_rows,
        sort_key="log",
    )


def print_error_summaries(report: AggregatedReport, *, center_errors: bool, sort_key: str) -> None:
    center_label = "centered" if center_errors else "raw"
    for pred_key, gt_key in COMPARISONS:
        print_table(
            title=f"Error stats on boring_mask ({center_label}): {pred_key} vs {gt_key}",
            columns=[
                ("log", "log"),
                ("t", "t"),
                ("rmse", "rmse"),
                ("bin_rmse", "bin_rmse"),
                ("mae", "mae"),
                ("me", "me"),
                ("rms_travel", "rms_trav"),
            ],
            rows=report.error_rows[pred_key],
            sort_key=sort_key,
        )


def print_failures(failures: list[tuple[str, Exception]]) -> None:
    if not failures:
        return
    print("Skipped logs")
    for log_name, exc in failures:
        print(f"{log_name:>12}  {exc}")


def diagnostic_condition_columns() -> Columns:
    return [
        ("log", "log"),
        ("low_trav", format_threshold_label("trav<", LOW_TRAVEL_MAX_MM)),
        ("high_trav", format_threshold_label("trav>", HIGH_TRAVEL_MIN_MM)),
        ("high_acc", f"acc>p{format_bin_edge(HIGH_ACCEL_PERCENTILE)}"),
        ("low_mag", f"mag<p{format_bin_edge(LOW_MAG_PERCENTILE)}"),
        #("high_mag", f"mag>p{format_bin_edge(HIGH_MAG_PERCENTILE)}"),
        ("high_mag", format_threshold_label("mag>", HIGH_MAG_THRESH)),
        ("bad_mag", "bad_mag"),
        ("zv", "zv"),
    ]


def print_diagnostics(report: AggregatedReport, *, center_errors: bool, sort_key: str) -> None:
    if not report.has_diagnostics:
        return

    center_label = "centered" if center_errors else "raw"
    print_table(
        title=(
            f"Binned RMSE summary on boring_mask ({center_label}, GT {format_bin_edge(TRAVEL_BIN_MIN_MM)}-"
            f"{format_bin_edge(TRAVEL_BIN_MAX_MM)} mm, bins>={TRAVEL_BIN_MIN_POINTS})"
        ),
        columns=[
            ("log", "log"),
            ("range_pct", "range_%"),
            ("eligible_bins", "n_bins"),
            ("mag_adj_bin_rmse", "mag_adj"),
            ("solved_bin_rmse", "solved"),
        ],
        rows=report.diagnostic_binned_summary_rows,
        sort_key=sort_key,
    )

    print_table(
        title=f"Stage RMSE summary on boring_mask ({center_label})",
        columns=[
            ("log", "log"),
            ("n", "n"),
            ("mag_model_rmse", "mag_rmse"),
            ("mag_adj_rmse", "mag_adj"),
            ("solved_rmse", "solved"),
            ("solver_delta", "solv-mag"),
            ("mag_anchor_pct", "anchor_%"),
            ("mag_abs_anchor_pct", "|mag|_%"),
            ("bad_mag_pct", "badmag_%"),
            ("zvs_per_1k", "zv/1k"),
        ],
        rows=report.diagnostic_stage_rows,
        sort_key=sort_key,
    )

    print_table(
        title=f"Conditioned solved RMSE on boring_mask ({center_label})",
        columns=diagnostic_condition_columns(),
        rows=report.diagnostic_condition_rows,
        sort_key=sort_key,
    )

    print_table(
        title=f"Condition occurrence on boring_mask ({center_label}, % of diagnostic samples)",
        columns=diagnostic_condition_columns(),
        rows=report.diagnostic_condition_ratio_rows,
        sort_key=sort_key,
    )

    print_table(
        title=f"Solver delta vs mag_adj RMSE on boring_mask ({center_label}, negative is better)",
        columns=[
            ("log", "log"),
            ("all_d", "all"),
            ("low_trav_d", format_threshold_label("trav<", LOW_TRAVEL_MAX_MM)),
            ("high_trav_d", format_threshold_label("trav>", HIGH_TRAVEL_MIN_MM)),
            ("anchor_off_d", "anchor_off"),
            ("anchor_on_d", "anchor_on"),
            ("bad_mag_d", "bad_mag"),
            ("zv_d", "zv"),
        ],
        rows=report.diagnostic_delta_rows,
        sort_key=sort_key,
    )

    print_table(
        title=(
            f"Travel-bin occurrence on boring_mask (% of in-range samples, GT {format_bin_edge(TRAVEL_BIN_MIN_MM)}-"
            f"{format_bin_edge(TRAVEL_BIN_MAX_MM)} mm)"
        ),
        columns=[("log", "log"), *travel_bin_columns("pct")],
        rows=report.diagnostic_binned_occurrence_rows,
        sort_key=sort_key,
    )

    print_table(
        title=(
            f"Per-bin RMSE on boring_mask ({center_label}): travel/mag_model/adj vs travel "
            f"[GT {format_bin_edge(TRAVEL_BIN_MIN_MM)}-{format_bin_edge(TRAVEL_BIN_MAX_MM)} mm]"
        ),
        columns=[("log", "log"), *travel_bin_columns("rmse")],
        rows=report.diagnostic_mag_adj_bin_rows,
        sort_key=sort_key,
    )

    print_table(
        title=(
            f"Per-bin RMSE on boring_mask ({center_label}): travel/solved vs travel "
            f"[GT {format_bin_edge(TRAVEL_BIN_MIN_MM)}-{format_bin_edge(TRAVEL_BIN_MAX_MM)} mm]"
        ),
        columns=[("log", "log"), *travel_bin_columns("rmse")],
        rows=report.diagnostic_solved_bin_rows,
        sort_key=sort_key,
    )

    print_pooled_correlations(report, center_label)


def print_pooled_correlations(report: AggregatedReport, center_label: str) -> None:
    solved_abs_err = np.concatenate(report.pooled_features["solved_abs_err"])
    pooled_rows = [
        {
            "feature": label,
            "corr": corrcoef_safe(np.concatenate(report.pooled_features[key]), solved_abs_err),
        }
        for key, label in POOLED_FEATURES
    ]
    print_table(
        title=f"Pooled correlation with |travel/solved error| on boring_mask ({center_label})",
        columns=[
            ("feature", "feature"),
            ("corr", "corr"),
        ],
        rows=pooled_rows,
        sort_key="feature",
    )


def render_report(report: AggregatedReport, *, center_errors: bool, sort_key: str) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print_cache_summary(report)
        print_error_summaries(report, center_errors=center_errors, sort_key=sort_key)
        print_failures(report.failures)
        print_diagnostics(report, center_errors=center_errors, sort_key=sort_key)
    return buffer.getvalue()


def write_csv(path: Path, rows: Iterable[Row], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = collect_fieldnames(rows)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def collect_fieldnames(rows: list[Row]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def csv_value(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def error_export_rows(report: AggregatedReport) -> list[Row]:
    rows: list[Row] = []
    for pred_key, _ in COMPARISONS:
        for row in report.error_rows[pred_key]:
            rows.append({"comparison": pred_key, **row})
    return rows


def pooled_correlation_rows(report: AggregatedReport) -> list[Row]:
    if not report.has_diagnostics:
        return []

    solved_abs_err = np.concatenate(report.pooled_features["solved_abs_err"])
    return [
        {
            "feature": label,
            "corr": corrcoef_safe(np.concatenate(report.pooled_features[key]), solved_abs_err),
        }
        for key, label in POOLED_FEATURES
    ]


def wide_report_tables(report: AggregatedReport) -> list[tuple[str, list[Row], list[str] | None]]:
    tables: list[tuple[str, list[Row], list[str] | None]] = [
        ("summary.csv", report.summary_rows, None),
        ("errors.csv", error_export_rows(report), None),
    ]

    if report.failures:
        failures = [{"log": log_name, "error": str(exc)} for log_name, exc in report.failures]
        tables.append(("failures.csv", failures, ["log", "error"]))

    if report.has_diagnostics:
        tables.extend(
            [
                ("diagnostics_stage.csv", report.diagnostic_stage_rows, None),
                ("diagnostics_condition_rmse.csv", report.diagnostic_condition_rows, None),
                ("diagnostics_condition_occurrence.csv", report.diagnostic_condition_ratio_rows, None),
                ("diagnostics_solver_delta.csv", report.diagnostic_delta_rows, None),
                ("diagnostics_binned_summary.csv", report.diagnostic_binned_summary_rows, None),
                ("diagnostics_binned_occurrence.csv", report.diagnostic_binned_occurrence_rows, None),
                ("diagnostics_mag_adj_bins.csv", report.diagnostic_mag_adj_bin_rows, None),
                ("diagnostics_solved_bins.csv", report.diagnostic_solved_bin_rows, None),
                ("pooled_correlations.csv", pooled_correlation_rows(report), None),
            ]
        )

    return tables


def tidy_metric_rows(report: AggregatedReport) -> list[Row]:
    rows: list[Row] = []
    add_tidy_metrics(rows, "summary", report.summary_rows)

    for pred_key, _ in COMPARISONS:
        add_tidy_metrics(rows, "error", report.error_rows[pred_key], comparison=pred_key)

    if report.has_diagnostics:
        add_tidy_metrics(rows, "diagnostic_stage", report.diagnostic_stage_rows)
        add_tidy_metrics(rows, "diagnostic_condition_rmse", report.diagnostic_condition_rows)
        add_tidy_metrics(rows, "diagnostic_condition_occurrence", report.diagnostic_condition_ratio_rows)
        add_tidy_metrics(rows, "diagnostic_solver_delta", report.diagnostic_delta_rows)
        add_tidy_metrics(rows, "diagnostic_binned_summary", report.diagnostic_binned_summary_rows)
        add_tidy_metrics(rows, "diagnostic_binned_occurrence", report.diagnostic_binned_occurrence_rows)
        add_tidy_metrics(rows, "diagnostic_mag_adj_bins", report.diagnostic_mag_adj_bin_rows)
        add_tidy_metrics(rows, "diagnostic_solved_bins", report.diagnostic_solved_bin_rows)
        add_tidy_metrics(rows, "pooled_correlation", pooled_correlation_rows(report), comparison_key="feature")

    return rows


def add_tidy_metrics(
    output: list[Row],
    section: str,
    rows: Iterable[Row],
    *,
    comparison: str = "",
    comparison_key: str | None = None,
) -> None:
    for row in rows:
        log_name = str(row.get("log", ""))
        item = comparison
        if comparison_key is not None:
            item = str(row.get(comparison_key, ""))

        for metric, value in row.items():
            if metric in {"log", "comparison", comparison_key}:
                continue
            if not is_numeric_metric(value):
                continue
            output.append(
                {
                    "section": section,
                    "log": log_name,
                    "comparison": item,
                    "metric": metric,
                    "value": float(value),
                }
            )


def is_numeric_metric(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def save_report(
    report: AggregatedReport,
    output_dir: Path,
    *,
    report_text: str,
    args: argparse.Namespace,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    report_text_path = output_dir / REPORT_TEXT_FILENAME
    report_text_path.write_text(report_text, encoding="utf-8")
    written.append(report_text_path)

    for filename, rows, fieldnames in wide_report_tables(report):
        path = output_dir / filename
        write_csv(path, rows, fieldnames)
        written.append(path)

    metrics_path = output_dir / METRICS_FILENAME
    write_csv(
        metrics_path,
        tidy_metric_rows(report),
        ["section", "log", "comparison", "metric", "value"],
    )
    written.append(metrics_path)

    manifest_path = output_dir / MANIFEST_FILENAME
    manifest = build_manifest(report, args, written)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=json_default) + "\n", encoding="utf-8")
    written.append(manifest_path)
    return written


def build_manifest(report: AggregatedReport, args: argparse.Namespace, written_files: list[Path]) -> Row:
    return {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": {
            "logs": list(args.logs),
            "cache_root": str(args.cache_root),
            "center_errors": bool(args.center_errors),
            "error_threshold": args.error_threshold,
            "sort_key": args.sort_key,
            "deep_dive": bool(args.deep_dive),
        },
        "report": {
            "logs_summarized": len(report.summary_rows),
            "failures": len(report.failures),
            "diagnostics_included": report.has_diagnostics,
        },
        "python": sys.version.split()[0],
        "git": git_info(Path(__file__).resolve().parents[1]),
        "files": [path.name for path in written_files],
    }


def git_info(repo_root: Path) -> Row:
    commit = run_git(repo_root, "rev-parse", "HEAD")
    branch = run_git(repo_root, "branch", "--show-current")
    status = run_git(repo_root, "status", "--short")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def run_git(repo_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def load_saved_metrics(run_dir: Path) -> dict[MetricKey, float]:
    metrics_path = run_dir / METRICS_FILENAME
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected saved metrics at {metrics_path}")

    metrics: dict[MetricKey, float] = {}
    with metrics_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (
                row.get("section", ""),
                row.get("log", ""),
                row.get("comparison", ""),
                row.get("metric", ""),
            )
            metrics[key] = float(row.get("value", "nan"))
    return metrics


def compare_saved_runs(base_dir: Path, current_dir: Path, *, top_n: int, item: str | None = None, compare_metric: str | None = None) -> str:
    top_n = max(1, top_n)
    base = load_saved_metrics(base_dir)
    current = load_saved_metrics(current_dir)
    common_keys = sorted(set(base) & set(current))
    added = sorted(set(current) - set(base))
    removed = sorted(set(base) - set(current))

    rows: list[Row] = []
    for section, log_name, comparison, metric in common_keys:
        before = base[(section, log_name, comparison, metric)]
        after = current[(section, log_name, comparison, metric)]
        delta = after - before
        if item is not None and comparison != item:
            continue
        if compare_metric is not None and metric != compare_metric:
            continue
        rows.append(
            {
                "section": section,
                "log": log_name,
                "item": comparison,
                "metric": metric,
                "before": before,
                "after": after,
                "delta": delta,
                "pct_delta": percent_delta(before, delta),
                "abs_delta": abs(delta),
            }
        )

    rows = [
        row
        for row in rows
        if np.isfinite(float(row["abs_delta"])) and float(row["abs_delta"]) > COMPARE_EPSILON
    ]
    rows.sort(key=lambda row: row["log"], reverse=True)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print("Saved stats comparison")
        print(f"baseline: {base_dir}")
        print(f"current:  {current_dir}")
        print(
            f"common_metrics={len(common_keys)}  changed={len(rows)}  "
            f"added={len(added)}  removed={len(removed)}"
        )
        print("delta=current-baseline; lower is not always better for every metric")
        print()

        if rows:
            print_table(
                title=f"Top {min(top_n, len(rows))} metric deltas by absolute change",
                columns=[
                    ("section", "section"),
                    ("log", "log"),
                    ("item", "item"),
                    ("metric", "metric"),
                    ("before", "before"),
                    ("after", "after"),
                    ("delta", "delta"),
                    ("pct_delta", "delta_%"),
                ],
                rows=rows[:top_n],
                sort_key="log",
                reverse=True,
            )
        else:
            print("No metric deltas found.")

        if added:
            print_key_summary("Added metrics", added)
        if removed:
            print_key_summary("Removed metrics", removed)

    return buffer.getvalue()


def percent_delta(before: float, delta: float) -> float:
    if not np.isfinite(before) or abs(before) <= COMPARE_EPSILON:
        return float("nan")
    return 100.0 * delta / abs(before)


def print_key_summary(title: str, keys: list[MetricKey], limit: int = 12) -> None:
    print(title)
    for section, log_name, comparison, metric in keys[:limit]:
        item = f" {comparison}" if comparison else ""
        log = f" {log_name}" if log_name else ""
        print(f"  {section}{log}{item} {metric}")
    if len(keys) > limit:
        print(f"  ... {len(keys) - limit} more")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize pipeline cache durations and error statistics.")
    parser.add_argument(
        "logs",
        nargs="*",
        default=DEFAULT_LOGS,
        help="Log names to summarize. Defaults to the logs used by refine_mag_proj.py.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="Root containing pipeline cache folders.",
    )
    parser.add_argument(
        "--center-errors",
        action="store_true",
        help="Subtract each masked signal mean before computing error stats.",
    )
    parser.add_argument(
        "--error-threshold",
        type=float,
        help="Minimum absolute GT travel included in error calculations.",
    )
    parser.add_argument(
        "--sort-key",
        type=str,
        default="log",
        help="Column key or label used to sort each report table.",
    )
    parser.add_argument(
        "--deep-dive",
        action="store_true",
        help="Print stage and feature-conditioned diagnostics for the selected logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where report.txt, CSV tables, metrics.csv, and manifest.json will be written.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("BASELINE_DIR", "CURRENT_DIR"),
        help="Compare two saved stats output directories produced with --output-dir.",
    )
    parser.add_argument(
        "--compare-top",
        type=int,
        default=40,
        help="Number of metric deltas to show in --compare mode.",
    )
    parser.add_argument(
        "--compare-item",
        type=str,
        help="Filter --compare results to a specific item.",
    )
    parser.add_argument(
        "--compare-metric",
        type=str,
        help="Filter --compare results to a specific metric.",
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Run the pipeline for the specified logs"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.compare is not None:
        comparison_text = compare_saved_runs(args.compare[0], args.compare[1], top_n=args.compare_top, item=args.compare_item, compare_metric=args.compare_metric)
        print(comparison_text, end="")
        return
    
    if args.run_pipeline:
        for log_filename in args.logs:
            print(f"Running pipeline for {log_filename}...")
            os.system("venv/bin/python3 backend/pipeline.py " + log_filename)

    report = collect_report(
        args.logs,
        args.cache_root,
        center_errors=args.center_errors,
        error_threshold=args.error_threshold,
        include_diagnostics=args.deep_dive,
    )

    if not report.summary_rows:
        raise SystemExit("No logs could be summarized.")

    report_text = render_report(report, center_errors=args.center_errors, sort_key=args.sort_key)
    print(report_text, end="")

    if args.output_dir is not None:
        written = save_report(report, args.output_dir, report_text=report_text, args=args)
        print(f"Saved report artifacts to {args.output_dir} ({len(written)} files)")


if __name__ == "__main__":
    main()
