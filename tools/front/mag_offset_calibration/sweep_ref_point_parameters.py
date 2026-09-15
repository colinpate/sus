#!/usr/bin/env python3
"""Tune post-nuisance magnetic travel reference-point estimation.

The experiment deliberately removes the historical scalar offset embedded in
cached corrected-travel signals.  Every candidate then estimates one new
constant offset from nuisance-corrected magnitude and projected acceleration.
Ground-truth travel is used only for scoring and diagnostics.

The default split tunes on every usable ``front-default`` log and evaluates
the frozen finalists on the completely held-out ``slayer-filtered`` cohort.
Registry ``fixed_reference`` values are never used by candidate estimators.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from log_registry import DEFAULT_REGISTRY_PATH, LogRegistry  # noqa: E402
from mag_to_travel_model_core import MagToTravelModel  # noqa: E402
from travel_solver_core import (  # noqa: E402
    SolverInputs,
    solve_travel,
    solver_weights_for_mag_baseline,
)


DEFAULT_TUNING_SET = "front-default"
DEFAULT_VALIDATION_SET = "slayer-filtered"
SUBCOHORTS = (
    "harry",
    "jamaal",
    "stumpjumper-front-pod-v1",
    "stumpjumper-front-pod-v2",
)
TRAVEL_MAX_MM = 200.0
LOW_TRAVEL_MAX_MM = 30.0


@dataclass(frozen=True)
class LogData:
    split: str
    cohort: str
    subgroup: str
    log: str
    parent_log: str
    time_s: np.ndarray
    truth_mm: np.ndarray
    mag_mg: np.ndarray
    accel_m_s2: np.ndarray
    bad_mask: np.ndarray
    eval_mask: np.ndarray
    active_mask: np.ndarray
    base_corrected_mm: np.ndarray
    coefficients: np.ndarray
    mag_zv_points: np.ndarray
    cached_offset_mm: float
    fixed_reference: tuple[float, float] | None

    @property
    def fs_hz(self) -> float:
        return 1.0 / float(np.median(np.diff(self.time_s)))


@dataclass(frozen=True)
class RefChunk:
    direction: str
    mag_bump: np.ndarray
    rel_x_mm: np.ndarray
    truth_bump_mm: np.ndarray
    start_mag_mg: float
    start_truth_mm: float


@dataclass(frozen=True)
class Config:
    name: str
    family: str = "interaction"
    baseline_percentile: float = 5.0
    baseline_still_len_s: float = 0.1
    baseline_still_a_max_mm_s2: float = 1000.0
    bump_mag_min_mg: float = 1000.0
    still_a_max_mm_s2: float = 1000.0
    bump_dx_min_mm: float = 20.0
    still_len_s: float = 0.1
    bump_len_s: float = 0.3
    stride_s: float = 0.05
    skips: int = 3
    selector: str = "mag_band"
    ref_mag_range_mg: float = 2000.0
    min_ref_mag_mg: float = 2000.0
    x_min_mm: float = float("-inf")
    x_max_mm: float = float("inf")
    min_ref_points: int = 1
    fallback_percentile: float = 8.0
    fallback_accel_quantile: float = 70.0
    neg_fallback_max_fraction: float = 0.08
    high_fallback_max_fraction: float = 0.08
    max_offset_delta_mm: float = float("inf")


def flatten(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def subgroup_for_log(registry: LogRegistry, log_name: str) -> str:
    record_sets = set(registry.resolve(log_name).sets)
    return next((name for name in SUBCOHORTS if name in record_sets), "other-front")


def historical_offset(cache: np.lib.npyio.NpzFile) -> float:
    if "mag_model_offset_mm" in cache:
        return float(flatten(cache["mag_model_offset_mm"])[0])
    return float(
        np.median(
            flatten(cache["travel/mag_model/adj__x"])
            - flatten(cache["travel/mag_model__x"])
        )
    )


def load_log(
    registry: LogRegistry,
    cache_root: Path,
    log_name: str,
    *,
    split: str,
    cohort: str,
) -> LogData:
    resolved = registry.resolve(log_name)
    cache_path = cache_root / log_name / "cache" / "all.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        required = (
            "travel__t",
            "travel__x",
            "mag/nuisance/corrected/norm__x",
            "accel/lpfhp/proj__x",
            "mag/norm/bad_mask__x",
            "travel/mag_nuisance/corrected__x",
            "mag_model_coeffs",
            "mag_zv_points",
        )
        missing = [key for key in required if key not in cache]
        if missing:
            raise KeyError(f"{log_name}: missing cache keys {missing}")
        offset = historical_offset(cache)
        truth = flatten(cache["travel__x"])
        eval_key = "boring_mask" if "boring_mask" in cache else "active_mask"
        eval_mask = np.asarray(cache[eval_key], dtype=bool).reshape(-1)
        active_mask = np.asarray(cache["active_mask"], dtype=bool).reshape(-1)
        corrected = flatten(cache["travel/mag_nuisance/corrected__x"])
        data = {
            "time_s": flatten(cache["travel__t"]),
            "truth_mm": truth,
            "mag_mg": flatten(cache["mag/nuisance/corrected/norm__x"]),
            "accel_m_s2": flatten(cache["accel/lpfhp/proj__x"]),
            "bad_mask": np.asarray(
                cache["mag/norm/bad_mask__x"], dtype=bool
            ).reshape(-1),
            "eval_mask": eval_mask,
            "active_mask": active_mask,
            "base_corrected_mm": corrected - offset,
            "coefficients": flatten(cache["mag_model_coeffs"]),
            "mag_zv_points": np.asarray(cache["mag_zv_points"], dtype=int).reshape(-1),
        }
    lengths = {
        len(data["time_s"]),
        len(data["truth_mm"]),
        len(data["mag_mg"]),
        len(data["accel_m_s2"]),
        len(data["bad_mask"]),
        len(data["eval_mask"]),
        len(data["active_mask"]),
        len(data["base_corrected_mm"]),
    }
    if len(lengths) != 1:
        raise ValueError(f"{log_name}: unaligned cache lengths {sorted(lengths)}")
    if not np.any(data["eval_mask"]):
        raise ValueError(f"{log_name}: empty evaluation mask")
    step_config = (
        resolved.processing_config.get("steps", {}).get("get_mag_travel_ref_point", {})
    )
    fixed = step_config.get("fixed_reference")
    fixed_reference = None if fixed is None else (float(fixed[0]), float(fixed[1]))
    return LogData(
        split=split,
        cohort=cohort,
        subgroup=subgroup_for_log(registry, log_name),
        log=log_name,
        parent_log=str(resolved.metadata.get("parent_log", log_name)),
        cached_offset_mm=offset,
        fixed_reference=fixed_reference,
        **data,
    )


def resolve_logs(
    registry: LogRegistry,
    cache_root: Path,
    tuning_set: str,
    validation_set: str,
) -> list[LogData]:
    rows: list[LogData] = []
    seen: set[str] = set()
    for split, cohort in (("tuning", tuning_set), ("validation", validation_set)):
        selected = registry.select(set_name=cohort, usable_only=True)
        if not selected:
            raise ValueError(f"registry set {cohort!r} is empty")
        for log in selected:
            if log.pipeline != "front":
                raise ValueError(f"{log.log_id}: expected front pipeline")
            if log.log_id in seen:
                raise ValueError(f"{log.log_id}: appears in both experiment splits")
            seen.add(log.log_id)
            rows.append(
                load_log(
                    registry,
                    cache_root,
                    log.log_id,
                    split=split,
                    cohort=cohort,
                )
            )
    return rows


def estimate_baseline(data: LogData, config: Config) -> tuple[float, int]:
    window = max(1, int(config.baseline_still_len_s * data.fs_hz))
    accel_mm_s2 = np.abs(data.accel_m_s2) * 1000.0
    stationary: list[np.ndarray] = []
    for start in range(0, len(data.mag_mg) - window, window):
        stop = start + window
        if float(np.max(accel_mm_s2[start:stop])) < config.baseline_still_a_max_mm_s2:
            stationary.append(data.mag_mg[start:stop])
    finite_mag = data.mag_mg[np.isfinite(data.mag_mg)]
    if not stationary:
        return float(np.percentile(finite_mag, config.baseline_percentile)), 0
    values = np.concatenate(stationary)
    baseline = min(
        float(np.median(values)),
        float(np.percentile(finite_mag, config.baseline_percentile)),
    ) + float(np.std(values))
    return baseline, len(stationary)


def find_chunks(data: LogData, baseline_mg: float, config: Config) -> list[RefChunk]:
    mag = data.mag_mg
    accel_mm_s2 = data.accel_m_s2 * 1000.0
    truth = data.truth_mm
    dt_s = np.diff(data.time_s, prepend=data.time_s[0] - 0.01)
    still_len = max(1, int(config.still_len_s * data.fs_hz))
    bump_len = max(1, int(config.bump_len_s * data.fs_hz))
    stride = max(1, int(config.stride_s * data.fs_hz))
    chunk_len = still_len + bump_len
    chunks: list[RefChunk] = []
    skip = 0
    for index in range(0, len(mag) - chunk_len, stride):
        if skip > 0:
            skip -= 1
            continue
        for direction, chunk_slice in (
            ("forward", slice(index, index + chunk_len)),
            # Intentionally matches the production reverse slice.
            ("reverse", slice(index + chunk_len, index, -1)),
        ):
            mag_chunk = mag[chunk_slice]
            accel_chunk = accel_mm_s2[chunk_slice]
            truth_chunk = truth[chunk_slice]
            dt_chunk = dt_s[chunk_slice]
            mag_still = mag_chunk[:still_len]
            accel_still = accel_chunk[:still_len]
            mag_bump = mag_chunk[still_len:]
            accel_bump = accel_chunk[still_len:]
            truth_bump = truth_chunk[still_len:]
            dt_bump = dt_chunk[still_len:]
            if not (
                np.all(np.isfinite(mag_still))
                and np.all(np.isfinite(accel_still))
                and np.all(np.isfinite(mag_bump))
                and np.all(np.isfinite(accel_bump))
            ):
                continue
            start_mag = float(np.mean(mag_still))
            if start_mag > baseline_mg:
                continue
            if float(np.max(np.abs(accel_still))) > config.still_a_max_mm_s2:
                continue
            if float(np.max(mag_bump)) < start_mag + config.bump_mag_min_mg:
                continue
            velocity = np.cumsum(accel_bump * dt_bump)
            rel_x = np.cumsum(velocity * dt_bump)
            if float(np.max(rel_x)) < config.bump_dx_min_mm:
                continue
            skip = config.skips
            chunks.append(
                RefChunk(
                    direction=direction,
                    mag_bump=mag_bump,
                    rel_x_mm=rel_x,
                    truth_bump_mm=truth_bump,
                    start_mag_mg=start_mag,
                    start_truth_mm=float(np.median(truth_chunk[:still_len])),
                )
            )
    return chunks


def select_reference_points(
    chunks: Sequence[RefChunk], baseline_mg: float, config: Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not chunks:
        empty = np.empty(0, dtype=float)
        return empty, empty, empty
    mag = np.concatenate([chunk.mag_bump for chunk in chunks])
    rel_x = np.concatenate([chunk.rel_x_mm for chunk in chunks])
    truth = np.concatenate([chunk.truth_bump_mm for chunk in chunks])
    mask = np.isfinite(mag) & np.isfinite(rel_x) & np.isfinite(truth)
    if config.selector in {"mag_band", "mag_band_x"}:
        center = max(
            baseline_mg + config.min_ref_mag_mg,
            float(np.median(mag[mask])),
        )
        mask &= np.abs(mag - center) < config.ref_mag_range_mg / 2.0
    if config.selector in {"x", "mag_band_x"}:
        mask &= (rel_x > config.x_min_mm) & (rel_x < config.x_max_mm)
    if config.selector == "all":
        pass
    elif config.selector not in {"mag_band", "x", "mag_band_x"}:
        raise ValueError(f"unknown selector {config.selector!r}")
    return rel_x[mask], mag[mask], truth[mask]


def model_for(data: LogData) -> MagToTravelModel:
    return MagToTravelModel(pred_soft_mg=50.0, coeffs=data.coefficients)


def motion_mask(data: LogData, config: Config) -> np.ndarray:
    accel_abs = np.abs(data.accel_m_s2)
    candidates = np.isfinite(accel_abs) & ~data.bad_mask
    if not np.any(candidates):
        return np.zeros(len(accel_abs), dtype=bool)
    threshold = float(
        np.percentile(accel_abs[candidates], config.fallback_accel_quantile)
    )
    return candidates & (accel_abs > threshold)


def apply_fallback(
    data: LogData,
    config: Config,
    raw_offset_mm: float | None,
) -> tuple[float, bool, str]:
    model = model_for(data)
    finite_mag = data.mag_mg[np.isfinite(data.mag_mg)]
    zero_mag = float(np.percentile(finite_mag, config.fallback_percentile))
    zero_offset = -float(model.pred_x(zero_mag))
    if raw_offset_mm is None or not np.isfinite(raw_offset_mm):
        return zero_offset, True, "no_reference"
    reasons: list[str] = []
    if abs(raw_offset_mm - zero_offset) > config.max_offset_delta_mm:
        reasons.append("offset_delta")
    candidate = data.base_corrected_mm + raw_offset_mm
    mask = motion_mask(data, config) & np.isfinite(candidate)
    if np.any(mask):
        neg_fraction = float(np.mean(candidate[mask] < 0.0))
        high_fraction = float(np.mean(candidate[mask] > TRAVEL_MAX_MM))
        if neg_fraction > config.neg_fallback_max_fraction:
            reasons.append("negative")
        if high_fraction > config.high_fallback_max_fraction:
            reasons.append("high")
    if reasons:
        return zero_offset, True, "+".join(reasons)
    return raw_offset_mm, False, ""


def config_id(config: Config) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, allow_nan=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def evaluate_config(
    data: LogData,
    config: Config,
    chunk_cache: dict[tuple[str, tuple[object, ...]], tuple[float, int, list[RefChunk]]],
) -> dict[str, object]:
    detection_key = (
        config.baseline_percentile,
        config.baseline_still_len_s,
        config.baseline_still_a_max_mm_s2,
        config.bump_mag_min_mg,
        config.still_a_max_mm_s2,
        config.bump_dx_min_mm,
        config.still_len_s,
        config.bump_len_s,
        config.stride_s,
        config.skips,
    )
    cache_key = (data.log, detection_key)
    if cache_key not in chunk_cache:
        baseline, stationary_windows = estimate_baseline(data, config)
        chunks = find_chunks(data, baseline, config)
        chunk_cache[cache_key] = (baseline, stationary_windows, chunks)
    baseline, stationary_windows, chunks = chunk_cache[cache_key]
    rel_x, mag, truth_at_ref = select_reference_points(chunks, baseline, config)
    raw_offset: float | None = None
    reference_x = float("nan")
    reference_mag = float("nan")
    reference_truth = float("nan")
    if len(rel_x) >= config.min_ref_points:
        reference_x = float(np.median(rel_x))
        reference_mag = float(np.median(mag))
        reference_truth = float(np.median(truth_at_ref))
        raw_offset = reference_x - float(model_for(data).pred_x(reference_mag))
    offset, used_fallback, fallback_reason = apply_fallback(data, config, raw_offset)
    prediction = data.base_corrected_mm + offset
    mask = data.eval_mask & np.isfinite(prediction) & np.isfinite(data.truth_mm)
    low_mask = mask & (data.truth_mm >= 0.0) & (data.truth_mm < LOW_TRAVEL_MAX_MM)
    oracle_offset = float(
        np.mean(data.truth_mm[mask] - data.base_corrected_mm[mask])
    )
    error = prediction - data.truth_mm
    starts = np.asarray([chunk.start_truth_mm for chunk in chunks], dtype=float)
    return {
        "split": data.split,
        "cohort": data.cohort,
        "subgroup": data.subgroup,
        "log": data.log,
        "parent_log": data.parent_log,
        "config_id": config_id(config),
        "config": config.name,
        "family": config.family,
        "baseline_mg": baseline,
        "stationary_windows": stationary_windows,
        "n_chunks": len(chunks),
        "n_ref_points": len(rel_x),
        "calibration_found": raw_offset is not None,
        "used_fallback": used_fallback,
        "fallback_reason": fallback_reason,
        "near_zero_start_fraction": (
            float(np.mean(starts <= 3.0)) if len(starts) else float("nan")
        ),
        "median_start_truth_mm": (
            float(np.median(starts)) if len(starts) else float("nan")
        ),
        "reference_x_mm": reference_x,
        "reference_mag_mg": reference_mag,
        "reference_truth_mm": reference_truth,
        "reference_error_mm": reference_x - reference_truth,
        "raw_offset_mm": float("nan") if raw_offset is None else raw_offset,
        "offset_mm": offset,
        "oracle_offset_mm": oracle_offset,
        "offset_error_mm": offset - oracle_offset,
        "abs_offset_error_mm": abs(offset - oracle_offset),
        "mean_error_mm": float(np.mean(error[mask])),
        "uncentered_rmse_mm": rmse(error[mask]),
        "low_uncentered_rmse_mm": (
            rmse(error[low_mask]) if np.any(low_mask) else float("nan")
        ),
        "centered_rmse_mm": rmse(error[mask] - float(np.mean(error[mask]))),
    }


def aggregate_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    metric_names = (
        "n_chunks",
        "n_ref_points",
        "calibration_found",
        "used_fallback",
        "near_zero_start_fraction",
        "median_start_truth_mm",
        "reference_error_mm",
        "offset_error_mm",
        "abs_offset_error_mm",
        "mean_error_mm",
        "uncentered_rmse_mm",
        "low_uncentered_rmse_mm",
        "centered_rmse_mm",
    )
    output: list[dict[str, object]] = []
    for split in sorted({str(row["split"]) for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        for config_name in sorted({str(row["config"]) for row in split_rows}):
            config_rows = [row for row in split_rows if row["config"] == config_name]
            if not config_rows:
                continue
            groups = sorted({str(row["subgroup"]) for row in config_rows})
            group_summaries: list[dict[str, object]] = []
            for group in groups:
                selected = [row for row in config_rows if row["subgroup"] == group]
                def safe_mean(metric: str) -> float:
                    values = np.asarray(
                        [float(row[metric]) for row in selected], dtype=float
                    )
                    finite = values[np.isfinite(values)]
                    return float(np.mean(finite)) if len(finite) else float("nan")

                summary = {
                    "split": split,
                    "scope": group,
                    "config": config_name,
                    "config_id": selected[0]["config_id"],
                    "family": selected[0]["family"],
                    "n_logs": len(selected),
                    **{
                        metric: safe_mean(metric)
                        for metric in metric_names
                    },
                }
                group_summaries.append(summary)
                output.append(summary)
            output.append(
                {
                    "split": split,
                    "scope": "cohort-balanced",
                    "config": config_name,
                    "config_id": config_rows[0]["config_id"],
                    "family": config_rows[0]["family"],
                    "n_logs": len(config_rows),
                    **{
                        metric: float(
                            np.mean(
                                [
                                    float(row[metric])
                                    for row in group_summaries
                                    if np.isfinite(float(row[metric]))
                                ]
                            )
                        )
                        if any(
                            np.isfinite(float(row[metric])) for row in group_summaries
                        )
                        else float("nan")
                        for metric in metric_names
                    },
                }
            )
    return output


def screen_configs() -> list[Config]:
    base = Config(name="current_generic", family="current")
    configs = [base]

    def add_family(field: str, values: Iterable[object]) -> None:
        for value in values:
            label = str(value).replace(".", "p").replace("-", "m")
            configs.append(
                replace(base, name=f"{field}_{label}", family=field, **{field: value})
            )

    add_family("baseline_percentile", (0.5, 1.0, 2.0, 5.0, 8.0, 10.0))
    add_family("baseline_still_len_s", (0.05, 0.1, 0.2, 0.3))
    add_family("baseline_still_a_max_mm_s2", (500.0, 1000.0, 2000.0, 5000.0))
    add_family("bump_mag_min_mg", (50.0, 100.0, 200.0, 300.0, 500.0, 750.0, 1000.0, 1500.0))
    add_family("still_a_max_mm_s2", (250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0))
    add_family("bump_dx_min_mm", (5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0))
    add_family("still_len_s", (0.05, 0.1, 0.2, 0.3))
    add_family("bump_len_s", (0.2, 0.3, 0.4, 0.5, 0.75, 1.0))
    add_family("stride_s", (0.02, 0.05, 0.1))
    add_family("skips", (0, 1, 3, 5, 10))
    add_family("ref_mag_range_mg", (250.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0))
    add_family("min_ref_mag_mg", (0.0, 100.0, 250.0, 500.0, 1000.0, 1500.0, 2000.0))
    add_family("min_ref_points", (1, 5, 10, 20, 40, 80))
    add_family("fallback_percentile", (0.5, 1.0, 2.0, 5.0, 8.0, 10.0, 15.0))
    add_family("fallback_accel_quantile", (40.0, 55.0, 70.0, 85.0, 95.0))
    add_family("neg_fallback_max_fraction", (0.02, 0.05, 0.08, 0.15, 0.3, 1.0))
    add_family("max_offset_delta_mm", (10.0, 20.0, 40.0, 80.0, float("inf")))

    configs.append(replace(base, name="selector_all", family="selector", selector="all"))
    for threshold in (0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 60.0):
        configs.append(
            replace(
                base,
                name=f"selector_x_gt_{threshold:g}",
                family="selector",
                selector="x",
                x_min_mm=threshold,
            )
        )
        configs.append(
            replace(
                base,
                name=f"selector_mag_and_x_gt_{threshold:g}",
                family="selector",
                selector="mag_band_x",
                x_min_mm=threshold,
            )
        )
    for low, high in ((0, 20), (10, 30), (20, 40), (40, 60), (60, 80), (80, 120), (120, 180)):
        configs.append(
            replace(
                base,
                name=f"selector_x_{low}_{high}",
                family="selector",
                selector="x",
                x_min_mm=float(low),
                x_max_mm=float(high),
            )
        )
    dedup: dict[str, Config] = {}
    for config in configs:
        dedup[config_id(config)] = config
    return list(dedup.values())


def recovery_configs() -> list[Config]:
    """Predeclared recovery hypotheses, not chosen from validation outcomes."""
    base = Config(name="current_generic", family="current")
    return [
        base,
        replace(
            base,
            name="production_safety_without_fixed_ref",
            family="safety",
            min_ref_points=20,
            max_offset_delta_mm=40.0,
        ),
        replace(
            base,
            name="recovery_low_bump_200",
            family="recovery",
            bump_mag_min_mg=200.0,
        ),
        replace(
            base,
            name="recovery_relaxed_still_1500",
            family="recovery",
            still_a_max_mm_s2=1500.0,
        ),
        replace(
            base,
            name="recovery_low_bump_relaxed_still",
            family="recovery",
            bump_mag_min_mg=200.0,
            still_a_max_mm_s2=1500.0,
        ),
        replace(
            base,
            name="recovery_short_low_bump_relaxed_still",
            family="recovery",
            bump_len_s=0.2,
            bump_mag_min_mg=200.0,
            still_a_max_mm_s2=1500.0,
        ),
        replace(
            base,
            name="recovery_guarded",
            family="recovery",
            bump_len_s=0.2,
            bump_mag_min_mg=200.0,
            still_a_max_mm_s2=1500.0,
            min_ref_points=10,
            max_offset_delta_mm=40.0,
        ),
    ]


def rejection_funnel(data: LogData, config: Config) -> dict[str, object]:
    """Count sequential gate passes for diagnosing missing calibration chunks."""
    baseline, stationary_windows = estimate_baseline(data, config)
    mag = data.mag_mg
    accel = data.accel_m_s2 * 1000.0
    dt_s = np.diff(data.time_s, prepend=data.time_s[0] - 0.01)
    still_len = max(1, int(config.still_len_s * data.fs_hz))
    bump_len = max(1, int(config.bump_len_s * data.fs_hz))
    stride = max(1, int(config.stride_s * data.fs_hz))
    chunk_len = still_len + bump_len
    counts = {
        "oriented_windows": 0,
        "baseline_pass": 0,
        "still_accel_pass": 0,
        "bump_mag_pass": 0,
        "bump_dx_pass": 0,
    }
    for index in range(0, len(mag) - chunk_len, stride):
        for chunk_slice in (
            slice(index, index + chunk_len),
            slice(index + chunk_len, index, -1),
        ):
            counts["oriented_windows"] += 1
            mag_chunk = mag[chunk_slice]
            accel_chunk = accel[chunk_slice]
            dt_chunk = dt_s[chunk_slice]
            mag_still = mag_chunk[:still_len]
            accel_still = accel_chunk[:still_len]
            mag_bump = mag_chunk[still_len:]
            accel_bump = accel_chunk[still_len:]
            dt_bump = dt_chunk[still_len:]
            if not (
                np.all(np.isfinite(mag_still))
                and np.all(np.isfinite(accel_still))
                and np.all(np.isfinite(mag_bump))
                and np.all(np.isfinite(accel_bump))
            ):
                continue
            start_mag = float(np.mean(mag_still))
            if start_mag > baseline:
                continue
            counts["baseline_pass"] += 1
            if float(np.max(np.abs(accel_still))) > config.still_a_max_mm_s2:
                continue
            counts["still_accel_pass"] += 1
            if float(np.max(mag_bump)) < start_mag + config.bump_mag_min_mg:
                continue
            counts["bump_mag_pass"] += 1
            velocity = np.cumsum(accel_bump * dt_bump)
            rel_x = np.cumsum(velocity * dt_bump)
            if float(np.max(rel_x)) < config.bump_dx_min_mm:
                continue
            counts["bump_dx_pass"] += 1
    return {
        "split": data.split,
        "cohort": data.cohort,
        "subgroup": data.subgroup,
        "log": data.log,
        "parent_log": data.parent_log,
        "config": config.name,
        "baseline_mg": baseline,
        "stationary_windows": stationary_windows,
        **counts,
    }


def best_levels(
    aggregate: Sequence[dict[str, object]],
    configs: Sequence[Config],
    family: str,
    field: str,
    count: int = 2,
) -> list[object]:
    config_lookup = {config.name: config for config in configs}
    rows = [
        row
        for row in aggregate
        if row["split"] == "tuning"
        and row["scope"] == "cohort-balanced"
        and row["family"] == family
    ]
    rows.sort(
        key=lambda row: (
            float(row["uncentered_rmse_mm"]),
            float(row["abs_offset_error_mm"]),
            float(row["used_fallback"]),
        )
    )
    values: list[object] = []
    for row in rows:
        value = getattr(config_lookup[str(row["config"])], field)
        if value not in values:
            values.append(value)
        if len(values) == count:
            break
    return values


def interaction_configs(
    screen_aggregate: Sequence[dict[str, object]], screen: Sequence[Config]
) -> list[Config]:
    fields = (
        ("baseline_percentile", "baseline_percentile"),
        ("bump_mag_min_mg", "bump_mag_min_mg"),
        ("still_a_max_mm_s2", "still_a_max_mm_s2"),
        ("bump_dx_min_mm", "bump_dx_min_mm"),
        ("bump_len_s", "bump_len_s"),
    )
    levels = {
        field: best_levels(screen_aggregate, screen, family, field)
        for family, field in fields
    }
    selector_rows = [
        row
        for row in screen_aggregate
        if row["split"] == "tuning"
        and row["scope"] == "cohort-balanced"
        and row["family"] == "selector"
    ]
    selector_rows.sort(key=lambda row: float(row["uncentered_rmse_mm"]))
    lookup = {config.name: config for config in screen}
    selectors = [lookup[str(row["config"])] for row in selector_rows[:4]]
    configs: list[Config] = []
    for values in itertools.product(*(levels[field] for _, field in fields)):
        detection = dict(zip((field for _, field in fields), values))
        for selector_config in selectors:
            config = replace(
                selector_config,
                family="interaction",
                **detection,
            )
            label = config_id(config)
            configs.append(replace(config, name=f"interaction_{label}"))
    return configs


def oracle_anchor_rows(data_rows: Sequence[LogData]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for data in data_rows:
        model = model_for(data)
        mask = data.eval_mask & ~data.bad_mask & np.isfinite(data.truth_mm) & np.isfinite(data.mag_mg)
        oracle_offset = float(
            np.mean(data.truth_mm[mask] - data.base_corrected_mm[mask])
        )
        for target in (5.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0):
            band = mask & (np.abs(data.truth_mm - target) <= 2.5)
            if not np.any(band):
                continue
            ref_x = float(np.median(data.truth_mm[band]))
            ref_mag = float(np.median(data.mag_mg[band]))
            offset = ref_x - float(model.pred_x(ref_mag))
            prediction = data.base_corrected_mm + offset
            error = prediction - data.truth_mm
            derivative = float(
                abs(
                    model.pred_x(ref_mag + 0.5)
                    - model.pred_x(ref_mag - 0.5)
                )
            )
            output.append(
                {
                    "split": data.split,
                    "cohort": data.cohort,
                    "subgroup": data.subgroup,
                    "log": data.log,
                    "target_travel_mm": target,
                    "n_anchor_samples": int(np.sum(band)),
                    "reference_x_mm": ref_x,
                    "reference_mag_mg": ref_mag,
                    "local_sensitivity_mm_per_mg": derivative,
                    "offset_mm": offset,
                    "offset_error_mm": offset - oracle_offset,
                    "abs_offset_error_mm": abs(offset - oracle_offset),
                    "uncentered_rmse_mm": rmse(error[mask]),
                    "mean_error_mm": float(np.mean(error[mask])),
                }
            )
    return output


def aggregate_oracle_anchors(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    metrics = (
        "n_anchor_samples",
        "local_sensitivity_mm_per_mg",
        "offset_error_mm",
        "abs_offset_error_mm",
        "uncentered_rmse_mm",
        "mean_error_mm",
    )
    for split in sorted({str(row["split"]) for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        for target in sorted({float(row["target_travel_mm"]) for row in split_rows}):
            selected = [row for row in split_rows if row["target_travel_mm"] == target]
            output.append(
                {
                    "split": split,
                    "target_travel_mm": target,
                    "n_logs": len(selected),
                    **{
                        metric: float(np.mean([float(row[metric]) for row in selected]))
                        for metric in metrics
                    },
                }
            )
    return output


def select_finalists(
    aggregate: Sequence[dict[str, object]],
    config_lookup: dict[str, Config],
    count: int,
) -> list[Config]:
    rows = [
        row
        for row in aggregate
        if row["split"] == "tuning" and row["scope"] == "cohort-balanced"
    ]
    rows.sort(
        key=lambda row: (
            float(row["uncentered_rmse_mm"]),
            float(row["low_uncentered_rmse_mm"]),
            float(row["abs_offset_error_mm"]),
        )
    )
    selected: list[Config] = []
    seen: set[str] = set()
    for row in rows:
        name = str(row["config"])
        if name in seen:
            continue
        seen.add(name)
        selected.append(config_lookup[name])
        if len(selected) == count:
            break
    return selected


def fixed_reference_row(data: LogData) -> dict[str, object] | None:
    if data.fixed_reference is None:
        return None
    ref_x, ref_mag = data.fixed_reference
    offset = ref_x - float(model_for(data).pred_x(ref_mag))
    prediction = data.base_corrected_mm + offset
    mask = data.eval_mask & np.isfinite(prediction) & np.isfinite(data.truth_mm)
    low = mask & (data.truth_mm >= 0.0) & (data.truth_mm < LOW_TRAVEL_MAX_MM)
    oracle_offset = float(np.mean(data.truth_mm[mask] - data.base_corrected_mm[mask]))
    error = prediction - data.truth_mm
    return {
        "split": data.split,
        "cohort": data.cohort,
        "subgroup": data.subgroup,
        "log": data.log,
        "parent_log": data.parent_log,
        "config_id": "registry",
        "config": "registry_fixed_reference",
        "family": "registry",
        "baseline_mg": float("nan"),
        "stationary_windows": float("nan"),
        "n_chunks": float("nan"),
        "n_ref_points": float("nan"),
        "calibration_found": True,
        "used_fallback": False,
        "fallback_reason": "",
        "near_zero_start_fraction": float("nan"),
        "median_start_truth_mm": float("nan"),
        "reference_x_mm": ref_x,
        "reference_mag_mg": ref_mag,
        "reference_truth_mm": float("nan"),
        "reference_error_mm": float("nan"),
        "raw_offset_mm": offset,
        "offset_mm": offset,
        "oracle_offset_mm": oracle_offset,
        "offset_error_mm": offset - oracle_offset,
        "abs_offset_error_mm": abs(offset - oracle_offset),
        "mean_error_mm": float(np.mean(error[mask])),
        "uncentered_rmse_mm": rmse(error[mask]),
        "low_uncentered_rmse_mm": rmse(error[low]) if np.any(low) else float("nan"),
        "centered_rmse_mm": rmse(error[mask] - float(np.mean(error[mask]))),
    }


def replay_solver(
    data_rows: Sequence[LogData],
    candidate_rows: Sequence[dict[str, object]],
    configs: Sequence[Config],
    *,
    max_nfev: int,
) -> list[dict[str, object]]:
    lookup = {
        (str(row["log"]), str(row["config"])): row for row in candidate_rows
    }
    output: list[dict[str, object]] = []
    for data in data_rows:
        baseline, _ = estimate_baseline(data, Config(name="solver_baseline"))
        weights = solver_weights_for_mag_baseline(baseline)
        ran_solver = False
        for config in configs:
            row = lookup.get((data.log, config.name))
            if row is None:
                continue
            ran_solver = True
            prediction = data.base_corrected_mm + float(row["offset_mm"])
            inputs = SolverInputs(
                time_s=data.time_s,
                accel_mm_s2=data.accel_m_s2 * 1000.0,
                mag=data.mag_mg,
                mag_preds_mm=prediction,
                mag_zv_points=data.mag_zv_points,
                mag_baseline=baseline,
            )
            result = solve_travel(inputs, weights, max_nfev=max_nfev)
            solved = result.x
            mask = data.eval_mask & np.isfinite(solved) & np.isfinite(data.truth_mm)
            low = mask & (data.truth_mm >= 0.0) & (data.truth_mm < LOW_TRAVEL_MAX_MM)
            error = solved - data.truth_mm
            output.append(
                {
                    "split": data.split,
                    "cohort": data.cohort,
                    "subgroup": data.subgroup,
                    "log": data.log,
                    "parent_log": data.parent_log,
                    "config": config.name,
                    "offset_mm": float(row["offset_mm"]),
                    "mean_error_mm": float(np.mean(error[mask])),
                    "uncentered_rmse_mm": rmse(error[mask]),
                    "low_uncentered_rmse_mm": (
                        rmse(error[low]) if np.any(low) else float("nan")
                    ),
                    "centered_rmse_mm": rmse(
                        error[mask] - float(np.mean(error[mask]))
                    ),
                    "success": bool(result.scipy_result.success),
                    "nfev": int(result.scipy_result.nfev),
                }
            )
        if ran_solver:
            print(f"[solver] {data.split}/{data.log}", flush=True)
    return output


def aggregate_solver_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    metrics = (
        "mean_error_mm",
        "uncentered_rmse_mm",
        "low_uncentered_rmse_mm",
        "centered_rmse_mm",
        "success",
        "nfev",
    )

    def metric_value(row: dict[str, object], metric: str) -> float:
        raw = row[metric]
        if metric == "success" and isinstance(raw, str):
            return float(raw.strip().lower() == "true")
        return float(raw)

    for split in sorted({str(row["split"]) for row in rows}):
        for config in sorted({str(row["config"]) for row in rows if row["split"] == split}):
            selected = [
                row for row in rows if row["split"] == split and row["config"] == config
            ]
            groups = sorted({str(row["subgroup"]) for row in selected})
            group_rows: list[dict[str, object]] = []
            for group in groups:
                group_selected = [row for row in selected if row["subgroup"] == group]
                summary = {
                    "split": split,
                    "scope": group,
                    "config": config,
                    "n_logs": len(group_selected),
                    **{
                        metric: float(
                            np.mean([metric_value(row, metric) for row in group_selected])
                        )
                        for metric in metrics
                    },
                }
                group_rows.append(summary)
                output.append(summary)
            output.append(
                {
                    "split": split,
                    "scope": "cohort-balanced",
                    "config": config,
                    "n_logs": len(selected),
                    **{
                        metric: float(
                            np.mean([metric_value(row, metric) for row in group_rows])
                        )
                        for metric in metrics
                    },
                }
            )
    return output


def aggregate_validation_by_parent(
    adjusted_rows: Sequence[dict[str, object]],
    solver_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Give each held-out parent log equal weight despite derived chunks."""
    output: list[dict[str, object]] = []
    metrics = (
        "mean_error_mm",
        "uncentered_rmse_mm",
        "low_uncentered_rmse_mm",
        "centered_rmse_mm",
    )
    for stage, rows in (("adjusted", adjusted_rows), ("final_solver", solver_rows)):
        validation = [row for row in rows if row["split"] == "validation"]
        for config in sorted({str(row["config"]) for row in validation}):
            selected = [row for row in validation if row["config"] == config]
            parents = sorted({str(row["parent_log"]) for row in selected})
            parent_metrics = []
            for parent in parents:
                parent_rows = [row for row in selected if row["parent_log"] == parent]
                parent_metrics.append(
                    {
                        metric: float(
                            np.mean([float(row[metric]) for row in parent_rows])
                        )
                        for metric in metrics
                    }
                )
            output.append(
                {
                    "stage": stage,
                    "config": config,
                    "n_logs": len(selected),
                    "n_parents": len(parents),
                    **{
                        metric: float(
                            np.mean([row[metric] for row in parent_metrics])
                        )
                        for metric in metrics
                    },
                }
            )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--cache-root", type=Path, default=REPO_ROOT / "backend" / "run_artifacts"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tuning-set", default=DEFAULT_TUNING_SET)
    parser.add_argument("--validation-set", default=DEFAULT_VALIDATION_SET)
    parser.add_argument("--finalists", type=int, default=8)
    parser.add_argument("--replay-solver", action="store_true")
    parser.add_argument(
        "--solver-only",
        action="store_true",
        help="Reuse an existing finalist sweep and only replay the final solver",
    )
    parser.add_argument("--max-nfev", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = LogRegistry.load(args.registry)
    data_rows = resolve_logs(
        registry,
        args.cache_root,
        args.tuning_set,
        args.validation_set,
    )
    tuning = [row for row in data_rows if row.split == "tuning"]
    validation = [row for row in data_rows if row.split == "validation"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.solver_only:
        selection = json.loads((args.output_dir / "selection.json").read_text())
        finalists = selection["finalists"]
        selected = Config(**finalists[selection["selected_config"]])
        current = Config(**finalists["current_generic"])
        solver_configs = [selected, current]
        candidate_rows = read_csv(args.output_dir / "finalist_per_log.csv")
        if any(row["config"] == "registry_fixed_reference" for row in candidate_rows):
            solver_configs.append(
                Config(name="registry_fixed_reference", family="registry")
            )
        solver_path = args.output_dir / "solver_per_log.csv"
        existing_rows = read_csv(solver_path) if solver_path.exists() else []
        existing_configs = {str(row["config"]) for row in existing_rows}
        missing_configs = [
            config for config in solver_configs if config.name not in existing_configs
        ]
        new_rows = replay_solver(
            data_rows,
            candidate_rows,
            missing_configs,
            max_nfev=args.max_nfev,
        )
        solver_rows = [*existing_rows, *new_rows]
        write_csv(args.output_dir / "solver_per_log.csv", solver_rows)
        write_csv(
            args.output_dir / "solver_aggregate.csv",
            aggregate_solver_rows(solver_rows),
        )
        write_csv(
            args.output_dir / "validation_parent_aggregate.csv",
            aggregate_validation_by_parent(candidate_rows, solver_rows),
        )
        manifest_path = args.output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
        manifest["solver_replayed"] = True
        manifest["solver_row_count"] = len(solver_rows)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return

    anchor_rows = oracle_anchor_rows(data_rows)
    write_csv(args.output_dir / "oracle_anchor_travel_per_log.csv", anchor_rows)
    write_csv(
        args.output_dir / "oracle_anchor_travel_aggregate.csv",
        aggregate_oracle_anchors(anchor_rows),
    )

    chunk_cache: dict[
        tuple[str, tuple[object, ...]], tuple[float, int, list[RefChunk]]
    ] = {}
    screen = screen_configs()
    screen_rows: list[dict[str, object]] = []
    for index, config in enumerate(screen, 1):
        for data in tuning:
            screen_rows.append(evaluate_config(data, config, chunk_cache))
        print(f"[screen {index}/{len(screen)}] {config.name}", flush=True)
    screen_aggregate = aggregate_rows(screen_rows)
    write_csv(args.output_dir / "screen_per_log.csv", screen_rows)
    write_csv(args.output_dir / "screen_aggregate.csv", screen_aggregate)

    interactions = interaction_configs(screen_aggregate, screen)
    interaction_rows: list[dict[str, object]] = []
    for index, config in enumerate(interactions, 1):
        for data in tuning:
            interaction_rows.append(evaluate_config(data, config, chunk_cache))
        print(f"[interaction {index}/{len(interactions)}] {config.name}", flush=True)
    interaction_aggregate = aggregate_rows(interaction_rows)
    write_csv(args.output_dir / "interaction_per_log.csv", interaction_rows)
    write_csv(args.output_dir / "interaction_aggregate.csv", interaction_aggregate)

    all_configs = {config.name: config for config in (*screen, *interactions)}
    all_aggregate = [*screen_aggregate, *interaction_aggregate]
    finalists = select_finalists(
        all_aggregate,
        all_configs,
        args.finalists,
    )
    # Preserve interpretable anchors even if closely related interactions rank higher.
    for name in ("current_generic", "selector_all", "selector_x_gt_20"):
        candidate = all_configs[name]
        if candidate not in finalists:
            finalists.append(candidate)

    validation_rows: list[dict[str, object]] = []
    for config in finalists:
        for data in validation:
            validation_rows.append(evaluate_config(data, config, chunk_cache))
    registry_rows = [row for data in validation if (row := fixed_reference_row(data))]
    finalist_tuning_rows = [
        row
        for row in (*screen_rows, *interaction_rows)
        if str(row["config"]) in {config.name for config in finalists}
    ]
    finalist_rows = [*finalist_tuning_rows, *validation_rows, *registry_rows]
    finalist_aggregate = aggregate_rows(finalist_rows)
    write_csv(args.output_dir / "finalist_per_log.csv", finalist_rows)
    write_csv(args.output_dir / "finalist_aggregate.csv", finalist_aggregate)

    recovery = recovery_configs()
    recovery_rows = [
        evaluate_config(data, config, chunk_cache)
        for config in recovery
        for data in data_rows
    ]
    write_csv(args.output_dir / "recovery_per_log.csv", recovery_rows)
    write_csv(args.output_dir / "recovery_aggregate.csv", aggregate_rows(recovery_rows))
    funnel_rows = [
        rejection_funnel(data, config)
        for config in recovery
        for data in data_rows
    ]
    write_csv(args.output_dir / "rejection_funnel.csv", funnel_rows)

    config_payload = {config.name: asdict(config) for config in finalists}
    selected = min(
        (
            row
            for row in finalist_aggregate
            if row["split"] == "tuning" and row["scope"] == "cohort-balanced"
        ),
        key=lambda row: (
            float(row["uncentered_rmse_mm"]),
            float(row["low_uncentered_rmse_mm"]),
        ),
    )
    selection = {
        "tuning_set": args.tuning_set,
        "validation_set": args.validation_set,
        "selection_metric": "cohort-balanced adjusted-travel uncentered RMSE",
        "selected_config": selected["config"],
        "selected_tuning_summary": selected,
        "finalists": config_payload,
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    # This broad validation screen runs only after the primary winner is frozen.
    # It is a non-transfer diagnostic, not part of model selection.
    diagnostic_validation_rows = [
        evaluate_config(data, config, chunk_cache)
        for config in screen
        for data in validation
    ]
    write_csv(
        args.output_dir / "diagnostic_validation_screen_per_log.csv",
        diagnostic_validation_rows,
    )
    write_csv(
        args.output_dir / "diagnostic_validation_screen_aggregate.csv",
        aggregate_rows(diagnostic_validation_rows),
    )

    solver_rows: list[dict[str, object]] = []
    if args.replay_solver:
        solver_configs = [all_configs[str(selected["config"])]]
        current = all_configs["current_generic"]
        if current not in solver_configs:
            solver_configs.append(current)
        if registry_rows:
            solver_configs.append(
                Config(name="registry_fixed_reference", family="registry")
            )
        solver_rows = replay_solver(
            data_rows,
            finalist_rows,
            solver_configs,
            max_nfev=args.max_nfev,
        )
        write_csv(args.output_dir / "solver_per_log.csv", solver_rows)
        write_csv(args.output_dir / "solver_aggregate.csv", aggregate_solver_rows(solver_rows))
    write_csv(
        args.output_dir / "validation_parent_aggregate.csv",
        aggregate_validation_by_parent(finalist_rows, solver_rows),
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "registry": str(args.registry),
        "cache_root": str(args.cache_root),
        "tuning_set": args.tuning_set,
        "validation_set": args.validation_set,
        "tuning_logs": [row.log for row in tuning],
        "validation_logs": [row.log for row in validation],
        "validation_parent_logs": sorted({row.parent_log for row in validation}),
        "screen_config_count": len(screen),
        "interaction_config_count": len(interactions),
        "post_selection_validation_diagnostic_config_count": len(screen),
        "solver_replayed": bool(args.replay_solver),
        "solver_row_count": len(solver_rows),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selection, indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
