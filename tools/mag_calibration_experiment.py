#!/usr/bin/env python3
"""Train and evaluate front/rear mag-to-travel calibrations on log windows."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable

os.environ["MPLCONFIGDIR"] = "/private/tmp"

import numpy as np
import scipy.optimize
from sklearn.isotonic import IsotonicRegression


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mag_calibration import (  # noqa: E402
    MagTravelCalibration,
    RecordingWindow,
    ResolvedWindow,
    TimeRange,
    resolve_window,
    sample_durations,
)
from mag_to_travel_model_core import MagToTravelModel, MagToTravelModelCore  # noqa: E402
from rear_mag_model import RearMagModel  # noqa: E402


FRONT_FEATURE = "mag/norm/corr/lpf"
REAR_FEATURE = "mag/angle/lpf"
TRAVEL_BIN_EDGES = np.linspace(0.0, 150.0, 6)
MIN_BIN_SAMPLES = 100


@dataclass
class CachedCalibrationData:
    log_name: str
    pipeline: str
    time_s: np.ndarray
    mag: np.ndarray
    accel: np.ndarray
    zv_points: np.ndarray
    travel: np.ndarray
    activity_mask: np.ndarray
    bad_mask: np.ndarray
    feature_key: str
    source_fingerprint: str | None
    mag_baseline: float | None = None
    mag_ref_point: np.ndarray | None = None


def flatten_1d(value: np.ndarray, *, dtype: Any = float) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]
    return array.reshape(-1)


def cached_pipeline_kind(cache: np.lib.npyio.NpzFile, log_name: str) -> str:
    has_front = f"{FRONT_FEATURE}__x" in cache
    has_rear = f"{REAR_FEATURE}__x" in cache
    if has_front == has_rear:
        raise ValueError(
            f"Cannot identify front/rear calibration inputs in cache for {log_name!r}"
        )
    return "front" if has_front else "rear"


def load_cached_log(log_name: str) -> CachedCalibrationData:
    path = REPO_ROOT / "backend" / "run_artifacts" / log_name / "cache" / "all.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing pipeline cache for {log_name!r}: {path}")

    with np.load(path, allow_pickle=False) as cache:
        pipeline = cached_pipeline_kind(cache, log_name)
        if pipeline == "front":
            feature_key = FRONT_FEATURE
            accel_key = "accel/lpfhp/proj"
            bad_key = "mag/norm/bad_mask__x"
            mag_baseline = float(flatten_1d(cache["mag_baseline"])[0])
            mag_ref_point = flatten_1d(cache["mag_travel_ref_point"])
        else:
            feature_key = REAR_FEATURE
            accel_key = "accel/lphp/proj/zv"
            bad_key = ""
            mag_baseline = None
            mag_ref_point = None

        time_s = flatten_1d(cache[f"{feature_key}__t"])
        mag = flatten_1d(cache[f"{feature_key}__x"])
        accel = flatten_1d(cache[f"{accel_key}__x"])
        travel = flatten_1d(cache["travel__x"])
        activity_mask = flatten_1d(cache["boring_mask"], dtype=bool)
        bad_mask = (
            flatten_1d(cache[bad_key], dtype=bool)
            if bad_key and bad_key in cache
            else np.zeros(len(time_s), dtype=bool)
        )
        zv_key = "mag_zv_points/accel_corr" if pipeline == "rear" else "mag_zv_points"
        zv_points = flatten_1d(cache[zv_key], dtype=int)
        source_fingerprint = (
            str(cache["__run_fingerprint"].item())
            if "__run_fingerprint" in cache
            else None
        )

    lengths = {
        "time": len(time_s),
        "mag": len(mag),
        "accel": len(accel),
        "travel": len(travel),
        "activity": len(activity_mask),
        "bad_mask": len(bad_mask),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Cache arrays are not aligned for {log_name!r}: {lengths}")

    return CachedCalibrationData(
        log_name=log_name,
        pipeline=pipeline,
        time_s=time_s,
        mag=mag,
        accel=accel,
        zv_points=zv_points,
        travel=travel,
        activity_mask=activity_mask,
        bad_mask=bad_mask,
        feature_key=feature_key,
        source_fingerprint=source_fingerprint,
        mag_baseline=mag_baseline,
        mag_ref_point=mag_ref_point,
    )


@contextlib.contextmanager
def fit_output(verbose: bool):
    if verbose:
        yield
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            yield


def model_config(model: MagToTravelModelCore) -> dict[str, Any]:
    names = (
        "chunk_min_dx",
        "chunk_max_dx",
        "chunk_rad",
        "train_with_mask",
        "bad_thresh",
        "dm_dx_thresh",
        "pred_soft_mg",
        "power_weight",
        "x0_weight",
        "chunking_method",
    )
    config = {name: getattr(model, name) for name in names}
    if isinstance(model, RearMagModel):
        for name in ("min_chunk_dt", "max_chunk_dt", "min_chunk_db", "pair_mode"):
            config[name] = getattr(model, name)
    return config


def fit_calibration(
    data: CachedCalibrationData,
    window: RecordingWindow,
    *,
    trainer: str = "self-supervised",
    oracle_bins: int = 100,
    verbose: bool = False,
) -> tuple[MagTravelCalibration, ResolvedWindow]:
    if data.log_name != window.log_name:
        raise ValueError(f"Loaded {data.log_name!r} but received window for {window.log_name!r}")
    resolved = resolve_window(
        data.time_s,
        window.time_range,
        time_basis=window.time_basis,
        activity_mask=data.activity_mask,
    )

    if trainer != "self-supervised":
        return fit_oracle_calibration(
            data,
            window,
            resolved,
            trainer=trainer,
            oracle_bins=oracle_bins,
        ), resolved

    if data.pipeline == "front":
        model: MagToTravelModelCore = MagToTravelModelCore(train_with_mask=False)
        with fit_output(verbose):
            training_data = model.create_training_data(
                mag=data.mag,
                accel=data.accel,
                train_mask=data.bad_mask,
                t=data.time_s,
                baseline_min_mag=float(data.mag_baseline),
                idxs=data.zv_points,
                sample_range=resolved.sample_range,
            )
            result = model.train(training_data)
    else:
        model = RearMagModel()
        with fit_output(verbose):
            training_data = model.create_training_data(
                mag=data.mag,
                accel=data.accel,
                t=data.time_s,
                idxs=data.zv_points,
                sample_range=resolved.sample_range,
            )
            result = model.train(training_data, guess_vec=[0.1, 250.0, 1.0 / 3.0])

    calibration = MagTravelCalibration(
        pipeline=data.pipeline,
        method="self_supervised_power",
        coefficients=tuple(float(item) for item in result.x),
        pred_soft_mg=float(model.pred_soft_mg),
        feature_key=data.feature_key,
        training_log=data.log_name,
        training_time_basis=window.time_basis,
        training_start_s=window.time_range.start_s,
        training_stop_s=window.time_range.stop_s,
        training_sample_start=resolved.sample_start,
        training_sample_stop=resolved.sample_stop,
        training_chunk_count=len(model.chunks),
        model_config=model_config(model),
        training_diagnostics=dict(model.stats.get("training_selection", {})),
        source_fingerprint=data.source_fingerprint,
    )
    return calibration, resolved


def compact_isotonic_knots(model: IsotonicRegression) -> tuple[tuple[float, ...], tuple[float, ...]]:
    mag = np.asarray(model.X_thresholds_, dtype=float)
    travel = np.asarray(model.y_thresholds_, dtype=float)
    if len(mag) <= 2:
        return tuple(mag.tolist()), tuple(travel.tolist())
    changes = np.diff(travel) != 0
    keep = np.zeros(len(mag), dtype=bool)
    keep[0] = True
    keep[-1] = True
    keep[:-1] |= changes
    keep[1:] |= changes
    return tuple(mag[keep].tolist()), tuple(travel[keep].tolist())


def fit_oracle_calibration(
    data: CachedCalibrationData,
    window: RecordingWindow,
    resolved: ResolvedWindow,
    *,
    trainer: str,
    oracle_bins: int,
) -> MagTravelCalibration:
    window_mask = resolved.sample_mask(len(data.time_s))
    training_mask = (
        window_mask
        & data.activity_mask
        & np.isfinite(data.mag)
        & np.isfinite(data.travel)
    )
    mag = data.mag[training_mask]
    travel = data.travel[training_mask]
    if len(mag) < 2 or np.ptp(mag) <= 0:
        raise ValueError("Oracle training window needs at least two distinct finite magnetic samples")
    correlation = float(np.corrcoef(mag, travel)[0, 1])
    increasing = bool(correlation >= 0)

    if trainer == "oracle-power":
        pred_soft_mg = (
            MagToTravelModelCore.pred_soft_mg
            if data.pipeline == "front"
            else RearMagModel.pred_soft_mg
        )
        coefficients, offset, fit_rmse = fit_supervised_power_oracle(
            mag,
            travel,
            pred_soft_mg=float(pred_soft_mg),
        )
        return MagTravelCalibration(
            pipeline=data.pipeline,
            method="oracle_power",
            coefficients=tuple(float(item) for item in coefficients),
            pred_soft_mg=float(pred_soft_mg),
            travel_offset_mm=float(offset),
            feature_key=data.feature_key,
            training_log=data.log_name,
            training_time_basis=window.time_basis,
            training_start_s=window.time_range.start_s,
            training_stop_s=window.time_range.stop_s,
            training_sample_start=resolved.sample_start,
            training_sample_stop=resolved.sample_stop,
            training_chunk_count=0,
            model_config={"supervised_offset": True},
            training_diagnostics={
                "training_samples": int(len(mag)),
                "fit_points": int(len(mag)),
                "mag_travel_correlation": correlation,
                "increasing": increasing,
                "training_rmse": fit_rmse,
            },
            source_fingerprint=data.source_fingerprint,
            format_version=2,
        )
    if trainer == "oracle-isotonic":
        fit_mag = mag
        fit_travel = travel
        sample_weight = None
        method = "oracle_isotonic"
    elif trainer == "oracle-binned-median":
        if oracle_bins < 2:
            raise ValueError("oracle_bins must be at least 2")
        edges = np.linspace(float(np.min(mag)), float(np.max(mag)), oracle_bins + 1)
        bin_ids = np.clip(np.digitize(mag, edges[1:-1]), 0, oracle_bins - 1)
        mag_medians: list[float] = []
        travel_medians: list[float] = []
        counts: list[int] = []
        for bin_id in range(oracle_bins):
            selected = bin_ids == bin_id
            if not np.any(selected):
                continue
            mag_medians.append(float(np.median(mag[selected])))
            travel_medians.append(float(np.median(travel[selected])))
            counts.append(int(np.sum(selected)))
        fit_mag = np.asarray(mag_medians, dtype=float)
        fit_travel = np.asarray(travel_medians, dtype=float)
        sample_weight = np.asarray(counts, dtype=float)
        method = "oracle_binned_median"
    else:
        raise ValueError(f"Unknown trainer {trainer!r}")

    if len(fit_mag) < 2:
        raise ValueError("Oracle training produced fewer than two populated magnetic bins")
    oracle = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
    oracle.fit(fit_mag, fit_travel, sample_weight=sample_weight)
    mag_knots, travel_knots = compact_isotonic_knots(oracle)
    return MagTravelCalibration(
        pipeline=data.pipeline,
        method=method,
        coefficients=None,
        pred_soft_mg=None,
        mag_knots=mag_knots,
        travel_knots=travel_knots,
        feature_key=data.feature_key,
        training_log=data.log_name,
        training_time_basis=window.time_basis,
        training_start_s=window.time_range.start_s,
        training_stop_s=window.time_range.stop_s,
        training_sample_start=resolved.sample_start,
        training_sample_stop=resolved.sample_stop,
        training_chunk_count=0,
        model_config={
            "increasing": increasing,
            "requested_bins": oracle_bins if trainer == "oracle-binned-median" else None,
        },
        training_diagnostics={
            "training_samples": int(len(mag)),
            "fit_points": int(len(fit_mag)),
            "stored_knots": int(len(mag_knots)),
            "mag_travel_correlation": correlation,
            "increasing": increasing,
        },
        source_fingerprint=data.source_fingerprint,
        format_version=2,
    )


def fit_supervised_power_oracle(
    mag: np.ndarray,
    travel: np.ndarray,
    *,
    pred_soft_mg: float,
) -> tuple[np.ndarray, float, float]:
    """Fit the production power curve family plus its otherwise-free offset."""
    mag = np.asarray(mag, dtype=float)
    travel = np.asarray(travel, dtype=float)
    mag_min = float(np.min(mag))
    mag_max = float(np.max(mag))
    mag_span = max(mag_max - mag_min, 1e-6)
    curve = MagToTravelModel(pred_soft_mg=pred_soft_mg)

    lower = np.array([mag_min - 2.0 * mag_span, -np.inf, 0.05, -np.inf])
    upper = np.array([mag_max + 2.0 * mag_span, np.inf, 1.5, np.inf])
    best_result: scipy.optimize.OptimizeResult | None = None
    for x0 in np.quantile(mag, [0.05, 0.5, 0.95]):
        for power in (0.2, 1.0 / 3.0, 0.5):
            unit_feature = curve.pred_x(mag, np.array([x0, 1.0, power]))
            design = np.column_stack([unit_feature, np.ones_like(unit_feature)])
            linear, *_ = np.linalg.lstsq(design, travel, rcond=None)
            initial = np.array([x0, linear[0], power, linear[1]], dtype=float)

            def residual(parameters: np.ndarray) -> np.ndarray:
                coefficients = parameters[:3]
                return curve.pred_x(mag, coefficients) + parameters[3] - travel

            result = scipy.optimize.least_squares(
                residual,
                x0=initial,
                bounds=(lower, upper),
                method="trf",
                max_nfev=500,
            )
            if best_result is None or np.mean(result.fun**2) < np.mean(best_result.fun**2):
                best_result = result

    if best_result is None:
        raise RuntimeError("Supervised power oracle fit produced no optimization result")
    rmse = float(np.sqrt(np.mean(best_result.fun**2)))
    return best_result.x[:3].copy(), float(best_result.x[3]), rmse


def training_diagnostic_columns(calibration: MagTravelCalibration) -> dict[str, Any]:
    return {
        f"train_{key}": value
        for key, value in calibration.training_diagnostics.items()
        if isinstance(value, (bool, int, float, str)) or value is None
    }


def calibration_columns(calibration: MagTravelCalibration) -> dict[str, Any]:
    if calibration.method == "self_supervised_power":
        observation_count = calibration.training_chunk_count
        observation_unit = "chunks"
    else:
        observation_count = int(calibration.training_diagnostics.get("training_samples", 0))
        observation_unit = "samples"
    columns: dict[str, Any] = {
        "calibration_method": calibration.method,
        "training_chunks": calibration.training_chunk_count,
        "training_observations": observation_count,
        "training_observation_unit": observation_unit,
    }
    if calibration.coefficients is not None:
        columns.update(
            {
                "coeff_x0": calibration.coefficients[0],
                "coeff_scale": calibration.coefficients[1],
                "coeff_power": calibration.coefficients[2],
            }
        )
    else:
        columns["oracle_knots"] = len(calibration.mag_knots)
    columns.update(training_diagnostic_columns(calibration))
    return columns


def apply_target_anchor(
    calibration: MagTravelCalibration,
    target: CachedCalibrationData,
    raw_prediction: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Apply the target recording's existing non-GT pipeline anchor policy."""
    if target.pipeline == "rear":
        offset = -float(np.percentile(raw_prediction, 8.0))
        return raw_prediction + offset, offset

    ref_x = float(target.mag_ref_point[0])
    ref_mag = float(target.mag_ref_point[1])
    offset = ref_x - float(calibration.predict(ref_mag))
    adjusted = raw_prediction + offset
    accel_abs = np.abs(target.accel)
    candidate = np.isfinite(accel_abs) & ~target.bad_mask
    if np.any(candidate):
        accel_thresh = float(np.percentile(accel_abs[candidate], 70.0))
        motion = candidate & (accel_abs > accel_thresh)
        if np.any(motion):
            neg_fraction = float(np.mean(adjusted[motion] < 0))
            print(
                f"Ref-point fallback check: {neg_fraction * 100:.1f}% "
                "of motion-mask samples have negative predicted travel"
            )
            if neg_fraction > 0.08:
                zero_mag = float(np.percentile(target.mag, 8.0))
                zero_offset = -float(calibration.predict(zero_mag))
                if zero_offset > offset:
                    print(
                        f"Ref-point fallback: neg_pct={neg_fraction * 100:.1f}% exceeds 8.0%, "
                        f"switching offset from {offset:.1f} to {zero_offset:.1f} "
                        f"using mag p8={zero_mag:.1f}"
                    )
                    offset = zero_offset
                    adjusted = raw_prediction + offset
    return adjusted, offset


def predict_calibration(
    calibration: MagTravelCalibration,
    target: CachedCalibrationData,
) -> tuple[np.ndarray, np.ndarray, float]:
    if calibration.pipeline != target.pipeline:
        raise ValueError(
            f"Cannot apply {calibration.pipeline!r} calibration to {target.pipeline!r} log"
        )
    if calibration.feature_key != target.feature_key:
        raise ValueError(
            f"Calibration feature {calibration.feature_key!r} does not match target feature {target.feature_key!r}"
        )
    raw = calibration.predict(target.mag)
    # Supervised oracle knots already encode absolute reference travel.  The
    # front/rear target anchoring policies exist to resolve the self-supervised
    # power curve's free offset and must not be applied a second time here.
    if calibration.method != "self_supervised_power":
        return raw, raw.copy(), 0.0
    adjusted, offset = apply_target_anchor(calibration, target, raw)
    return raw, adjusted, offset


def score_prediction(
    prediction: np.ndarray,
    target: CachedCalibrationData,
    resolved: ResolvedWindow,
    *,
    include_mask: np.ndarray | None = None,
    exclude_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    window_mask = resolved.sample_mask(len(target.time_s))
    if include_mask is not None:
        include = np.asarray(include_mask, dtype=bool).reshape(-1)
        if include.shape != window_mask.shape:
            raise ValueError("Evaluation include mask has the wrong shape")
        window_mask &= include
    if exclude_mask is not None:
        exclude = np.asarray(exclude_mask, dtype=bool).reshape(-1)
        if exclude.shape != window_mask.shape:
            raise ValueError("Evaluation exclude mask has the wrong shape")
        window_mask &= ~exclude
    mask = (
        window_mask
        & target.activity_mask
        & np.isfinite(prediction)
        & np.isfinite(target.travel)
    )
    count = int(np.sum(mask))
    if count == 0:
        raise ValueError("Evaluation window contains no finite active samples")

    pred = np.asarray(prediction, dtype=float)[mask]
    travel = target.travel[mask]
    raw_error = pred - travel
    aligned_offset = -float(np.mean(raw_error))
    error = raw_error + aligned_offset
    row: dict[str, Any] = {
        "eval_samples": count,
        "eval_active_s": float(np.sum(sample_durations(target.time_s)[mask])),
        "anchored_rmse": float(np.sqrt(np.mean(raw_error**2))),
        "anchored_mae": float(np.mean(np.abs(raw_error))),
        "aligned_offset_mm": aligned_offset,
        "aligned_rmse": float(np.sqrt(np.mean(error**2))),
        "aligned_mae": float(np.mean(np.abs(error))),
        "correlation": float(np.corrcoef(pred, travel)[0, 1]),
    }

    eligible_mses: list[float] = []
    for index, (low, high) in enumerate(zip(TRAVEL_BIN_EDGES[:-1], TRAVEL_BIN_EDGES[1:])):
        upper = travel <= high if index == len(TRAVEL_BIN_EDGES) - 2 else travel < high
        bin_mask = (travel >= low) & upper
        bin_count = int(np.sum(bin_mask))
        row[f"bin{index}_n"] = bin_count
        row[f"bin{index}_rmse"] = (
            float(np.sqrt(np.mean(error[bin_mask] ** 2))) if bin_count else float("nan")
        )
        if bin_count >= MIN_BIN_SAMPLES:
            eligible_mses.append(float(np.mean(error[bin_mask] ** 2)))
    row["bin_rmse"] = (
        float(np.sqrt(np.mean(eligible_mses))) if eligible_mses else float("nan")
    )
    return row


def run_pair(
    train_window: RecordingWindow,
    eval_window: RecordingWindow,
    *,
    trainer: str = "self-supervised",
    oracle_bins: int = 100,
    verbose_fit: bool = False,
) -> tuple[MagTravelCalibration, dict[str, Any], np.ndarray]:
    train_data = load_cached_log(train_window.log_name)
    target = train_data if eval_window.log_name == train_window.log_name else load_cached_log(eval_window.log_name)
    calibration, train_resolved = fit_calibration(
        train_data,
        train_window,
        trainer=trainer,
        oracle_bins=oracle_bins,
        verbose=verbose_fit,
    )
    _, adjusted, target_offset = predict_calibration(calibration, target)
    eval_resolved = resolve_window(
        target.time_s,
        eval_window.time_range,
        time_basis=eval_window.time_basis,
        activity_mask=target.activity_mask,
    )
    row = {
        "pipeline": calibration.pipeline,
        "train_log": train_window.log_name,
        "train_time_basis": train_window.time_basis,
        "train_start_s": train_window.time_range.start_s,
        "train_stop_s": train_window.time_range.stop_s,
        "train_wall_start_s": train_resolved.wall_start_s,
        "train_wall_stop_s": train_resolved.wall_stop_s,
        "train_active_s": train_resolved.active_duration_s,
        "train_cache_fingerprint": calibration.source_fingerprint,
        "eval_log": eval_window.log_name,
        "eval_cache_fingerprint": target.source_fingerprint,
        "eval_time_basis": eval_window.time_basis,
        "eval_start_s": eval_window.time_range.start_s,
        "eval_stop_s": eval_window.time_range.stop_s,
        "eval_wall_start_s": eval_resolved.wall_start_s,
        "eval_wall_stop_s": eval_resolved.wall_stop_s,
        "target_anchor_offset_mm": target_offset,
        **calibration_columns(calibration),
        **score_prediction(adjusted, target, eval_resolved),
    }
    return calibration, row, adjusted


def run_apply(
    calibration: MagTravelCalibration,
    eval_window: RecordingWindow,
) -> tuple[dict[str, Any], np.ndarray, CachedCalibrationData]:
    target = load_cached_log(eval_window.log_name)
    _, adjusted, target_offset = predict_calibration(calibration, target)
    eval_resolved = resolve_window(
        target.time_s,
        eval_window.time_range,
        time_basis=eval_window.time_basis,
        activity_mask=target.activity_mask,
    )
    row = {
        "pipeline": calibration.pipeline,
        "train_log": calibration.training_log,
        "train_time_basis": calibration.training_time_basis,
        "train_start_s": calibration.training_start_s,
        "train_stop_s": calibration.training_stop_s,
        "train_cache_fingerprint": calibration.source_fingerprint,
        "eval_log": eval_window.log_name,
        "eval_cache_fingerprint": target.source_fingerprint,
        "eval_time_basis": eval_window.time_basis,
        "eval_start_s": eval_window.time_range.start_s,
        "eval_stop_s": eval_window.time_range.stop_s,
        "eval_wall_start_s": eval_resolved.wall_start_s,
        "eval_wall_stop_s": eval_resolved.wall_stop_s,
        "target_anchor_offset_mm": target_offset,
        **calibration_columns(calibration),
        **score_prediction(adjusted, target, eval_resolved),
    }
    return row, adjusted, target


def available_duration(data: CachedCalibrationData, time_basis: str) -> float:
    full = resolve_window(
        data.time_s,
        TimeRange(),
        time_basis=time_basis,
        activity_mask=data.activity_mask,
    )
    if time_basis == "active":
        return full.active_duration_s
    return full.wall_stop_s - float(data.time_s[0])


def run_same_block_experiment(
    log_name: str,
    *,
    block_s: float,
    time_basis: str,
    start_s: float = 0.0,
    stop_s: float | None = None,
    trainer: str = "self-supervised",
    oracle_bins: int = 100,
    verbose_fit: bool = False,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, CachedCalibrationData]:
    """Fit each fixed window on itself and compare its stitched output to one fit."""
    if block_s <= 0:
        raise ValueError("block_s must be positive")
    data = load_cached_log(log_name)
    limit = available_duration(data, time_basis) if stop_s is None else float(stop_s)
    if limit <= start_s:
        raise ValueError("Experiment stop must be greater than start")
    block_count = int(np.floor((limit - start_s) / block_s + 1e-9))
    if block_count == 0:
        raise ValueError(
            f"Requested interval contains no complete {block_s:.3f}s blocks"
        )
    covered_stop = start_s + block_count * block_s

    stitched = np.full(len(data.time_s), np.nan, dtype=float)
    rows: list[dict[str, Any]] = []
    boundary_jumps: list[float] = []
    total_chunks = 0
    total_observations = 0
    previous_stop: int | None = None

    for block_index in range(block_count):
        block_start = start_s + block_index * block_s
        block_stop = block_start + block_s
        window = RecordingWindow(
            log_name=log_name,
            time_range=TimeRange(block_start, block_stop),
            time_basis=time_basis,
        )
        calibration, resolved = fit_calibration(
            data,
            window,
            trainer=trainer,
            oracle_bins=oracle_bins,
            verbose=verbose_fit,
        )
        _, adjusted, anchor_offset = predict_calibration(calibration, data)
        stitched[resolved.sample_start:resolved.sample_stop] = adjusted[
            resolved.sample_start:resolved.sample_stop
        ]
        if previous_stop is not None:
            left_candidates = np.flatnonzero(np.isfinite(stitched[:resolved.sample_start]))
            if len(left_candidates):
                left = int(left_candidates[-1])
                boundary_jumps.append(float(abs(stitched[resolved.sample_start] - stitched[left])))
        previous_stop = resolved.sample_stop
        total_chunks += calibration.training_chunk_count
        total_observations += int(calibration_columns(calibration)["training_observations"])
        rows.append(
            {
                "row_type": "block",
                "method": "same_block",
                "pipeline": data.pipeline,
                "log": log_name,
                "cache_fingerprint": data.source_fingerprint,
                "block_index": block_index,
                "time_basis": time_basis,
                "requested_start_s": block_start,
                "requested_stop_s": block_stop,
                "wall_start_s": resolved.wall_start_s,
                "wall_stop_s": resolved.wall_stop_s,
                "active_s": resolved.active_duration_s,
                "target_anchor_offset_mm": anchor_offset,
                **calibration_columns(calibration),
                **score_prediction(adjusted, data, resolved),
            }
        )

    overall_window = RecordingWindow(
        log_name=log_name,
        time_range=TimeRange(start_s, covered_stop),
        time_basis=time_basis,
    )
    full_calibration, overall_resolved = fit_calibration(
        data,
        overall_window,
        trainer=trainer,
        oracle_bins=oracle_bins,
        verbose=verbose_fit,
    )
    _, full_prediction, full_anchor_offset = predict_calibration(full_calibration, data)

    common = {
        "row_type": "aggregate",
        "pipeline": data.pipeline,
        "log": log_name,
        "cache_fingerprint": data.source_fingerprint,
        "block_count": block_count,
        "block_s": block_s,
        "time_basis": time_basis,
        "requested_start_s": start_s,
        "requested_stop_s": covered_stop,
        "wall_start_s": overall_resolved.wall_start_s,
        "wall_stop_s": overall_resolved.wall_stop_s,
        "active_s": overall_resolved.active_duration_s,
    }
    rows.append(
        {
            **common,
            "method": "same_block_stitched",
            "calibration_method": rows[0]["calibration_method"],
            "training_chunks": total_chunks,
            "training_observations": total_observations,
            "training_observation_unit": rows[0]["training_observation_unit"],
            "mean_boundary_jump_mm": float(np.mean(boundary_jumps)) if boundary_jumps else 0.0,
            "max_boundary_jump_mm": float(np.max(boundary_jumps)) if boundary_jumps else 0.0,
            **score_prediction(stitched, data, overall_resolved),
        }
    )
    rows.append(
        {
            **common,
            "method": "one_curve",
            "target_anchor_offset_mm": full_anchor_offset,
            **calibration_columns(full_calibration),
            **score_prediction(full_prediction, data, overall_resolved),
        }
    )
    return rows, stitched, full_prediction, data


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def add_window_arguments(parser: argparse.ArgumentParser, prefix: str) -> None:
    parser.add_argument(f"--{prefix}-log", required=True)
    parser.add_argument(f"--{prefix}-start-s", type=float)
    parser.add_argument(f"--{prefix}-stop-s", type=float)
    parser.add_argument(
        f"--{prefix}-time-basis",
        choices=("elapsed", "active"),
        default="active",
        help="Interpret bounds as wall-clock or accumulated boring_mask seconds (default: active)",
    )


def add_trainer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--trainer",
        choices=(
            "self-supervised",
            "oracle-power",
            "oracle-isotonic",
            "oracle-binned-median",
        ),
        default="self-supervised",
        help="Calibration learner (oracle methods use reference travel only in the training window)",
    )
    parser.add_argument(
        "--oracle-bins",
        type=int,
        default=100,
        help="Uniform magnetic bins for oracle-binned-median (default: 100)",
    )


def window_from_args(args: argparse.Namespace, prefix: str) -> RecordingWindow:
    return RecordingWindow(
        log_name=getattr(args, f"{prefix}_log"),
        time_range=TimeRange(
            start_s=getattr(args, f"{prefix}_start_s"),
            stop_s=getattr(args, f"{prefix}_stop_s"),
        ),
        time_basis=getattr(args, f"{prefix}_time_basis"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit a front or rear mag-to-travel curve on one cached log window and evaluate another."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pair = subparsers.add_parser("pair", help="Run one train-window/evaluation-window pair")
    add_window_arguments(pair, "train")
    add_window_arguments(pair, "eval")
    add_trainer_arguments(pair)
    pair.add_argument("--calibration-out", type=Path, help="Write the portable fitted calibration as JSON")
    pair.add_argument("--metrics-out", type=Path, help="Write the one-row metrics CSV")
    pair.add_argument("--prediction-out", type=Path, help="Write target prediction, travel, time, and masks as NPZ")
    pair.add_argument("--verbose-fit", action="store_true")

    apply_parser = subparsers.add_parser("apply", help="Apply a saved calibration without fitting")
    apply_parser.add_argument("--calibration", type=Path, required=True)
    add_window_arguments(apply_parser, "eval")
    apply_parser.add_argument("--metrics-out", type=Path, help="Write the one-row metrics CSV")
    apply_parser.add_argument("--prediction-out", type=Path, help="Write target prediction, travel, time, and masks as NPZ")

    matrix = subparsers.add_parser("matrix", help="Evaluate every train-log/eval-log pair")
    matrix.add_argument("--train-logs", nargs="+", required=True)
    matrix.add_argument("--eval-logs", nargs="+", required=True)
    matrix.add_argument("--train-start-s", type=float)
    matrix.add_argument("--train-stop-s", type=float)
    matrix.add_argument("--eval-start-s", type=float)
    matrix.add_argument("--eval-stop-s", type=float)
    matrix.add_argument("--time-basis", choices=("elapsed", "active"), default="active")
    add_trainer_arguments(matrix)
    matrix.add_argument("--metrics-out", type=Path, required=True)
    matrix.add_argument("--verbose-fit", action="store_true")

    blocks = subparsers.add_parser(
        "blocks",
        help="Fit every fixed block on itself and compare stitched versus one-curve predictions",
    )
    blocks.add_argument("--log", required=True)
    blocks.add_argument("--block-s", type=float, required=True)
    blocks.add_argument("--time-basis", choices=("elapsed", "active"), default="active")
    blocks.add_argument("--start-s", type=float, default=0.0)
    blocks.add_argument("--stop-s", type=float)
    add_trainer_arguments(blocks)
    blocks.add_argument("--metrics-out", type=Path, required=True)
    blocks.add_argument("--prediction-out", type=Path)
    blocks.add_argument("--verbose-fit", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "pair":
        calibration, row, prediction = run_pair(
            window_from_args(args, "train"),
            window_from_args(args, "eval"),
            trainer=args.trainer,
            oracle_bins=args.oracle_bins,
            verbose_fit=args.verbose_fit,
        )
        if args.calibration_out:
            calibration.save(args.calibration_out)
        if args.metrics_out:
            write_csv(args.metrics_out, [row])
        if args.prediction_out:
            args.prediction_out.parent.mkdir(parents=True, exist_ok=True)
            target = load_cached_log(args.eval_log)
            np.savez_compressed(
                args.prediction_out,
                time_s=target.time_s,
                prediction_mm=prediction,
                travel_mm=target.travel,
                activity_mask=target.activity_mask,
            )
        print(json.dumps(row, indent=2, sort_keys=True))
        return

    if args.command == "apply":
        row, prediction, target = run_apply(
            MagTravelCalibration.load(args.calibration),
            window_from_args(args, "eval"),
        )
        if args.metrics_out:
            write_csv(args.metrics_out, [row])
        if args.prediction_out:
            args.prediction_out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.prediction_out,
                time_s=target.time_s,
                prediction_mm=prediction,
                travel_mm=target.travel,
                activity_mask=target.activity_mask,
            )
        print(json.dumps(row, indent=2, sort_keys=True))
        return

    if args.command == "blocks":
        rows, stitched, one_curve, target = run_same_block_experiment(
            args.log,
            block_s=args.block_s,
            time_basis=args.time_basis,
            start_s=args.start_s,
            stop_s=args.stop_s,
            trainer=args.trainer,
            oracle_bins=args.oracle_bins,
            verbose_fit=args.verbose_fit,
        )
        write_csv(args.metrics_out, rows)
        if args.prediction_out:
            args.prediction_out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.prediction_out,
                time_s=target.time_s,
                same_block_stitched_mm=stitched,
                one_curve_mm=one_curve,
                travel_mm=target.travel,
                activity_mask=target.activity_mask,
            )
        aggregate = [row for row in rows if row["row_type"] == "aggregate"]
        for row in aggregate:
            print(
                f"{row['method']}: aligned RMSE {row['aligned_rmse']:.3f} mm, "
                f"bin RMSE {row['bin_rmse']:.3f} mm, "
                f"{row['training_observations']} {row['training_observation_unit']}"
            )
        print(f"Wrote {len(rows)} rows to {args.metrics_out}")
        return

    rows: list[dict[str, Any]] = []
    for train_log in args.train_logs:
        train_window = RecordingWindow(
            log_name=train_log,
            time_range=TimeRange(args.train_start_s, args.train_stop_s),
            time_basis=args.time_basis,
        )
        calibration, train_resolved = fit_calibration(
            load_cached_log(train_log),
            train_window,
            trainer=args.trainer,
            oracle_bins=args.oracle_bins,
            verbose=args.verbose_fit,
        )
        for eval_log in args.eval_logs:
            row, _, _ = run_apply(
                calibration,
                RecordingWindow(
                    log_name=eval_log,
                    time_range=TimeRange(args.eval_start_s, args.eval_stop_s),
                    time_basis=args.time_basis,
                ),
            )
            row["train_wall_start_s"] = train_resolved.wall_start_s
            row["train_wall_stop_s"] = train_resolved.wall_stop_s
            row["train_active_s"] = train_resolved.active_duration_s
            rows.append(row)
            print(
                f"{train_log} -> {eval_log}: aligned RMSE {row['aligned_rmse']:.3f} mm, "
                f"{row['training_observations']} {row['training_observation_unit']}"
            )
    write_csv(args.metrics_out, rows)
    print(f"Wrote {len(rows)} comparisons to {args.metrics_out}")


if __name__ == "__main__":
    main()
