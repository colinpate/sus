#!/usr/bin/env python3
"""Evaluate continuous body/world magnetometer correction on front logs.

The low-field XYZ model and every scalar-to-travel curve are trained on
alternating calibration blocks. Metrics are reported only on the held-out
blocks. The body/world smoother may inspect the full magnetometer/gyro record,
but receives no held-out encoder travel.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sus-matplotlib-cache")
)

import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.mag_nuisance_core import (  # noqa: E402
    PRIMARY_MAG_TO_GYRO,
    MagSolverWeights,
    fit_linear_xyz_model,
    solve_iterative_correction,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOGS = (
    "log103",
    "log110",
    "log-0046",
    "log-0072_184",
    "log-0078-valid",
    "log-0079",
    "log-0080-valid",
    "log-0081",
)
POD_VERSION = {
    "log085": "v1",
    "log103": "v1",
    "log110": "v1",
    "log-0062": "v2",
    "log-0056": "v2",
    "log-0063": "v2",
    "log-0046": "v2",
    "log-0048": "v2",
    "log-0049": "v2",
    "log-0054": "v2",
    "log-0055": "v2",
    "log-0058": "v2",
    "log-0071_183": "v2",
    "log-0072_184": "v2",
    "log-0073_185": "v2",
    "log-0078-valid": "v2-new",
    "log-0079": "v2-new",
    "log-0080-valid": "v2-new",
    "log-0081": "v2-new",
}


@dataclass(frozen=True)
class ScalarTravelCurve:
    travel_grid: np.ndarray
    signal_grid: np.ndarray

    def infer(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal, dtype=float).reshape(-1, 1)
        tree = cKDTree(self.signal_grid.reshape(-1, 1))
        return self.travel_grid[tree.query(signal)[1]]


@dataclass
class LogData:
    name: str
    time_s: np.ndarray
    mag_body: np.ndarray
    gyro_dps: np.ndarray
    travel_gt: np.ndarray
    active_mask: np.ndarray
    pipeline_mag: np.ndarray | None
    pipeline_solved: np.ndarray | None


def flatten_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    return values.reshape(-1)


def interpolate_columns(
    target_time: np.ndarray, source_time: np.ndarray, values: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return np.interp(target_time, source_time, values)
    return np.column_stack(
        [np.interp(target_time, source_time, values[:, axis]) for axis in range(values.shape[1])]
    )


def optional_cache_series(
    cache: np.lib.npyio.NpzFile,
    key: str,
    target_time: np.ndarray,
    source_length: int,
    source_index: np.ndarray,
) -> np.ndarray | None:
    time_key = f"{key}__t"
    value_key = f"{key}__x"
    if time_key not in cache or value_key not in cache:
        return None
    values = flatten_1d(cache[value_key])
    # Pipeline outputs are array-aligned even when a step accidentally carries
    # a different time vector (travel/solved does this in current front caches).
    if len(values) == source_length:
        return values[source_index]
    return flatten_1d(
        interpolate_columns(target_time, cache[time_key], values)
    )


def load_log(name: str, cache_root: Path, state_hz: float) -> LogData:
    cache_path = cache_root / name / "cache" / "all.npz"
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    cache = np.load(cache_path)
    required = (
        "mag/lpf__t",
        "mag/lpf__x",
        "gyro/lpf/gyro1__t",
        "gyro/lpf/gyro1__x",
        "travel__t",
        "travel__x",
    )
    missing = [key for key in required if key not in cache]
    if missing:
        raise KeyError(f"{name}: cache is missing {missing}; rerun backend/pipeline.py")

    source_time = np.asarray(cache["mag/lpf__t"], dtype=float)
    fs_hz = 1.0 / np.median(np.diff(source_time))
    stride = max(1, round(fs_hz / state_hz))
    index = np.arange(0, len(source_time), stride)
    time_s = source_time[index]
    mag_recorded = np.asarray(cache["mag/lpf__x"], dtype=float)[index]
    mag_body = mag_recorded @ PRIMARY_MAG_TO_GYRO.T
    gyro_dps = interpolate_columns(
        time_s,
        np.asarray(cache["gyro/lpf/gyro1__t"], dtype=float),
        np.asarray(cache["gyro/lpf/gyro1__x"], dtype=float),
    )
    travel_gt = flatten_1d(
        interpolate_columns(time_s, cache["travel__t"], cache["travel__x"])
    )
    pipeline_mag = optional_cache_series(
        cache, "travel/mag_model/adj", time_s, len(source_time), index
    )
    pipeline_solved = optional_cache_series(
        cache, "travel/solved", time_s, len(source_time), index
    )
    if "boring_mask" in cache and len(cache["boring_mask"]) == len(source_time):
        active_mask = np.asarray(cache["boring_mask"], dtype=bool).reshape(-1)[index]
    else:
        active_mask = np.ones(len(time_s), dtype=bool)
    finite = (
        np.all(np.isfinite(mag_body), axis=1)
        & np.all(np.isfinite(gyro_dps), axis=1)
        & np.isfinite(travel_gt)
    )
    if not np.all(finite):
        time_s = time_s[finite]
        mag_body = mag_body[finite]
        gyro_dps = gyro_dps[finite]
        travel_gt = travel_gt[finite]
        active_mask = active_mask[finite]
        if pipeline_mag is not None:
            pipeline_mag = pipeline_mag[finite]
        if pipeline_solved is not None:
            pipeline_solved = pipeline_solved[finite]
    return LogData(
        name=name,
        time_s=time_s,
        mag_body=mag_body,
        gyro_dps=gyro_dps,
        travel_gt=travel_gt,
        active_mask=active_mask,
        pipeline_mag=pipeline_mag,
        pipeline_solved=pipeline_solved,
    )


def alternating_masks(time_s: np.ndarray, block_s: float) -> tuple[np.ndarray, np.ndarray]:
    block = np.floor((time_s - time_s[0]) / block_s).astype(int)
    training = block % 2 == 0
    return training, ~training


def fit_scalar_curve(
    travel: np.ndarray,
    signal: np.ndarray,
    training_mask: np.ndarray,
    *,
    bin_mm: float = 5.0,
    min_bin_samples: int = 5,
) -> ScalarTravelCurve:
    travel = flatten_1d(travel)
    signal = flatten_1d(signal)
    valid = training_mask & np.isfinite(travel) & np.isfinite(signal)
    bin_id = np.floor(travel / bin_mm).astype(int)
    centers: list[float] = []
    medians: list[float] = []
    for value in np.unique(bin_id[valid]):
        selected = valid & (bin_id == value)
        if np.sum(selected) < min_bin_samples:
            continue
        centers.append(float(np.median(travel[selected])))
        medians.append(float(np.median(signal[selected])))
    if len(centers) < 3:
        raise ValueError("Fewer than three populated scalar travel bins")
    centers_arr = np.asarray(centers)
    medians_arr = np.asarray(medians)
    order = np.argsort(centers_arr)
    centers_arr = centers_arr[order]
    medians_arr = medians_arr[order]
    travel_grid = np.arange(centers_arr[0], centers_arr[-1] + 0.25, 0.25)
    signal_grid = np.interp(travel_grid, centers_arr, medians_arr)
    return ScalarTravelCurve(travel_grid=travel_grid, signal_grid=signal_grid)


def projection_vector(mag: np.ndarray, training_mask: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(mag, axis=1)
    selected = training_mask & (norm > 1500.0)
    if np.sum(selected) < 20:
        selected = training_mask & (norm >= np.percentile(norm[training_mask], 80))
    directions = mag[selected] / np.maximum(norm[selected, None], 1e-12)
    vector = np.mean(directions, axis=0)
    return vector / np.linalg.norm(vector)


def predict_scalar_method(
    signal: np.ndarray, travel_gt: np.ndarray, training_mask: np.ndarray
) -> np.ndarray:
    return fit_scalar_curve(travel_gt, signal, training_mask).infer(signal)


def metric_row(
    log_name: str,
    method: str,
    region: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | str]:
    selected = mask & np.isfinite(prediction) & np.isfinite(truth)
    error = prediction[selected] - truth[selected]
    if len(error) == 0:
        return {
            "log": log_name,
            "pod": POD_VERSION.get(log_name, "unknown"),
            "method": method,
            "region": region,
            "samples": 0,
            "rmse_mm": float("nan"),
            "mae_mm": float("nan"),
            "p95_abs_mm": float("nan"),
            "bias_mm": float("nan"),
            "centered_rmse_mm": float("nan"),
        }
    return {
        "log": log_name,
        "pod": POD_VERSION.get(log_name, "unknown"),
        "method": method,
        "region": region,
        "samples": int(len(error)),
        "rmse_mm": float(np.sqrt(np.mean(error * error))),
        "mae_mm": float(np.mean(np.abs(error))),
        "p95_abs_mm": float(np.percentile(np.abs(error), 95)),
        "bias_mm": float(np.mean(error)),
        "centered_rmse_mm": float(np.sqrt(np.mean((error - np.mean(error)) ** 2))),
    }


def plot_log(
    output_path: Path,
    data: LogData,
    evaluation_mask: np.ndarray,
    weak_mask: np.ndarray,
    predictions: dict[str, np.ndarray],
    correction_norm: np.ndarray,
) -> None:
    methods = [
        key
        for key in ("raw_projection", "corrected_xyz_line", "corrected_magnitude_weak")
        if key in predictions
    ]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(data.time_s, data.travel_gt, color="black", linewidth=1, label="encoder")
    for method in methods:
        axes[0].plot(data.time_s, predictions[method], linewidth=0.8, label=method)
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("Travel (mm)")

    for method in methods:
        error = predictions[method] - data.travel_gt
        axes[1].plot(data.time_s[evaluation_mask], error[evaluation_mask], ".", markersize=1, label=method)
    axes[1].axhline(0.0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Held-out error (mm)")
    axes[1].legend(loc="upper right")

    axes[2].plot(data.time_s, correction_norm, label="|body + world|")
    axes[2].fill_between(
        data.time_s,
        0,
        np.nanmax(correction_norm),
        where=weak_mask,
        alpha=0.12,
        label="low-field region",
    )
    axes[2].set_ylabel("Correction (mG)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right")
    fig.suptitle(data.name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_log(
    name: str,
    cache_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
    weights: MagSolverWeights,
) -> tuple[list[dict], dict]:
    data = load_log(name, cache_root, args.state_hz)
    training, evaluation = alternating_masks(data.time_s, args.calibration_block_s)
    evaluation &= data.active_mask & (data.travel_gt >= 0.0)

    xyz_model = fit_linear_xyz_model(
        data.travel_gt,
        data.mag_body,
        training,
        threshold_mg=weights.mag_update_threshold,
        bin_mm=args.bin_mm,
    )
    projection = projection_vector(data.mag_body, training)
    raw_projection_signal = data.mag_body @ projection
    raw_magnitude_signal = np.linalg.norm(data.mag_body, axis=1)
    raw_projection = predict_scalar_method(
        raw_projection_signal, data.travel_gt, training
    )
    raw_magnitude = predict_scalar_method(raw_magnitude_signal, data.travel_gt, training)

    correction = solve_iterative_correction(
        data.time_s,
        data.gyro_dps,
        data.mag_body,
        raw_projection,
        xyz_model,
        weights,
        iterations=args.iterations,
    )
    corrected_projection_weak_signal = correction.corrected_mag_weak @ projection
    corrected_projection_all_signal = correction.corrected_mag_all @ projection
    corrected_magnitude_weak_signal = np.linalg.norm(correction.corrected_mag_weak, axis=1)
    corrected_magnitude_all_signal = np.linalg.norm(correction.corrected_mag_all, axis=1)

    weak_training = training & xyz_model.weak(
        data.travel_gt, weights.mag_update_threshold
    )
    corrected_projection_weak_low = predict_scalar_method(
        corrected_projection_weak_signal, data.travel_gt, weak_training
    )
    corrected_magnitude_weak_low = predict_scalar_method(
        corrected_magnitude_weak_signal, data.travel_gt, weak_training
    )
    corrected_projection_weak = raw_projection.copy()
    corrected_projection_weak[correction.update_mask] = (
        corrected_projection_weak_low[correction.update_mask]
    )
    corrected_magnitude_weak = raw_magnitude.copy()
    corrected_magnitude_weak[correction.update_mask] = (
        corrected_magnitude_weak_low[correction.update_mask]
    )
    raw_xyz_line = raw_projection.copy()
    raw_xyz_mask = xyz_model.weak(raw_projection, weights.mag_update_threshold)
    raw_xyz_line[raw_xyz_mask] = xyz_model.infer(data.mag_body)[raw_xyz_mask]

    predictions = {
        "raw_projection": raw_projection,
        "raw_magnitude": raw_magnitude,
        "raw_xyz_line": raw_xyz_line,
        "corrected_xyz_line": correction.travel,
        "corrected_projection_weak": corrected_projection_weak,
        "corrected_projection_all": predict_scalar_method(
            corrected_projection_all_signal, data.travel_gt, training
        ),
        "corrected_magnitude_weak": corrected_magnitude_weak,
        "corrected_magnitude_all": predict_scalar_method(
            corrected_magnitude_all_signal, data.travel_gt, training
        ),
    }
    training_score_mask = training & data.active_mask & (data.travel_gt >= 0.0)
    raw_training_rmse = {
        method: float(
            np.sqrt(
                np.mean(
                    (prediction[training_score_mask] - data.travel_gt[training_score_mask])
                    ** 2
                )
            )
        )
        for method, prediction in (
            ("raw_projection", raw_projection),
            ("raw_magnitude", raw_magnitude),
        )
    }
    selected_raw_method = min(raw_training_rmse, key=raw_training_rmse.get)
    xyz_slope_norm = float(np.linalg.norm(xyz_model.slope))
    correction_enabled = xyz_slope_norm < args.correction_slope_max
    predictions["setup_adaptive"] = (
        corrected_magnitude_weak
        if correction_enabled
        else predictions[selected_raw_method]
    )
    if data.pipeline_mag is not None:
        predictions["pipeline_mag_cached"] = data.pipeline_mag
    if data.pipeline_solved is not None:
        predictions["pipeline_solved_cached"] = data.pipeline_solved

    weak_truth = xyz_model.weak(data.travel_gt, weights.mag_update_threshold)
    regions = {
        "all": evaluation,
        "weak": evaluation & weak_truth,
        "strong": evaluation & ~weak_truth,
    }
    metrics = [
        metric_row(name, method, region, prediction, data.travel_gt, mask)
        for method, prediction in predictions.items()
        for region, mask in regions.items()
    ]
    details = {
        "log": name,
        "pod": POD_VERSION.get(name, "unknown"),
        "samples": len(data.time_s),
        "duration_s": float(data.time_s[-1] - data.time_s[0]),
        "training_fraction": float(np.mean(training)),
        "weak_truth_fraction": float(np.mean(weak_truth)),
        "solver_update_fraction": float(np.mean(correction.update_mask)),
        "xyz_model": {
            "slope": xyz_model.slope.tolist(),
            "intercept": xyz_model.intercept.tolist(),
            "travel_min": xyz_model.travel_min,
            "travel_max": xyz_model.travel_max,
            "bin_count": xyz_model.bin_count,
            "slope_norm_mg_per_mm": xyz_slope_norm,
        },
        "raw_training_rmse_mm": raw_training_rmse,
        "selected_raw_method": selected_raw_method,
        "correction_enabled_by_slope": correction_enabled,
        "iteration_change_mm": correction.iteration_change_mm,
        "body_norm_median_mg": float(np.median(np.linalg.norm(correction.body_field, axis=1))),
        "world_norm_median_mg": float(np.median(np.linalg.norm(correction.world_field, axis=1))),
        "correction_norm_median_mg": float(np.median(np.linalg.norm(correction.correction, axis=1))),
    }
    plot_log(
        output_dir / f"{name}.png",
        data,
        evaluation,
        weak_truth,
        predictions,
        np.linalg.norm(correction.correction, axis=1),
    )
    return metrics, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", default=list(DEFAULT_LOGS))
    parser.add_argument(
        "--cache-root", type=Path, default=REPO_ROOT / "backend/run_artifacts"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports/front_mag_nuisance/supervised_body_world",
    )
    parser.add_argument("--state-hz", type=float, default=10.0)
    parser.add_argument("--calibration-block-s", type=float, default=20.0)
    parser.add_argument("--bin-mm", type=float, default=5.0)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument(
        "--correction-slope-max",
        type=float,
        default=15.0,
        help="Enable the exploratory setup-adaptive correction below this XYZ slope norm.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.state_hz <= 0
        or args.calibration_block_s <= 0
        or args.bin_mm <= 0
        or args.correction_slope_max <= 0
    ):
        raise ValueError(
            "state-hz, calibration-block-s, bin-mm, and correction-slope-max "
            "must be positive"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = MagSolverWeights()
    all_metrics: list[dict] = []
    all_details: list[dict] = []
    for name in args.logs:
        print(f"Evaluating {name}...", flush=True)
        metrics, details = evaluate_log(
            name, args.cache_root, args.output_dir, args, weights
        )
        all_metrics.extend(metrics)
        all_details.append(details)
        weak_rows = {
            row["method"]: row["rmse_mm"]
            for row in metrics
            if row["region"] == "weak"
        }
        print(
            "  weak RMSE:",
            "raw projection",
            f"{weak_rows['raw_projection']:.2f}",
            "XYZ",
            f"{weak_rows['corrected_xyz_line']:.2f}",
            "corrected magnitude",
            f"{weak_rows['corrected_magnitude_weak']:.2f}",
            flush=True,
        )

    frame = pd.DataFrame(all_metrics)
    frame.to_csv(args.output_dir / "metrics.csv", index=False)
    payload = {
        "weights": asdict(weights),
        "arguments": {
            "logs": args.logs,
            "state_hz": args.state_hz,
            "calibration_block_s": args.calibration_block_s,
            "bin_mm": args.bin_mm,
            "iterations": args.iterations,
            "correction_slope_max": args.correction_slope_max,
        },
        "logs": all_details,
    }
    (args.output_dir / "details.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    aggregate = (
        frame.groupby(["region", "method"], as_index=False)["rmse_mm"]
        .median()
        .sort_values(["region", "rmse_mm"])
    )
    aggregate.to_csv(args.output_dir / "aggregate.csv", index=False)
    print("\nMedian per-log RMSE (mm):")
    print(aggregate.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\nResults: {args.output_dir}")


if __name__ == "__main__":
    main()
