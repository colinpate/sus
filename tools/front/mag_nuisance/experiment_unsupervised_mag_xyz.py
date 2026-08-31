#!/usr/bin/env python3
"""Evaluate encoder-free XYZ ambient correction and output mappings on front logs.

The expected XYZ path is parameterized by the existing front pipeline's
accelerometer-trained scalar magnet model. Direct XYZ inversion, unchanged
scalar-model outputs, and blends are compared. Encoder travel is never used to
fit the path, estimate the ambient field, or generate a prediction; it is read
only afterward to calculate evaluation metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.front.mag_nuisance.mag_correction_solver import (  # noqa: E402
    PRIMARY_MAG_TO_GYRO,
    MagSolverWeights,
    solve_iterative_correction,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOGS = (
    "log-0056",
    "log-0063",
    "log-0046",
    "log-0048",
    "log-0049",
    "log-0054",
    "log-0055",
    "log-0058",
    "log-0071_183",
    "log-0072_184",
    "log-0073_185",
    "log-0078-valid",
    "log-0079",
    "log-0080-valid",
    "log-0081",
)
FORK = {
    **{name: "fox36" for name in DEFAULT_LOGS[:11]},
    **{name: "boxxer" for name in DEFAULT_LOGS[11:]},
}


@dataclass(frozen=True)
class ScalarParameterizedXYZModel:
    """Dense travel-to-XYZ path with nearest-path inversion."""

    travel_grid: np.ndarray
    xyz_grid: np.ndarray
    scalar_center: float
    scalar_scale: float
    coefficients: np.ndarray
    bin_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "_tree", cKDTree(self.xyz_grid))

    @property
    def travel_min(self) -> float:
        return float(self.travel_grid[0])

    @property
    def travel_max(self) -> float:
        return float(self.travel_grid[-1])

    def predict(self, travel: np.ndarray | float) -> np.ndarray:
        travel = np.asarray(travel, dtype=float)
        return np.column_stack(
            [
                np.interp(travel, self.travel_grid, self.xyz_grid[:, axis])
                for axis in range(3)
            ]
        )

    def infer(self, field_xyz: np.ndarray, clip: bool = True) -> np.ndarray:
        del clip  # The dense path is already bounded by travel_grid.
        return self.travel_grid[self._tree.query(np.asarray(field_xyz, dtype=float))[1]]

    def covers(self, travel: np.ndarray) -> np.ndarray:
        travel = np.asarray(travel, dtype=float)
        return (travel >= self.travel_min) & (travel <= self.travel_max)

    def weak(self, travel: np.ndarray, threshold_mg: float) -> np.ndarray:
        travel = np.asarray(travel, dtype=float)
        return self.covers(travel) & (
            np.linalg.norm(self.predict(travel), axis=-1) <= threshold_mg
        )


def flatten(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def invert_scalar_travel_model(
    travel: np.ndarray,
    coefficients: np.ndarray,
    offset_mm: float,
    *,
    soft_mg: float = 50.0,
) -> np.ndarray:
    """Invert ``MagToTravelModel.pred_x`` including its absolute offset."""

    x0, y_scale, power = np.asarray(coefficients, dtype=float)
    if y_scale <= 0 or power <= 0:
        raise ValueError(
            f"Expected positive scalar-model scale/power, got {y_scale}, {power}"
        )
    normalized = (np.asarray(travel, dtype=float) - offset_mm) / y_scale
    delta = np.sign(normalized) * (
        (np.abs(normalized) + soft_mg**power) ** (1.0 / power) - soft_mg
    )
    return x0 + delta


def predict_scalar_travel(
    scalar_mag: np.ndarray,
    coefficients: np.ndarray,
    offset_mm: float,
    *,
    soft_mg: float = 50.0,
) -> np.ndarray:
    """Apply the existing encoder-free scalar model and its absolute offset."""

    x0, y_scale, power = np.asarray(coefficients, dtype=float)
    delta = np.asarray(scalar_mag, dtype=float) - x0
    softened = (np.abs(delta) + soft_mg) ** power - soft_mg**power
    return np.sign(delta) * softened * y_scale + offset_mm


def fit_scalar_parameterized_xyz(
    scalar_mag: np.ndarray,
    mag_xyz: np.ndarray,
    scalar_coefficients: np.ndarray,
    scalar_offset_mm: float,
    *,
    scalar_bin_mg: float = 100.0,
    degree: int = 2,
    travel_max_mm: float = 210.0,
    travel_step_mm: float = 0.25,
    min_bin_samples: int = 5,
) -> ScalarParameterizedXYZModel:
    """Fit XYZ versus the independently calibrated scalar magnetic coordinate."""

    scalar_mag = flatten(scalar_mag)
    mag_xyz = np.asarray(mag_xyz, dtype=float)
    if mag_xyz.shape != (len(scalar_mag), 3):
        raise ValueError("scalar_mag and mag_xyz must have shapes (N,) and (N, 3)")
    if degree not in (1, 2):
        raise ValueError("degree must be one or two")

    finite = np.isfinite(scalar_mag) & np.all(np.isfinite(mag_xyz), axis=1)
    bin_id = np.floor(scalar_mag / scalar_bin_mg).astype(int)
    scalar_centers: list[float] = []
    xyz_medians: list[np.ndarray] = []
    for value in np.unique(bin_id[finite]):
        selected = finite & (bin_id == value)
        if np.sum(selected) < min_bin_samples:
            continue
        scalar_centers.append(float(np.median(scalar_mag[selected])))
        xyz_medians.append(np.median(mag_xyz[selected], axis=0))
    if len(scalar_centers) < degree + 2:
        raise ValueError("Not enough populated scalar-field bins for XYZ fit")

    scalar_centers_array = np.asarray(scalar_centers)
    xyz_medians_array = np.asarray(xyz_medians)
    center = float(np.median(scalar_centers_array))
    scale = max(float(np.std(scalar_centers_array)), scalar_bin_mg)
    normalized = (scalar_centers_array - center) / scale
    design = np.column_stack(
        [normalized**order for order in range(degree + 1)]
    )
    coefficients = np.linalg.lstsq(design, xyz_medians_array, rcond=None)[0]

    travel_grid = np.arange(0.0, travel_max_mm + 0.5 * travel_step_mm, travel_step_mm)
    scalar_grid = invert_scalar_travel_model(
        travel_grid, scalar_coefficients, scalar_offset_mm
    )
    normalized_grid = (scalar_grid - center) / scale
    xyz_grid = sum(
        normalized_grid[:, np.newaxis] ** order * coefficients[order]
        for order in range(degree + 1)
    )
    return ScalarParameterizedXYZModel(
        travel_grid=travel_grid,
        xyz_grid=xyz_grid,
        scalar_center=center,
        scalar_scale=scale,
        coefficients=coefficients,
        bin_count=len(scalar_centers_array),
    )


def aligned(cache: np.lib.npyio.NpzFile, key: str, index: np.ndarray) -> np.ndarray:
    values = np.asarray(cache[f"{key}__x"])
    if len(values) != len(cache["mag/lpf__t"]):
        raise ValueError(f"{key} is not array-aligned with mag/lpf")
    return values[index]


def aligned_first(
    cache: np.lib.npyio.NpzFile, index: np.ndarray, *keys: str
) -> np.ndarray:
    """Read the first available aligned series, preferring current cache keys."""

    for key in keys:
        if f"{key}__x" in cache:
            return aligned(cache, key, index)
    raise KeyError(f"None of the cache series are present: {', '.join(keys)}")


def metric(
    log_name: str,
    fork: str,
    alpha: float,
    region: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | str]:
    error = prediction[mask] - truth[mask]
    return {
        "log": log_name,
        "fork": fork,
        "alpha": alpha,
        "region": region,
        "samples": int(np.sum(mask)),
        "rmse_mm": float(np.sqrt(np.mean(error**2))),
        "mae_mm": float(np.mean(np.abs(error))),
        "bias_mm": float(np.mean(error)),
        "centered_rmse_mm": float(np.std(error)),
    }


def evaluate_log(
    name: str,
    cache_root: Path,
    state_hz: float,
    alphas: list[float],
    weights: MagSolverWeights,
    iterations: int,
    degree: int,
    travel_max_mm: float,
) -> tuple[list[dict], dict]:
    cache_path = cache_root / name / "cache" / "all.npz"
    cache = np.load(cache_path)
    source_time = flatten(cache["mag/lpf__t"])
    source_hz = 1.0 / np.median(np.diff(source_time))
    stride = max(1, round(source_hz / state_hz))
    index = np.arange(0, len(source_time), stride)
    time_s = source_time[index]

    mag_xyz = np.asarray(cache["mag/lpf__x"], dtype=float)[index] @ PRIMARY_MAG_TO_GYRO.T
    gyro_dps = np.asarray(cache["gyro/lpf/gyro1__x"], dtype=float)[index]
    scalar_mag = flatten(
        aligned_first(cache, index, "mag/norm/corr/lpf", "mag/proj/corr/lpf")
    )
    projected_mag = flatten(
        aligned_first(cache, index, "mag/norm/lpf", "mag/proj/lpf")
    )
    initial_travel = flatten(aligned(cache, "travel/solved", index))
    raw_scalar_travel = flatten(aligned(cache, "travel/mag_model", index))
    adjusted_scalar_travel = flatten(aligned(cache, "travel/mag_model/adj", index))
    scalar_offset = float(np.median(adjusted_scalar_travel - raw_scalar_travel))

    scalar_coefficients = np.asarray(cache["mag_model_coeffs"], dtype=float)
    xyz_model = fit_scalar_parameterized_xyz(
        scalar_mag,
        mag_xyz,
        scalar_coefficients,
        scalar_offset,
        degree=degree,
        travel_max_mm=travel_max_mm,
    )
    correction = solve_iterative_correction(
        time_s,
        gyro_dps,
        mag_xyz,
        initial_travel,
        xyz_model,
        weights,
        iterations=iterations,
    )
    projection_vector = np.linalg.lstsq(mag_xyz, projected_mag, rcond=None)[0]
    projection_vector /= np.linalg.norm(projection_vector)
    corrected_xyz = correction.corrected_mag_weak
    corrected_projection = corrected_xyz @ projection_vector
    corrected_norm = np.linalg.norm(corrected_xyz, axis=1)
    corrected_projection_or_norm = corrected_projection.copy()
    use_norm = (
        (np.abs(corrected_norm - corrected_projection) > 500.0)
        & (corrected_norm > 1000.0)
    )
    corrected_projection_or_norm[use_norm] = corrected_norm[use_norm]

    def scalar_proposal(signal: np.ndarray) -> np.ndarray:
        inferred = np.clip(
            predict_scalar_travel(signal, scalar_coefficients, scalar_offset),
            0.0,
            travel_max_mm,
        )
        proposal = initial_travel.copy()
        proposal[correction.update_mask] = inferred[correction.update_mask]
        return proposal

    xyz_proposal = correction.travel
    projection_proposal = scalar_proposal(corrected_projection)
    magnitude_proposal = scalar_proposal(corrected_norm)
    proposals = {
        "xyz_path": xyz_proposal,
        "raw_scalar_control": scalar_proposal(scalar_mag),
        "corrected_projection": projection_proposal,
        "corrected_projection_or_norm": scalar_proposal(
            corrected_projection_or_norm
        ),
        "corrected_magnitude": magnitude_proposal,
        "xyz_projection_blend": 0.5 * (xyz_proposal + projection_proposal),
        "xyz_magnitude_blend": 0.5 * (xyz_proposal + magnitude_proposal),
    }

    # Ground truth first appears here. Everything above this line is deployable
    # without an encoder.
    truth = flatten(aligned(cache, "travel", index))
    if "boring_mask" in cache and len(cache["boring_mask"]) == len(source_time):
        active = np.asarray(cache["boring_mask"], dtype=bool).reshape(-1)[index]
    else:
        active = np.ones(len(index), dtype=bool)
    active &= np.isfinite(truth) & (truth >= 0.0)
    measured_weak = np.linalg.norm(mag_xyz, axis=1) < weights.mag_update_threshold
    regions = {
        "all": active,
        "weak": active & measured_weak,
        "strong": active & ~measured_weak,
    }

    metrics: list[dict] = []
    for method, proposal in proposals.items():
        proposed_change = proposal - initial_travel
        for alpha in alphas:
            prediction = initial_travel + alpha * proposed_change
            for region, mask in regions.items():
                row = metric(
                    name,
                    FORK.get(name, "unknown"),
                    alpha,
                    region,
                    prediction,
                    truth,
                    mask,
                )
                row["method"] = method
                metrics.append(row)
    updated = correction.update_mask
    xyz_change = proposals["xyz_path"] - initial_travel
    details = {
        "log": name,
        "fork": FORK.get(name, "unknown"),
        "samples": len(index),
        "duration_s": float(time_s[-1] - time_s[0]),
        "scalar_offset_mm": scalar_offset,
        "xyz_bin_count": xyz_model.bin_count,
        "xyz_scalar_center_mg": xyz_model.scalar_center,
        "xyz_scalar_scale_mg": xyz_model.scalar_scale,
        "xyz_coefficients": xyz_model.coefficients.tolist(),
        "projection_vector_in_gyro_frame": projection_vector.tolist(),
        "corrected_projection_norm_fallback_fraction": float(np.mean(use_norm)),
        "update_fraction": float(np.mean(updated)),
        "proposed_update_rms_mm": float(
            np.sqrt(np.mean(xyz_change[updated] ** 2))
        ) if np.any(updated) else 0.0,
        "inner_iteration_change_mm": correction.iteration_change_mm,
    }
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
        default=REPO_ROOT / "reports/front_mag_nuisance/encoder_free_xyz",
    )
    parser.add_argument("--state-hz", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--degree", type=int, choices=(1, 2), default=2)
    parser.add_argument("--travel-max-mm", type=float, default=210.0)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.state_hz <= 0 or args.iterations < 1 or args.travel_max_mm <= 0:
        raise ValueError("state-hz/travel-max must be positive and iterations >= 1")
    if any(alpha < 0 or alpha > 1 for alpha in args.alphas):
        raise ValueError("alphas must be between zero and one")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = MagSolverWeights()
    metrics: list[dict] = []
    details: list[dict] = []
    for name in args.logs:
        print(f"Evaluating {name}...", flush=True)
        log_metrics, log_details = evaluate_log(
            name,
            args.cache_root,
            args.state_hz,
            args.alphas,
            weights,
            args.iterations,
            args.degree,
            args.travel_max_mm,
        )
        metrics.extend(log_metrics)
        details.append(log_details)

    frame = pd.DataFrame(metrics)
    frame.to_csv(args.output_dir / "metrics.csv", index=False)
    aggregate = (
        frame.groupby(["fork", "region", "method", "alpha"], as_index=False)["rmse_mm"]
        .median()
        .sort_values(["region", "fork", "rmse_mm"])
    )
    aggregate.to_csv(args.output_dir / "aggregate.csv", index=False)
    payload = {
        "encoder_use": "metrics_only",
        "weights": asdict(weights),
        "arguments": {
            "logs": args.logs,
            "state_hz": args.state_hz,
            "iterations": args.iterations,
            "degree": args.degree,
            "travel_max_mm": args.travel_max_mm,
            "alphas": args.alphas,
        },
        "logs": details,
    }
    (args.output_dir / "details.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print("\nMedian per-log RMSE (mm):")
    print(aggregate.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\nResults: {args.output_dir}")


if __name__ == "__main__":
    main()
