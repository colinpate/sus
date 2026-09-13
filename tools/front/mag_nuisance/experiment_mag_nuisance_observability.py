#!/usr/bin/env python3
"""Test encoder-blind ways to observe the front magnetometer nuisance field.

Two ideas are compared:

1. Estimate the body/world field only where the existing magnetic path is
   strong and locally observable, then propagate that state through weak-field
   intervals with gyro1 and the random-walk model.
2. Use all samples, but downweight the residual along the local magnetic-curve
   tangent.  A second tangent estimate is learned from short, zero-velocity-
   centered accelerometer integrations, where slowly changing field offsets
   mostly cancel.

Encoder travel is loaded only after every prediction has been generated and is
used solely for metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.front.mag_nuisance.experiment_unsupervised_mag_xyz import (  # noqa: E402
    DEFAULT_LOGS,
    FORK,
    aligned,
    fit_scalar_parameterized_xyz,
    flatten,
)
from backend.mag_nuisance_core import (  # noqa: E402
    PRIMARY_MAG_TO_GYRO,
    MagSolverWeights,
    smooth_body_world_fields,
    solve_iterative_correction,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def cache_values(cache: np.lib.npyio.NpzFile, *keys: str) -> np.ndarray:
    """Read the first available cache series, preferring current key names."""

    for key in keys:
        cache_key = f"{key}__x"
        if cache_key in cache:
            return np.asarray(cache[cache_key])
    raise KeyError(f"None of the cache series are present: {', '.join(keys)}")


@dataclass(frozen=True)
class DenseXYZPath:
    travel_grid: np.ndarray
    xyz_grid: np.ndarray

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

    def infer(self, xyz: np.ndarray) -> np.ndarray:
        return self.travel_grid[self._tree.query(np.asarray(xyz, dtype=float))[1]]

    def covers(self, travel: np.ndarray) -> np.ndarray:
        travel = np.asarray(travel, dtype=float)
        return (travel >= self.travel_min) & (travel <= self.travel_max)


@dataclass
class LogSignals:
    cache: np.lib.npyio.NpzFile
    source_time: np.ndarray
    state_index: np.ndarray
    time_s: np.ndarray
    mag_xyz: np.ndarray
    gyro_dps: np.ndarray
    scalar_mag: np.ndarray
    scalar_travel: np.ndarray
    initial_travel: np.ndarray
    xyz_model: object
    full_mag_xyz: np.ndarray
    full_gyro_dps: np.ndarray
    full_accel: np.ndarray
    full_scalar_mag: np.ndarray
    full_scalar_travel: np.ndarray
    zv_points: np.ndarray


@dataclass(frozen=True)
class AccelTangentFit:
    coefficients: np.ndarray
    chunk_count: int
    bin_count: int
    median_dx_mm: float
    median_fit_rms_mg: float

    def derivative(self, travel: np.ndarray) -> np.ndarray:
        travel = np.asarray(travel, dtype=float)
        return self.coefficients[0] + 2.0 * travel[:, np.newaxis] * self.coefficients[1]


@dataclass(frozen=True)
class AccelWindow:
    center: int
    sample_slice: slice
    displacement_mm: np.ndarray
    weak_center: bool


def load_signals(
    name: str,
    cache_root: Path,
    state_hz: float,
    degree: int,
    travel_max_mm: float,
) -> LogSignals:
    cache = np.load(cache_root / name / "cache" / "all.npz")
    source_time = flatten(cache["mag/lpf__t"])
    source_hz = 1.0 / np.median(np.diff(source_time))
    stride = max(1, round(source_hz / state_hz))
    index = np.arange(0, len(source_time), stride)

    full_mag_xyz = np.asarray(cache["mag/lpf__x"], dtype=float) @ PRIMARY_MAG_TO_GYRO.T
    full_scalar_mag = flatten(
        cache_values(cache, "mag/norm/corr/lpf", "mag/proj/corr/lpf")
    )
    full_raw_scalar_travel = flatten(cache["travel/mag_model__x"])
    full_scalar_travel = flatten(cache["travel/mag_model/adj__x"])
    scalar_offset = float(np.median(full_scalar_travel - full_raw_scalar_travel))
    xyz_model = fit_scalar_parameterized_xyz(
        full_scalar_mag[index],
        full_mag_xyz[index],
        np.asarray(cache["mag_model_coeffs"], dtype=float),
        scalar_offset,
        degree=degree,
        travel_max_mm=travel_max_mm,
    )

    return LogSignals(
        cache=cache,
        source_time=source_time,
        state_index=index,
        time_s=source_time[index],
        mag_xyz=full_mag_xyz[index],
        gyro_dps=np.asarray(cache["gyro/lpf/gyro1__x"], dtype=float)[index],
        scalar_mag=full_scalar_mag[index],
        scalar_travel=full_scalar_travel[index],
        initial_travel=flatten(cache["travel/solved__x"])[index],
        xyz_model=xyz_model,
        full_mag_xyz=full_mag_xyz,
        full_gyro_dps=np.asarray(cache["gyro/lpf/gyro1__x"], dtype=float),
        full_accel=flatten(cache["accel/lpfhp/proj__x"]),
        full_scalar_mag=full_scalar_mag,
        full_scalar_travel=full_scalar_travel,
        zv_points=np.asarray(cache["mag_zv_points"], dtype=int),
    )


def path_derivative(model: object, travel: np.ndarray) -> np.ndarray:
    grid = np.asarray(model.travel_grid, dtype=float)
    derivative_grid = np.gradient(np.asarray(model.xyz_grid), grid, axis=0)
    travel = np.asarray(travel, dtype=float)
    return np.column_stack(
        [np.interp(travel, grid, derivative_grid[:, axis]) for axis in range(3)]
    )


def unit_rows(vectors: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float).copy()
    norms = np.linalg.norm(vectors, axis=1)
    bad = norms < 1e-8
    if np.any(bad):
        if fallback is None:
            vectors[bad] = np.array([1.0, 0.0, 0.0])
        else:
            vectors[bad] = np.asarray(fallback, dtype=float)[bad]
        norms = np.linalg.norm(vectors, axis=1)
    return vectors / norms[:, np.newaxis]


def high_confidence_mask(
    signals: LogSignals,
    expected_travel: np.ndarray,
    min_norm_mg: float,
    min_sensitivity_quantile: float,
    agreement_mm: float,
) -> tuple[np.ndarray, dict[str, float]]:
    expected = signals.xyz_model.predict(expected_travel)
    sensitivity = np.linalg.norm(
        path_derivative(signals.xyz_model, expected_travel), axis=1
    )
    finite_sensitivity = sensitivity[np.isfinite(sensitivity)]
    sensitivity_threshold = float(
        np.quantile(finite_sensitivity, min_sensitivity_quantile)
    )
    mask = (
        signals.xyz_model.covers(expected_travel)
        & (np.linalg.norm(expected, axis=1) >= min_norm_mg)
        & (sensitivity >= sensitivity_threshold)
        & (np.abs(signals.initial_travel - signals.scalar_travel) <= agreement_mm)
        & np.all(np.isfinite(signals.mag_xyz), axis=1)
    )
    return mask, {
        "fraction": float(np.mean(mask)),
        "sensitivity_threshold_mg_per_mm": sensitivity_threshold,
    }


def anisotropic_covariances(
    tangents: np.ndarray,
    anchor_mask: np.ndarray,
    normal_sigma_mg: float,
    tangent_sigma_ratio: float,
) -> np.ndarray:
    tangent_unit = unit_rows(tangents)
    normal_variance = normal_sigma_mg**2
    tangent_variance = (normal_sigma_mg * tangent_sigma_ratio) ** 2
    covariances = np.empty((len(tangents), 3, 3), dtype=float)
    identity = np.eye(3)
    for index, tangent in enumerate(tangent_unit):
        variance = normal_variance if anchor_mask[index] else tangent_variance
        covariances[index] = (
            normal_variance * identity
            + (variance - normal_variance) * np.outer(tangent, tangent)
        )
    return covariances


def estimate_fields(
    signals: LogSignals,
    expected_travel: np.ndarray,
    measurement_mask: np.ndarray,
    weights: MagSolverWeights,
    covariances: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = signals.xyz_model.predict(expected_travel)
    body, world = smooth_body_world_fields(
        signals.time_s,
        signals.gyro_dps,
        signals.mag_xyz,
        expected,
        measurement_mask,
        weights,
        measurement_covariances=covariances,
    )
    return body, world, body + world


def make_proposals(
    signals: LogSignals,
    model: object,
    correction: np.ndarray,
    weak_threshold_mg: float,
) -> tuple[np.ndarray, np.ndarray]:
    corrected = signals.mag_xyz - correction
    inferred = np.clip(model.infer(corrected), model.travel_min, model.travel_max)
    all_proposal = inferred
    weak_mask = (
        (np.linalg.norm(signals.mag_xyz, axis=1) < weak_threshold_mg)
        & model.covers(signals.initial_travel)
    )
    weak_proposal = signals.initial_travel.copy()
    weak_proposal[weak_mask] = inferred[weak_mask]
    return weak_proposal, all_proposal


def interpolate_xyz(
    source_t: np.ndarray, source_xyz: np.ndarray, target_t: np.ndarray
) -> np.ndarray:
    return np.column_stack(
        [np.interp(target_t, source_t, source_xyz[:, axis]) for axis in range(3)]
    )


def build_accel_windows(
    signals: LogSignals,
    *,
    chunk_radius: int,
    min_dx_mm: float,
    max_dx_mm: float,
    min_abs_corr: float,
    max_angle_deg: float,
    weak_threshold_mg: float,
) -> list[AccelWindow]:
    time_s = signals.source_time
    full_mag_norm = np.linalg.norm(signals.full_mag_xyz, axis=1)
    windows: list[AccelWindow] = []
    last_center = -2 * chunk_radius
    for center in signals.zv_points:
        if center - last_center < max(2, chunk_radius // 2):
            continue
        if center < chunk_radius or center + chunk_radius >= len(time_s):
            continue
        sl = slice(center - chunk_radius, center + chunk_radius)
        local_t = time_s[sl]
        local_accel = signals.full_accel[sl] * 1000.0
        velocity = cumulative_trapezoid(local_accel, local_t, initial=0.0)
        velocity -= velocity[chunk_radius]
        displacement = cumulative_trapezoid(velocity, local_t, initial=0.0)
        displacement -= displacement[chunk_radius]
        dx = float(np.ptp(displacement))
        if not min_dx_mm <= dx <= max_dx_mm:
            continue
        corr = spearmanr(signals.full_scalar_mag[sl], displacement).correlation
        if not np.isfinite(corr) or abs(float(corr)) < min_abs_corr:
            continue
        dt = np.diff(local_t, prepend=local_t[0])
        angle_deg = float(
            np.sum(np.linalg.norm(signals.full_gyro_dps[sl], axis=1) * dt)
        )
        if angle_deg > max_angle_deg:
            continue
        windows.append(
            AccelWindow(
                center=int(center),
                sample_slice=sl,
                displacement_mm=displacement,
                weak_center=bool(full_mag_norm[center] < weak_threshold_mg),
            )
        )
        last_center = int(center)
    return windows


def fit_accel_tangent(
    signals: LogSignals,
    state_correction: np.ndarray,
    windows: list[AccelWindow],
    *,
    travel_bin_mm: float,
) -> AccelTangentFit | None:
    full_correction = interpolate_xyz(
        signals.time_s, state_correction, signals.source_time
    )
    mag = signals.full_mag_xyz - full_correction
    slopes: list[np.ndarray] = []
    centers: list[float] = []
    dx_values: list[float] = []
    fit_rms_values: list[float] = []
    for window in windows:
        center = window.center
        displacement = window.displacement_mm
        delta_mag = mag[window.sample_slice] - mag[center]
        selected = np.abs(displacement) >= 1.0
        denominator = float(displacement[selected] @ displacement[selected])
        if denominator < 1e-6:
            continue
        slope = displacement[selected] @ delta_mag[selected] / denominator
        predicted = displacement[selected, np.newaxis] * slope
        fit_rms = float(np.sqrt(np.mean((delta_mag[selected] - predicted) ** 2)))
        slope_norm = float(np.linalg.norm(slope))
        if not 0.25 <= slope_norm <= 150.0 or fit_rms > 300.0:
            continue
        slopes.append(slope)
        centers.append(float(signals.full_scalar_travel[center]))
        dx_values.append(float(np.ptp(displacement)))
        fit_rms_values.append(fit_rms)

    if len(slopes) < 3:
        return None
    slopes_array = np.asarray(slopes)
    centers_array = np.asarray(centers)
    bin_id = np.floor(centers_array / travel_bin_mm).astype(int)
    bin_centers: list[float] = []
    bin_slopes: list[np.ndarray] = []
    for value in np.unique(bin_id):
        selected = bin_id == value
        bin_centers.append(float(np.median(centers_array[selected])))
        bin_slopes.append(np.median(slopes_array[selected], axis=0))
    bin_centers_array = np.asarray(bin_centers)
    bin_slopes_array = np.asarray(bin_slopes)
    if len(bin_centers_array) == 1:
        coefficients = np.vstack((bin_slopes_array[0], np.zeros(3)))
    else:
        design = np.column_stack((np.ones(len(bin_centers_array)), 2.0 * bin_centers_array))
        coefficients = np.linalg.lstsq(design, bin_slopes_array, rcond=None)[0]
    return AccelTangentFit(
        coefficients=coefficients,
        chunk_count=len(slopes),
        bin_count=len(bin_centers_array),
        median_dx_mm=float(np.median(dx_values)),
        median_fit_rms_mg=float(np.median(fit_rms_values)),
    )


def path_from_accel_tangent(
    base_model: object,
    fit: AccelTangentFit,
    alignment_mask: np.ndarray,
) -> DenseXYZPath:
    grid = np.asarray(base_model.travel_grid, dtype=float)
    c1, c2 = fit.coefficients
    unaligned = grid[:, np.newaxis] * c1 + grid[:, np.newaxis] ** 2 * c2
    if np.any(alignment_mask):
        offset = np.median(
            np.asarray(base_model.xyz_grid)[alignment_mask] - unaligned[alignment_mask],
            axis=0,
        )
    else:
        offset = np.median(np.asarray(base_model.xyz_grid) - unaligned, axis=0)
    return DenseXYZPath(grid, unaligned + offset)


def blend_paths(base_model: object, alternate: DenseXYZPath, amount: float) -> DenseXYZPath:
    return DenseXYZPath(
        np.asarray(base_model.travel_grid, dtype=float),
        (1.0 - amount) * np.asarray(base_model.xyz_grid, dtype=float)
        + amount * alternate.xyz_grid,
    )


def accel_displacement_score(
    signals: LogSignals,
    state_travel: np.ndarray,
    windows: list[AccelWindow],
) -> dict[str, float | int]:
    """Score travel increments against independent short accel integrations."""

    full_travel = np.interp(signals.source_time, signals.time_s, state_travel)
    all_scores: list[float] = []
    weak_scores: list[float] = []
    for window in windows:
        predicted_delta = (
            full_travel[window.sample_slice] - full_travel[window.center]
        )
        chunk_rmse = float(
            np.sqrt(np.mean((predicted_delta - window.displacement_mm) ** 2))
        )
        all_scores.append(chunk_rmse)
        if window.weak_center:
            weak_scores.append(chunk_rmse)
    return {
        "chunks": len(all_scores),
        "median_rmse_mm": float(np.median(all_scores)) if all_scores else float("nan"),
        "weak_chunks": len(weak_scores),
        "weak_median_rmse_mm": (
            float(np.median(weak_scores)) if weak_scores else float("nan")
        ),
    }


def score(
    name: str,
    method: str,
    alpha: float,
    region: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | str]:
    error = prediction[mask] - truth[mask]
    return {
        "log": name,
        "fork": FORK.get(name, "unknown"),
        "method": method,
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
    args: argparse.Namespace,
    weights: MagSolverWeights,
) -> tuple[list[dict], dict]:
    signals = load_signals(
        name, args.cache_root, args.state_hz, args.degree, args.travel_max_mm
    )
    accel_windows = build_accel_windows(
        signals,
        chunk_radius=args.chunk_radius,
        min_dx_mm=args.chunk_min_dx_mm,
        max_dx_mm=args.chunk_max_dx_mm,
        min_abs_corr=args.chunk_min_abs_corr,
        max_angle_deg=args.chunk_max_angle_deg,
        weak_threshold_mg=args.weak_threshold_mg,
    )
    if args.accel_fit_block_parity in (0, 1):
        accel_fit_windows = [
            window
            for window in accel_windows
            if int(signals.source_time[window.center] // args.accel_block_s) % 2
            == args.accel_fit_block_parity
        ]
    else:
        accel_fit_windows = accel_windows
    proposals: dict[str, np.ndarray] = {}
    diagnostics: dict[str, object] = {}

    previous = solve_iterative_correction(
        signals.time_s,
        signals.gyro_dps,
        signals.mag_xyz,
        signals.initial_travel,
        signals.xyz_model,
        weights,
        iterations=args.iterations,
    )
    proposals["previous_weak_iterative"] = previous.travel

    base_tangent_solved = path_derivative(signals.xyz_model, signals.initial_travel)
    base_tangent_scalar = path_derivative(signals.xyz_model, signals.scalar_travel)
    accel_fit: AccelTangentFit | None = None
    accel_tangent_solved: np.ndarray | None = None
    accel_curve: DenseXYZPath | None = None

    all_measurements = np.ones(len(signals.time_s), dtype=bool)
    no_full_vector_anchors = np.zeros(len(signals.time_s), dtype=bool)
    for coordinate_name, expected_travel, base_tangent in (
        ("solved", signals.initial_travel, base_tangent_solved),
        ("scalar", signals.scalar_travel, base_tangent_scalar),
    ):
        covariances = anisotropic_covariances(
            base_tangent,
            no_full_vector_anchors,
            weights.mag_sigma,
            args.tangent_sigma_ratio,
        )
        _, _, normal_only_correction = estimate_fields(
            signals,
            expected_travel,
            all_measurements,
            weights,
            covariances,
        )
        weak, all_samples = make_proposals(
            signals,
            signals.xyz_model,
            normal_only_correction,
            args.weak_threshold_mg,
        )
        proposals[f"normal_only_{coordinate_name}_weak"] = weak
        proposals[f"normal_only_{coordinate_name}_all"] = all_samples

    for threshold in args.anchor_norms:
        for coordinate_name, expected_travel, base_tangent in (
            ("solved", signals.initial_travel, base_tangent_solved),
            ("scalar", signals.scalar_travel, base_tangent_scalar),
        ):
            anchor_mask, mask_details = high_confidence_mask(
                signals,
                expected_travel,
                threshold,
                args.min_sensitivity_quantile,
                args.agreement_mm,
            )
            tag = f"h{threshold:g}_{coordinate_name}"
            diagnostics[f"anchor_{tag}"] = mask_details
            if np.sum(anchor_mask) < 3:
                continue
            _, _, anchor_correction = estimate_fields(
                signals, expected_travel, anchor_mask, weights
            )
            weak, all_samples = make_proposals(
                signals, signals.xyz_model, anchor_correction, args.weak_threshold_mg
            )
            proposals[f"anchor_full_{tag}_weak"] = weak
            proposals[f"anchor_full_{tag}_all"] = all_samples

            covariances = anisotropic_covariances(
                base_tangent,
                anchor_mask,
                weights.mag_sigma,
                args.tangent_sigma_ratio,
            )
            _, _, normal_correction = estimate_fields(
                signals,
                expected_travel,
                all_measurements,
                weights,
                covariances,
            )
            weak, all_samples = make_proposals(
                signals, signals.xyz_model, normal_correction, args.weak_threshold_mg
            )
            proposals[f"anchor_normal_{tag}_weak"] = weak
            proposals[f"anchor_normal_{tag}_all"] = all_samples

            if (
                coordinate_name == "solved"
                and threshold == args.accel_anchor_norm_mg
            ):
                accel_fit = fit_accel_tangent(
                    signals,
                    anchor_correction,
                    accel_fit_windows,
                    travel_bin_mm=args.chunk_travel_bin_mm,
                )
                if accel_fit is not None:
                    accel_tangent_solved = accel_fit.derivative(
                        signals.initial_travel
                    )
                    base_grid_norm = np.linalg.norm(
                        signals.xyz_model.xyz_grid, axis=1
                    )
                    accel_curve = path_from_accel_tangent(
                        signals.xyz_model,
                        accel_fit,
                        base_grid_norm >= threshold,
                    )

    if accel_fit is not None and accel_tangent_solved is not None:
        diagnostics["accel_tangent"] = asdict(accel_fit)
        threshold = args.accel_anchor_norm_mg
        anchor_mask, _ = high_confidence_mask(
            signals,
            signals.initial_travel,
            threshold,
            args.min_sensitivity_quantile,
            args.agreement_mm,
        )
        if accel_curve is not None:
            for blend in args.accel_blends:
                blend_tag = f"b{blend:g}"
                base_tangent_unit = unit_rows(base_tangent_solved)
                accel_tangent_unit = unit_rows(accel_tangent_solved)
                opposite = np.sum(
                    base_tangent_unit * accel_tangent_unit, axis=1
                ) < 0.0
                accel_tangent_unit[opposite] *= -1.0
                blended_tangent = unit_rows(
                    (1.0 - blend) * base_tangent_unit
                    + blend * accel_tangent_unit
                )
                covariances = anisotropic_covariances(
                    blended_tangent,
                    anchor_mask,
                    weights.mag_sigma,
                    args.tangent_sigma_ratio,
                )
                _, _, correction = estimate_fields(
                    signals,
                    signals.initial_travel,
                    np.ones(len(signals.time_s), dtype=bool),
                    weights,
                    covariances,
                )
                weak, all_samples = make_proposals(
                    signals, signals.xyz_model, correction, args.weak_threshold_mg
                )
                proposals[f"accel_tangent_{blend_tag}_normal_weak"] = weak
                proposals[f"accel_tangent_{blend_tag}_normal_all"] = all_samples

                normal_only_covariances = anisotropic_covariances(
                    blended_tangent,
                    no_full_vector_anchors,
                    weights.mag_sigma,
                    args.tangent_sigma_ratio,
                )
                _, _, normal_only_correction = estimate_fields(
                    signals,
                    signals.initial_travel,
                    all_measurements,
                    weights,
                    normal_only_covariances,
                )
                weak, all_samples = make_proposals(
                    signals,
                    signals.xyz_model,
                    normal_only_correction,
                    args.weak_threshold_mg,
                )
                proposals[f"accel_tangent_{blend_tag}_normal_only_weak"] = weak
                proposals[f"accel_tangent_{blend_tag}_normal_only_all"] = all_samples

                blended_curve = blend_paths(signals.xyz_model, accel_curve, blend)
                expected = blended_curve.predict(signals.initial_travel)
                body, world = smooth_body_world_fields(
                    signals.time_s,
                    signals.gyro_dps,
                    signals.mag_xyz,
                    expected,
                    anchor_mask,
                    weights,
                )
                weak, all_samples = make_proposals(
                    signals, blended_curve, body + world, args.weak_threshold_mg
                )
                proposals[f"accel_curve_{blend_tag}_anchor_weak"] = weak
                proposals[f"accel_curve_{blend_tag}_anchor_all"] = all_samples
    else:
        diagnostics["accel_tangent"] = None

    diagnostics["proposal_accel_scores"] = {
        method: accel_displacement_score(
            signals,
            proposal,
            accel_windows,
        )
        for method, proposal in {"pipeline": signals.initial_travel, **proposals}.items()
    }

    # Encoder ground truth first appears here. Everything above is deployable
    # without it.
    truth = flatten(aligned(signals.cache, "travel", signals.state_index))
    active = np.isfinite(truth) & (truth >= 0.0)
    if "boring_mask" in signals.cache:
        active &= np.asarray(signals.cache["boring_mask"], dtype=bool)[
            signals.state_index
        ]
    measured_weak = (
        np.linalg.norm(signals.mag_xyz, axis=1) < args.weak_threshold_mg
    )
    regions = {
        "all": active,
        "weak": active & measured_weak,
        "strong": active & ~measured_weak,
    }
    metrics: list[dict] = []
    for method, proposal in proposals.items():
        delta = proposal - signals.initial_travel
        for alpha in args.alphas:
            prediction = signals.initial_travel + alpha * delta
            accel_score = accel_displacement_score(
                signals, prediction, accel_windows
            )
            for region, mask in regions.items():
                row = score(name, method, alpha, region, prediction, truth, mask)
                row["accel_window_rmse_mm"] = accel_score["median_rmse_mm"]
                row["weak_accel_window_rmse_mm"] = accel_score[
                    "weak_median_rmse_mm"
                ]
                row["accel_window_count"] = accel_score["chunks"]
                row["weak_accel_window_count"] = accel_score["weak_chunks"]
                metrics.append(row)
    for region, mask in regions.items():
        row = score(
            name,
            "pipeline",
            0.0,
            region,
            signals.initial_travel,
            truth,
            mask,
        )
        accel_score = diagnostics["proposal_accel_scores"]["pipeline"]
        row["accel_window_rmse_mm"] = accel_score["median_rmse_mm"]
        row["weak_accel_window_rmse_mm"] = accel_score["weak_median_rmse_mm"]
        row["accel_window_count"] = accel_score["chunks"]
        row["weak_accel_window_count"] = accel_score["weak_chunks"]
        metrics.append(row)
    diagnostics.update(
        {
            "log": name,
            "fork": FORK.get(name, "unknown"),
            "samples": len(signals.time_s),
            "duration_s": float(signals.time_s[-1] - signals.time_s[0]),
            "accel_windows": len(accel_windows),
            "accel_fit_windows": len(accel_fit_windows),
            "proposal_methods": sorted(proposals),
        }
    )
    return metrics, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", default=list(DEFAULT_LOGS))
    parser.add_argument(
        "--cache-root", type=Path, default=REPO_ROOT / "backend/run_artifacts"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports/front_mag_nuisance/observability",
    )
    parser.add_argument("--state-hz", type=float, default=10.0)
    parser.add_argument("--degree", type=int, choices=(1, 2), default=2)
    parser.add_argument("--travel-max-mm", type=float, default=210.0)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--weak-threshold-mg", type=float, default=1500.0)
    parser.add_argument(
        "--anchor-norms", type=float, nargs="+", default=[1000, 1500, 2000, 3000]
    )
    parser.add_argument("--min-sensitivity-quantile", type=float, default=0.0)
    parser.add_argument("--agreement-mm", type=float, default=20.0)
    parser.add_argument("--tangent-sigma-ratio", type=float, default=5.0)
    parser.add_argument("--accel-anchor-norm-mg", type=float, default=1500.0)
    parser.add_argument("--chunk-radius", type=int, default=20)
    parser.add_argument("--chunk-min-dx-mm", type=float, default=10.0)
    parser.add_argument("--chunk-max-dx-mm", type=float, default=150.0)
    parser.add_argument("--chunk-min-abs-corr", type=float, default=0.5)
    parser.add_argument("--chunk-max-angle-deg", type=float, default=15.0)
    parser.add_argument("--chunk-travel-bin-mm", type=float, default=10.0)
    parser.add_argument("--accel-fit-block-parity", type=int, choices=(-1, 0, 1), default=-1)
    parser.add_argument("--accel-block-s", type=float, default=20.0)
    parser.add_argument(
        "--accel-blends", type=float, nargs="+", default=[0.75, 0.9, 1.0]
    )
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0]
    )
    return parser.parse_args()


def json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_sensitivity_quantile <= 1.0:
        raise ValueError("min-sensitivity-quantile must be between zero and one")
    if any(not 0.0 <= value <= 1.0 for value in (*args.alphas, *args.accel_blends)):
        raise ValueError("alphas and accel-blends must be between zero and one")
    if args.tangent_sigma_ratio < 1.0:
        raise ValueError("tangent-sigma-ratio must be at least one")
    if args.accel_anchor_norm_mg not in args.anchor_norms:
        raise ValueError("accel-anchor-norm-mg must also appear in anchor-norms")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = MagSolverWeights()
    metrics: list[dict] = []
    details: list[dict] = []
    for name in args.logs:
        print(f"Evaluating {name}...", flush=True)
        log_metrics, log_details = evaluate_log(name, args, weights)
        metrics.extend(log_metrics)
        details.append(log_details)

    frame = pd.DataFrame(metrics)
    frame.to_csv(args.output_dir / "metrics.csv", index=False)
    aggregate = (
        frame.groupby(["fork", "region", "method", "alpha"], as_index=False)[
            "rmse_mm"
        ]
        .median()
        .sort_values(["region", "fork", "rmse_mm"])
    )
    aggregate.to_csv(args.output_dir / "aggregate.csv", index=False)
    payload = {
        "encoder_use": "metrics_only",
        "weights": asdict(weights),
        "arguments": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "logs": details,
    }
    (args.output_dir / "details.json").write_text(
        json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8"
    )
    weak = aggregate[aggregate["region"] == "weak"]
    print("\nBest weak-field median RMSE by fork:")
    print(
        weak.sort_values(["fork", "rmse_mm"])
        .groupby("fork", as_index=False)
        .head(12)
        .to_string(index=False, float_format=lambda value: f"{value:.2f}")
    )
    print(f"\nResults: {args.output_dir}")


if __name__ == "__main__":
    main()
