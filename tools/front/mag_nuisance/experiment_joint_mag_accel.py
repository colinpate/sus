#!/usr/bin/env python3
"""Evaluate a joint encoder-free travel and nuisance-field factor graph.

The solver jointly estimates latent travel, body-fixed magnetic field, and a
gyro-transported world field.  Magnetometer residuals use the local XYZ curve
tangent while zero-velocity-centered accelerometer integrations constrain
relative travel directly.  Encoder travel is loaded only after predictions are
complete and is used solely for metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.front.mag_nuisance.experiment_mag_nuisance_observability import (  # noqa: E402
    AccelWindow,
    anisotropic_covariances,
    build_accel_windows,
    fit_accel_tangent,
    high_confidence_mask,
    load_signals,
    make_proposals,
    path_derivative,
    unit_rows,
)
from tools.front.mag_nuisance.experiment_unsupervised_mag_xyz import (  # noqa: E402
    DEFAULT_LOGS,
    FORK,
    aligned,
    flatten,
)
from tools.front.mag_nuisance.joint_mag_accel_solver import (  # noqa: E402
    JointSolverWeights,
    RelativeTravelFactor,
    solve_joint_mag_accel,
)
from tools.front.mag_nuisance.mag_correction_solver import (  # noqa: E402
    MagSolverWeights,
    smooth_body_world_fields,
    solve_iterative_correction,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def blend_tangent_fn(signals: object, accel_fit: object, blend: float):
    def tangent(travel: np.ndarray) -> np.ndarray:
        base = unit_rows(path_derivative(signals.xyz_model, travel))
        accel = unit_rows(accel_fit.derivative(np.asarray(travel, dtype=float)))
        opposite = np.sum(base * accel, axis=1) < 0.0
        accel[opposite] *= -1.0
        return unit_rows((1.0 - blend) * base + blend * accel)

    return tangent


def relative_factors_from_windows(
    state_time: np.ndarray,
    source_time: np.ndarray,
    windows: list[AccelWindow],
) -> list[RelativeTravelFactor]:
    factors: list[RelativeTravelFactor] = []
    seen: set[tuple[int, int, int]] = set()
    for window_number, window in enumerate(windows):
        window_time = source_time[window.sample_slice]
        state_indices = np.flatnonzero(
            (state_time >= window_time[0]) & (state_time <= window_time[-1])
        )
        if len(state_indices) < 2:
            continue
        center_time = source_time[window.center]
        center_index = int(np.argmin(np.abs(state_time - center_time)))
        noncenter = state_indices[state_indices != center_index]
        if len(noncenter) == 0:
            continue
        weight_scale = 1.0 / np.sqrt(len(noncenter))
        for sample_index in noncenter:
            displacement = float(
                np.interp(
                    state_time[sample_index],
                    window_time,
                    window.displacement_mm,
                )
            )
            key = (window_number, center_index, int(sample_index))
            if key in seen:
                continue
            seen.add(key)
            factors.append(
                RelativeTravelFactor(
                    center_index=center_index,
                    sample_index=int(sample_index),
                    displacement_mm=displacement,
                    weight_scale=weight_scale,
                )
            )
    return factors


def metric(
    log: str,
    method: str,
    alpha: float,
    region: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | str]:
    error = prediction[mask] - truth[mask]
    return {
        "log": log,
        "fork": FORK.get(log, "unknown"),
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
    field_weights: MagSolverWeights,
) -> tuple[list[dict], dict]:
    signals = load_signals(
        name,
        args.cache_root,
        args.state_hz,
        args.degree,
        args.travel_max_mm,
    )
    tangent_windows = build_accel_windows(
        signals,
        chunk_radius=args.chunk_radius,
        min_dx_mm=args.chunk_min_dx_mm,
        max_dx_mm=args.chunk_max_dx_mm,
        min_abs_corr=args.chunk_min_abs_corr,
        max_angle_deg=args.chunk_max_angle_deg,
        weak_threshold_mg=args.weak_threshold_mg,
    )
    factor_windows = build_accel_windows(
        signals,
        chunk_radius=args.factor_radius,
        min_dx_mm=args.factor_min_dx_mm,
        max_dx_mm=args.factor_max_dx_mm,
        min_abs_corr=args.factor_min_abs_corr,
        max_angle_deg=args.factor_max_angle_deg,
        weak_threshold_mg=args.weak_threshold_mg,
    )
    anchor_mask, _ = high_confidence_mask(
        signals,
        signals.initial_travel,
        args.accel_anchor_norm_mg,
        0.0,
        args.agreement_mm,
    )
    anchor_expected = signals.xyz_model.predict(signals.initial_travel)
    body, world = smooth_body_world_fields(
        signals.time_s,
        signals.gyro_dps,
        signals.mag_xyz,
        anchor_expected,
        anchor_mask,
        field_weights,
    )
    accel_fit = fit_accel_tangent(
        signals,
        body + world,
        tangent_windows,
        travel_bin_mm=args.chunk_travel_bin_mm,
    )
    if accel_fit is None:
        raise ValueError(f"{name}: not enough acceleration windows for tangent fit")
    tangent_fn = blend_tangent_fn(signals, accel_fit, args.tangent_blend)
    factors = relative_factors_from_windows(
        signals.time_s, signals.source_time, factor_windows
    )
    if args.disable_accel_factors:
        factors = []

    joint_weights = JointSolverWeights(
        field=field_weights,
        tangent_sigma_ratio=args.tangent_sigma_ratio,
        accel_sigma_mm=args.accel_sigma_mm,
        mag_huber_sigma=args.mag_huber_sigma,
        travel_prior_sigma_mm=args.travel_prior_sigma_mm,
        travel_prior_stride_s=args.travel_prior_stride_s,
        travel_correction_rw=args.travel_correction_rw,
        travel_min_mm=0.0,
        travel_max_mm=args.travel_max_mm,
        max_travel_step_mm=args.max_travel_step_mm,
        max_field_step_mg=args.max_field_step_mg,
    )
    joint = solve_joint_mag_accel(
        signals.time_s,
        signals.gyro_dps,
        signals.mag_xyz,
        signals.initial_travel,
        signals.xyz_model,
        tangent_fn,
        factors,
        joint_weights,
        travel_prior_mask=(
            anchor_mask if args.travel_prior_mode == "anchors" else None
        ),
        iterations=args.iterations,
        damping=args.damping,
        lsmr_maxiter=args.lsmr_maxiter,
        lsmr_tolerance=args.lsmr_tolerance,
    )

    previous = solve_iterative_correction(
        signals.time_s,
        signals.gyro_dps,
        signals.mag_xyz,
        signals.initial_travel,
        signals.xyz_model,
        field_weights,
        iterations=4,
    )
    tangent_initial = tangent_fn(signals.initial_travel)
    normal_covariances = anisotropic_covariances(
        tangent_initial,
        np.zeros(len(signals.time_s), dtype=bool),
        field_weights.mag_sigma,
        args.tangent_sigma_ratio,
    )
    normal_body, normal_world = smooth_body_world_fields(
        signals.time_s,
        signals.gyro_dps,
        signals.mag_xyz,
        signals.xyz_model.predict(signals.initial_travel),
        np.ones(len(signals.time_s), dtype=bool),
        field_weights,
        measurement_covariances=normal_covariances,
    )
    normal_weak, normal_all = make_proposals(
        signals,
        signals.xyz_model,
        normal_body + normal_world,
        args.weak_threshold_mg,
    )
    measured_weak = (
        np.linalg.norm(signals.mag_xyz, axis=1) < args.weak_threshold_mg
    )
    corrected_xyz = signals.mag_xyz - joint.correction
    corrected_xyz_travel = signals.xyz_model.infer(corrected_xyz)
    alignment_mask = anchor_mask & ~measured_weak
    if np.sum(alignment_mask) < 3:
        alignment_mask = ~measured_weak
    if np.sum(alignment_mask) < 3:
        alignment_mask = np.ones(len(signals.time_s), dtype=bool)
    latent_offset = float(
        np.median(
            signals.initial_travel[alignment_mask] - joint.travel[alignment_mask]
        )
    )
    xyz_offset = float(
        np.median(
            signals.initial_travel[alignment_mask]
            - corrected_xyz_travel[alignment_mask]
        )
    )
    joint_aligned = np.clip(
        joint.travel + latent_offset, 0.0, args.travel_max_mm
    )
    corrected_xyz_aligned = np.clip(
        corrected_xyz_travel + xyz_offset, 0.0, args.travel_max_mm
    )
    joint_weak = signals.initial_travel.copy()
    joint_weak[measured_weak] = joint.travel[measured_weak]
    joint_aligned_weak = signals.initial_travel.copy()
    joint_aligned_weak[measured_weak] = joint_aligned[measured_weak]
    joint_xyz_weak = signals.initial_travel.copy()
    joint_xyz_weak[measured_weak] = corrected_xyz_travel[measured_weak]
    joint_xyz_aligned_weak = signals.initial_travel.copy()
    joint_xyz_aligned_weak[measured_weak] = corrected_xyz_aligned[measured_weak]
    proposals = {
        "previous_iterative": previous.travel,
        "curve_normal_weak": normal_weak,
        "curve_normal_all": normal_all,
        "joint_latent_weak": joint_weak,
        "joint_latent_all": joint.travel,
        "joint_latent_aligned_weak": joint_aligned_weak,
        "joint_latent_aligned_all": joint_aligned,
        "joint_corrected_xyz_weak": joint_xyz_weak,
        "joint_corrected_xyz_all": corrected_xyz_travel,
        "joint_corrected_xyz_aligned_weak": joint_xyz_aligned_weak,
        "joint_corrected_xyz_aligned_all": corrected_xyz_aligned,
    }

    # Encoder ground truth first appears here.
    truth = flatten(aligned(signals.cache, "travel", signals.state_index))
    active = np.isfinite(truth) & (truth >= 0.0)
    if "boring_mask" in signals.cache:
        active &= np.asarray(signals.cache["boring_mask"], dtype=bool)[
            signals.state_index
        ]
    regions = {
        "all": active,
        "weak": active & measured_weak,
        "strong": active & ~measured_weak,
    }
    metrics: list[dict] = []
    for method, proposal in proposals.items():
        delta = proposal - signals.initial_travel
        for alpha in args.output_alphas:
            prediction = signals.initial_travel + alpha * delta
            for region, mask in regions.items():
                metrics.append(
                    metric(name, method, alpha, region, prediction, truth, mask)
                )
    for region, mask in regions.items():
        metrics.append(
            metric(
                name,
                "pipeline",
                0.0,
                region,
                signals.initial_travel,
                truth,
                mask,
            )
        )
    details = {
        "log": name,
        "fork": FORK.get(name, "unknown"),
        "samples": len(signals.time_s),
        "tangent_windows": len(tangent_windows),
        "factor_windows": len(factor_windows),
        "accel_factors": len(factors),
        "accel_tangent": asdict(accel_fit),
        "joint_iterations": joint.iteration_diagnostics,
        "joint_body_norm_median_mg": float(
            np.median(np.linalg.norm(joint.body_field, axis=1))
        ),
        "joint_world_norm_median_mg": float(
            np.median(np.linalg.norm(joint.world_field, axis=1))
        ),
        "joint_correction_norm_median_mg": float(
            np.median(np.linalg.norm(joint.correction, axis=1))
        ),
        "joint_latent_alignment_offset_mm": latent_offset,
        "joint_xyz_alignment_offset_mm": xyz_offset,
        "alignment_samples": int(np.sum(alignment_mask)),
    }
    return metrics, details


def json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", default=list(DEFAULT_LOGS))
    parser.add_argument(
        "--cache-root", type=Path, default=REPO_ROOT / "backend/run_artifacts"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports/front_mag_nuisance/joint_latent",
    )
    parser.add_argument("--state-hz", type=float, default=10.0)
    parser.add_argument("--degree", type=int, choices=(1, 2), default=2)
    parser.add_argument("--travel-max-mm", type=float, default=210.0)
    parser.add_argument("--weak-threshold-mg", type=float, default=1500.0)
    parser.add_argument("--accel-anchor-norm-mg", type=float, default=1500.0)
    parser.add_argument("--agreement-mm", type=float, default=20.0)
    parser.add_argument("--chunk-radius", type=int, default=20)
    parser.add_argument("--chunk-min-dx-mm", type=float, default=10.0)
    parser.add_argument("--chunk-max-dx-mm", type=float, default=150.0)
    parser.add_argument("--chunk-min-abs-corr", type=float, default=0.5)
    parser.add_argument("--chunk-max-angle-deg", type=float, default=15.0)
    parser.add_argument("--chunk-travel-bin-mm", type=float, default=10.0)
    parser.add_argument("--factor-radius", type=int, default=15)
    parser.add_argument("--factor-min-dx-mm", type=float, default=5.0)
    parser.add_argument("--factor-max-dx-mm", type=float, default=150.0)
    parser.add_argument("--factor-min-abs-corr", type=float, default=0.9)
    parser.add_argument("--factor-max-angle-deg", type=float, default=10.0)
    parser.add_argument("--tangent-blend", type=float, default=0.9)
    parser.add_argument("--tangent-sigma-ratio", type=float, default=5.0)
    parser.add_argument("--accel-sigma-mm", type=float, default=5.0)
    parser.add_argument("--disable-accel-factors", action="store_true")
    parser.add_argument("--mag-huber-sigma", type=float, default=4.0)
    parser.add_argument("--travel-prior-sigma-mm", type=float, default=1.0)
    parser.add_argument("--travel-prior-stride-s", type=float, default=0.1)
    parser.add_argument(
        "--travel-prior-mode", choices=("anchors", "all"), default="anchors"
    )
    parser.add_argument("--travel-correction-rw", type=float, default=0.25)
    parser.add_argument("--max-travel-step-mm", type=float, default=30.0)
    parser.add_argument("--max-field-step-mg", type=float, default=600.0)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--lsmr-maxiter", type=int, default=2000)
    parser.add_argument("--lsmr-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--output-alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.tangent_blend <= 1.0:
        raise ValueError("tangent-blend must be between zero and one")
    if any(not 0.0 <= alpha <= 1.0 for alpha in args.output_alphas):
        raise ValueError("output-alphas must be between zero and one")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    field_weights = MagSolverWeights()
    metrics: list[dict] = []
    details: list[dict] = []
    for name in args.logs:
        print(f"Evaluating {name}...", flush=True)
        log_metrics, log_details = evaluate_log(name, args, field_weights)
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
        "arguments": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "field_weights": asdict(field_weights),
        "logs": details,
    }
    (args.output_dir / "details.json").write_text(
        json.dumps(payload, indent=2, default=json_default) + "\n",
        encoding="utf-8",
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
