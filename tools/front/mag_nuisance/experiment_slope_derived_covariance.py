#!/usr/bin/env python3
"""Test slope-derived magnetometer covariance for nuisance-field correction.

Two encoder-blind families are evaluated:

1. Iterative correction with the existing weak-field observation/application
   gates and travel uncertainty propagated along the local XYZ-curve tangent.
2. A single all-sample curve-normal field solve, followed by weak-only travel
   application. Optional slope-based normal inflation tests whether high-slope
   curve-normal residuals also deserve less weight.

Encoder travel is loaded only after all predictions have been generated and is
used solely for metrics.
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
    REPO_ROOT,
    load_signals,
    path_derivative,
    score,
)
from tools.front.mag_nuisance.experiment_unsupervised_mag_xyz import (  # noqa: E402
    DEFAULT_LOGS,
    aligned,
    flatten,
)
from tools.front.mag_nuisance.mag_correction_solver import (  # noqa: E402
    MagSolverWeights,
    curve_slope_covariances,
    smooth_body_world_fields,
    solve_iterative_correction,
)


def value_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def iterative_name(iterations: int, travel_sigma_mm: float) -> str:
    return f"iter{iterations}_travel_sigma{value_tag(travel_sigma_mm)}"


def all_sample_name(travel_sigma_mm: float, normal_fraction: float) -> str:
    return (
        f"all_curve_travel_sigma{value_tag(travel_sigma_mm)}"
        f"_normal_fraction{value_tag(normal_fraction)}"
    )


def weak_application(
    signals: object,
    inferred: np.ndarray,
    weak_threshold_mg: float,
) -> np.ndarray:
    predicted_weak = np.linalg.norm(
        signals.xyz_model.predict(signals.initial_travel), axis=1
    ) <= weak_threshold_mg
    measured_weak = np.linalg.norm(signals.mag_xyz, axis=1) <= weak_threshold_mg
    mask = (
        predicted_weak
        & measured_weak
        & signals.xyz_model.covers(signals.initial_travel)
    )
    proposal = signals.initial_travel.copy()
    proposal[mask] = inferred[mask]
    return proposal


def evaluate_log(
    name: str,
    args: argparse.Namespace,
    weights: MagSolverWeights,
) -> tuple[list[dict[str, float | int | str]], dict[str, object]]:
    signals = load_signals(
        name,
        args.cache_root,
        args.state_hz,
        args.degree,
        args.travel_max_mm,
    )
    candidates: dict[str, tuple[np.ndarray, str, int, float, float]] = {}
    iteration_changes: dict[str, list[float]] = {}

    for iterations in args.iteration_counts:
        for travel_sigma in args.travel_sigmas_mm:
            result = solve_iterative_correction(
                signals.time_s,
                signals.gyro_dps,
                signals.mag_xyz,
                signals.initial_travel,
                signals.xyz_model,
                weights,
                iterations=iterations,
                travel_sigma_mm=travel_sigma,
            )
            method = iterative_name(iterations, travel_sigma)
            candidates[method] = (
                result.travel,
                "iterative_weak",
                iterations,
                travel_sigma,
                0.0,
            )
            iteration_changes[method] = result.iteration_change_mm

    expected = signals.xyz_model.predict(signals.initial_travel)
    all_measurements = np.ones(len(signals.time_s), dtype=bool)
    all_sample_settings = [
        (travel_sigma, 0.0) for travel_sigma in args.travel_sigmas_mm
    ] + [
        (args.normal_fraction_travel_sigma_mm, fraction)
        for fraction in args.normal_slope_fractions
        if fraction != 0.0
    ]
    for travel_sigma, normal_fraction in all_sample_settings:
        covariances = (
            None
            if travel_sigma == 0.0
            else curve_slope_covariances(
                signals.xyz_model,
                signals.initial_travel,
                weights.mag_sigma,
                travel_sigma,
                normal_slope_fraction=normal_fraction,
            )
        )
        body, world = smooth_body_world_fields(
            signals.time_s,
            signals.gyro_dps,
            signals.mag_xyz,
            expected,
            all_measurements,
            weights,
            measurement_covariances=covariances,
        )
        corrected = signals.mag_xyz - body - world
        inferred = signals.xyz_model.infer(corrected)
        proposal = weak_application(signals, inferred, args.weak_threshold_mg)
        method = all_sample_name(travel_sigma, normal_fraction)
        candidates[method] = (
            proposal,
            "all_sample_curve",
            1,
            travel_sigma,
            normal_fraction,
        )

    # Encoder ground truth first appears here. Everything above is deployable
    # without it.
    truth = flatten(aligned(signals.cache, "travel", signals.state_index))
    active = np.isfinite(truth) & (truth >= 0.0)
    if "boring_mask" in signals.cache:
        active &= np.asarray(signals.cache["boring_mask"], dtype=bool)[
            signals.state_index
        ]
    measured_weak = np.linalg.norm(signals.mag_xyz, axis=1) < args.weak_threshold_mg
    regions = {
        "all": active,
        "weak": active & measured_weak,
        "strong": active & ~measured_weak,
    }

    metrics: list[dict[str, float | int | str]] = []
    for method, candidate in candidates.items():
        _, family, iterations, travel_sigma, normal_fraction = candidates[method]
        for alpha in args.alphas:
            prediction = signals.initial_travel + alpha * (
                candidate[0] - signals.initial_travel
            )
            for region, mask in regions.items():
                row = score(
                    name,
                    method,
                    alpha,
                    region,
                    prediction,
                    truth,
                    mask,
                )
                row.update(
                    {
                        "family": family,
                        "iterations": iterations,
                        "travel_sigma_mm": travel_sigma,
                        "normal_slope_fraction": normal_fraction,
                    }
                )
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
        row.update(
            {
                "family": "pipeline",
                "iterations": 0,
                "travel_sigma_mm": np.nan,
                "normal_slope_fraction": np.nan,
            }
        )
        metrics.append(row)

    grid = np.asarray(signals.xyz_model.travel_grid, dtype=float)
    xyz_grid = np.asarray(signals.xyz_model.xyz_grid, dtype=float)
    grid_slope = np.linalg.norm(np.gradient(xyz_grid, grid, axis=0), axis=1)
    grid_weak = np.linalg.norm(xyz_grid, axis=1) <= args.weak_threshold_mg
    details = {
        "log": name,
        "samples": len(signals.time_s),
        "weak_grid_slope_mg_per_mm": {
            "p10": float(np.percentile(grid_slope[grid_weak], 10)),
            "p50": float(np.percentile(grid_slope[grid_weak], 50)),
            "p90": float(np.percentile(grid_slope[grid_weak], 90)),
        },
        "iteration_change_mm": iteration_changes,
    }
    return metrics, details


def cohort_summary(frame: pd.DataFrame) -> pd.DataFrame:
    weak = frame[frame["region"] == "weak"]
    baseline = weak[weak["method"] == "pipeline"].set_index("log")["rmse_mm"]
    rows: list[dict[str, float | int | str]] = []
    for (method, alpha), group in weak[weak["method"] != "pipeline"].groupby(
        ["method", "alpha"]
    ):
        values = group.set_index("log")["rmse_mm"].reindex(baseline.index)
        delta = values - baseline
        rows.append(
            {
                "method": method,
                "alpha": float(alpha),
                "family": group["family"].iloc[0],
                "iterations": int(group["iterations"].iloc[0]),
                "travel_sigma_mm": float(group["travel_sigma_mm"].iloc[0]),
                "normal_slope_fraction": float(
                    group["normal_slope_fraction"].iloc[0]
                ),
                "mean_rmse_mm": float(values.mean()),
                "median_rmse_mm": float(values.median()),
                "mean_delta_mm": float(delta.mean()),
                "median_delta_mm": float(delta.median()),
                "logs_improved": int((delta < 0.0).sum()),
                "worst_delta_mm": float(delta.max()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["logs_improved", "mean_rmse_mm"], ascending=[False, True]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", default=list(DEFAULT_LOGS))
    parser.add_argument(
        "--cache-root", type=Path, default=REPO_ROOT / "backend/run_artifacts"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "reports/front_mag_nuisance/observability/slope_derived_covariance",
    )
    parser.add_argument("--state-hz", type=float, default=10.0)
    parser.add_argument("--degree", type=int, choices=(1, 2), default=2)
    parser.add_argument("--travel-max-mm", type=float, default=210.0)
    parser.add_argument("--weak-threshold-mg", type=float, default=1500.0)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.5, 0.75, 1.0]
    )
    parser.add_argument(
        "--iteration-counts", type=int, nargs="+", default=[1, 2, 4]
    )
    parser.add_argument(
        "--travel-sigmas-mm",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    parser.add_argument(
        "--normal-slope-fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 1.0],
    )
    parser.add_argument("--normal-fraction-travel-sigma-mm", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(not 0.0 <= value <= 1.0 for value in args.alphas):
        raise ValueError("alphas must be between zero and one")
    if any(value < 1 for value in args.iteration_counts):
        raise ValueError("iteration-counts must all be at least one")
    if any(value < 0.0 or not np.isfinite(value) for value in args.travel_sigmas_mm):
        raise ValueError("travel-sigmas-mm must all be finite and nonnegative")
    if any(not 0.0 <= value <= 1.0 for value in args.normal_slope_fractions):
        raise ValueError("normal-slope-fractions must be between zero and one")
    if args.normal_fraction_travel_sigma_mm < 0.0:
        raise ValueError("normal-fraction-travel-sigma-mm must be nonnegative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = MagSolverWeights(mag_update_threshold=args.weak_threshold_mg)
    metrics: list[dict[str, float | int | str]] = []
    details: list[dict[str, object]] = []
    for name in args.logs:
        print(f"Evaluating {name}...", flush=True)
        log_metrics, log_details = evaluate_log(name, args, weights)
        metrics.extend(log_metrics)
        details.append(log_details)

    frame = pd.DataFrame(metrics)
    frame.to_csv(args.output_dir / "metrics.csv", index=False)
    summary = cohort_summary(frame)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
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
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    print("\nWeak-field cohort summary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nResults: {args.output_dir}")


if __name__ == "__main__":
    main()
