#!/usr/bin/env python3
"""Isolate iteration count from tangent weighting in weak-field correction.

Every candidate uses the same pipeline travel, encoder-blind XYZ model,
body/world dynamics, weak-field observation mask, weak-field application mask,
and output alpha. Only the number of outer travel/field iterations and the
measurement sigma along the local XYZ-curve tangent change.

Encoder travel is loaded only after all candidate predictions are generated
and is used solely for metrics.
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
    score,
)
from tools.front.mag_nuisance.experiment_unsupervised_mag_xyz import (  # noqa: E402
    DEFAULT_LOGS,
    aligned,
    flatten,
)
from backend.mag_nuisance_core import (  # noqa: E402
    MagSolverWeights,
    solve_iterative_correction,
)


def method_name(iterations: int, tangent_sigma_ratio: float) -> str:
    return f"iter{iterations}_tangent_ratio{tangent_sigma_ratio:g}"


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

    predictions: dict[tuple[int, float], np.ndarray] = {}
    iteration_changes: dict[str, list[float]] = {}
    for iterations in args.iteration_counts:
        for ratio in args.tangent_sigma_ratios:
            result = solve_iterative_correction(
                signals.time_s,
                signals.gyro_dps,
                signals.mag_xyz,
                signals.initial_travel,
                signals.xyz_model,
                weights,
                iterations=iterations,
                tangent_sigma_ratio=ratio,
            )
            predictions[(iterations, ratio)] = (
                signals.initial_travel
                + args.alpha * (result.travel - signals.initial_travel)
            )
            iteration_changes[method_name(iterations, ratio)] = (
                result.iteration_change_mm
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
    for (iterations, ratio), prediction in predictions.items():
        method = method_name(iterations, ratio)
        for region, mask in regions.items():
            row = score(
                name,
                method,
                args.alpha,
                region,
                prediction,
                truth,
                mask,
            )
            row["iterations"] = iterations
            row["tangent_sigma_ratio"] = ratio
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
        row["iterations"] = 0
        row["tangent_sigma_ratio"] = np.nan
        metrics.append(row)

    details = {
        "log": name,
        "samples": len(signals.time_s),
        "duration_s": float(signals.time_s[-1] - signals.time_s[0]),
        "iteration_change_mm": iteration_changes,
    }
    return metrics, details


def cohort_summary(frame: pd.DataFrame) -> pd.DataFrame:
    weak = frame[frame["region"] == "weak"].copy()
    baseline = weak[weak["method"] == "pipeline"].set_index("log")["rmse_mm"]
    rows: list[dict[str, float | int | str]] = []
    for method, group in weak[weak["method"] != "pipeline"].groupby("method"):
        values = group.set_index("log")["rmse_mm"].reindex(baseline.index)
        delta = values - baseline
        rows.append(
            {
                "method": method,
                "iterations": int(group["iterations"].iloc[0]),
                "tangent_sigma_ratio": float(
                    group["tangent_sigma_ratio"].iloc[0]
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
        / "reports/front_mag_nuisance/observability/iterative_tangent_ablation",
    )
    parser.add_argument("--state-hz", type=float, default=10.0)
    parser.add_argument("--degree", type=int, choices=(1, 2), default=2)
    parser.add_argument("--travel-max-mm", type=float, default=210.0)
    parser.add_argument("--weak-threshold-mg", type=float, default=1500.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument(
        "--iteration-counts", type=int, nargs="+", default=[1, 2, 4]
    )
    parser.add_argument(
        "--tangent-sigma-ratios",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 5.0, 10.0, 100.0],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    if any(value < 1 for value in args.iteration_counts):
        raise ValueError("iteration-counts must all be at least one")
    if any(
        value < 1.0 or not np.isfinite(value)
        for value in args.tangent_sigma_ratios
    ):
        raise ValueError(
            "tangent-sigma-ratios must all be finite and at least one"
        )

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
