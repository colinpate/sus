#!/usr/bin/env python3
"""Compare stage-one corrected travel with standard pipeline error metrics.

Both the original pipeline solution and corrected travel are evaluated on the
same 10 Hz nuisance-state samples. The mask and five-bin RMSE calculation come
from ``tools/stats_aggregator.py`` so the definitions match regular reports.
Encoder travel is used only after the corrected prediction has been generated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.mag_nuisance_core import (  # noqa: E402
    PRIMARY_MAG_TO_GYRO,
    MagSolverWeights,
    fit_scalar_parameterized_xyz,
    integrate_gyro,
    solve_iterative_correction,
)
from tools.front.mag_nuisance.experiment_unsupervised_mag_xyz import (  # noqa: E402
    DEFAULT_LOGS,
    FORK,
    aligned_first,
    flatten,
)
from tools.stats_aggregator import (  # noqa: E402
    build_angle_bad_mask,
    make_travel_bins,
    summarize_binned_rmse,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def metric_row(
    log_name: str,
    method: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    *,
    centered: bool,
) -> dict[str, float | int | str]:
    prediction = np.asarray(prediction, dtype=float)[mask]
    truth = np.asarray(truth, dtype=float)[mask]
    error = prediction - truth
    if centered:
        error = error - np.mean(error)
    binned = summarize_binned_rmse(error, truth)
    bin_rmses = np.array(
        [binned[f"{bin_spec.key}_rmse"] for bin_spec in make_travel_bins()]
    )
    row: dict[str, float | int | str] = {
        "log": log_name,
        "fork": FORK.get(log_name, "unknown"),
        "method": method,
        "centering": "centered" if centered else "raw",
        "samples": len(error),
        "rmse_mm": float(np.sqrt(np.mean(error**2))),
        "mae_mm": float(np.mean(np.abs(error))),
        "bias_mm": float(np.mean(prediction - truth)),
        "bin_rmse_mm": binned["bin_rmse"],
        "worst_bin_rmse_mm": float(np.nanmax(bin_rmses)),
        "eligible_bins": int(binned["eligible_bins"]),
    }
    for bin_spec in make_travel_bins():
        row[f"{bin_spec.key}_rmse_mm"] = binned[f"{bin_spec.key}_rmse"]
        row[f"{bin_spec.key}_n"] = int(binned[f"{bin_spec.key}_n"])
    return row


def evaluate_log(
    name: str,
    cache_root: Path,
    state_hz: float,
    alpha: float,
    iterations: int,
    weights: MagSolverWeights,
) -> list[dict]:
    cache = np.load(cache_root / name / "cache" / "all.npz")
    time_s = flatten(cache["mag/lpf__t"])
    source_hz = 1.0 / float(np.median(np.diff(time_s)))
    stride = max(1, round(source_hz / state_hz))
    index = np.arange(0, len(time_s), stride)

    mag_xyz = np.asarray(cache["mag/lpf__x"], dtype=float) @ PRIMARY_MAG_TO_GYRO.T
    gyro_dps = np.asarray(cache["gyro/lpf/gyro1__x"], dtype=float)
    scalar_mag = flatten(
        aligned_first(
            cache, index, "mag/norm/corr/lpf", "mag/proj/corr/lpf"
        )
    )
    initial_travel = flatten(cache["travel/solved__x"])[index]
    raw_scalar_travel = flatten(cache["travel/mag_model__x"])
    adjusted_scalar_travel = flatten(cache["travel/mag_model/adj__x"])
    scalar_offset = float(np.median(adjusted_scalar_travel - raw_scalar_travel))
    xyz_model = fit_scalar_parameterized_xyz(
        scalar_mag,
        mag_xyz[index],
        np.asarray(cache["mag_model_coeffs"], dtype=float),
        scalar_offset,
        degree=2,
        travel_max_mm=210.0,
    )
    correction = solve_iterative_correction(
        time_s[index],
        gyro_dps[index],
        mag_xyz[index],
        initial_travel,
        xyz_model,
        weights,
        iterations=iterations,
        body_to_reference_rotations=integrate_gyro(time_s, gyro_dps)[index],
    )
    corrected_travel = initial_travel + alpha * (
        correction.travel - initial_travel
    )

    # Encoder-derived signals first appear here.
    truth = flatten(cache["travel__x"])[index]
    target_time = flatten(cache["travel__t"])[index]
    boring = np.asarray(cache["boring_mask"], dtype=bool).reshape(-1)[index]
    valid = (
        boring
        & np.isfinite(initial_travel)
        & np.isfinite(corrected_travel)
        & np.isfinite(truth)
        & ~build_angle_bad_mask(cache, target_time)
    )

    rows: list[dict] = []
    for method, prediction in (
        ("pipeline", initial_travel),
        ("body_world_corrected", corrected_travel),
    ):
        for centered in (False, True):
            rows.append(
                metric_row(
                    name,
                    method,
                    prediction,
                    truth,
                    valid,
                    centered=centered,
                )
            )
    return rows


def write_report(frame: pd.DataFrame, output_dir: Path) -> None:
    aggregate_rows: list[dict] = []
    for centering in ("raw", "centered"):
        subset = frame[frame["centering"] == centering]
        baseline = subset[subset["method"] == "pipeline"].set_index("log")
        for method, group in subset.groupby("method"):
            values = group.set_index("log")
            rmse_delta = values["rmse_mm"] - baseline["rmse_mm"]
            bin_delta = values["bin_rmse_mm"] - baseline["bin_rmse_mm"]
            aggregate_rows.append(
                {
                    "centering": centering,
                    "method": method,
                    "mean_rmse_mm": float(values["rmse_mm"].mean()),
                    "median_rmse_mm": float(values["rmse_mm"].median()),
                    "mean_rmse_delta_mm": float(rmse_delta.mean()),
                    "rmse_improved_logs": int(np.sum(rmse_delta < 0)),
                    "mean_bin_rmse_mm": float(values["bin_rmse_mm"].mean()),
                    "median_bin_rmse_mm": float(values["bin_rmse_mm"].median()),
                    "mean_bin_delta_mm": float(bin_delta.mean()),
                    "bin_improved_logs": int(np.sum(bin_delta < 0)),
                }
            )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output_dir / "aggregate.csv", index=False)

    lines = [
        "# Standard metrics for stage-one magnetic correction",
        "",
        "Both methods are evaluated on identical 10 Hz samples using the regular",
        "boring mask, finite-value checks, and angle-corruption exclusion. Standard",
        "`bin_rmse` is the equal-weight RMS over eligible 0--30, 30--60, 60--90,",
        "90--120, and 120--150 mm bins; a bin needs at least 100 sampled points.",
        "",
        "| Centering | Method | Overall RMSE | Bin RMSE | RMSE wins | Bin wins |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    log_count = frame["log"].nunique()
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.centering} | `{row.method}` | {row.mean_rmse_mm:.3f} | "
            f"{row.mean_bin_rmse_mm:.3f} | {row.rmse_improved_logs}/{log_count} | "
            f"{row.bin_improved_logs}/{log_count} |"
        )

    for centering in ("raw", "centered"):
        subset = frame[frame["centering"] == centering]
        baseline = subset[subset["method"] == "pipeline"].set_index("log")
        corrected = subset[
            subset["method"] == "body_world_corrected"
        ].set_index("log")
        lines.extend(
            [
                "",
                f"## Per-log {centering} errors",
                "",
                "| Log | Pipeline RMSE | Corrected RMSE | Delta | Pipeline bin RMSE | Corrected bin RMSE | Delta |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for log_name in baseline.index:
            base = baseline.loc[log_name]
            corr = corrected.loc[log_name]
            lines.append(
                f"| `{log_name}` | {base.rmse_mm:.3f} | {corr.rmse_mm:.3f} | "
                f"{corr.rmse_mm - base.rmse_mm:+.3f} | {base.bin_rmse_mm:.3f} | "
                f"{corr.bin_rmse_mm:.3f} | "
                f"{corr.bin_rmse_mm - base.bin_rmse_mm:+.3f} |"
            )
    lines.extend(
        [
            "",
            "`per_log.csv` contains the five individual travel-bin RMSE values and",
            "sample counts for every log. Aggregate values above are means of the",
            "per-log metrics, matching the way the experiment's weak RMSE is summarized.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", default=list(DEFAULT_LOGS))
    parser.add_argument(
        "--cache-root", type=Path, default=REPO_ROOT / "backend/run_artifacts"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "reports/front_mag_nuisance/observability/standard_correction_metrics"
        ),
    )
    parser.add_argument("--state-hz", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.state_hz <= 0 or args.iterations < 1:
        raise ValueError("state rate must be positive and iterations at least one")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    weights = MagSolverWeights()
    for name in args.logs:
        print(f"Evaluating {name}...", flush=True)
        rows.extend(
            evaluate_log(
                name,
                args.cache_root,
                args.state_hz,
                args.alpha,
                args.iterations,
                weights,
            )
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "per_log.csv", index=False)
    write_report(frame, args.output_dir)
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
