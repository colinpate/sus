#!/usr/bin/env python3
"""Compare full-rate magnetic nuisance outputs with the pipeline baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.front.mag_nuisance.evaluate_correction_standard_metrics import (  # noqa: E402
    metric_row,
)
from tools.front.mag_nuisance.experiment_unsupervised_mag_xyz import (  # noqa: E402
    DEFAULT_LOGS,
    flatten,
)
from tools.stats_aggregator import build_angle_bad_mask  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
METHOD_KEYS = {
    "pipeline": "travel/solved__x",
    "delta_lifted": "travel/solved/mag_nuisance/delta_lifted__x",
    "corrected_mag_observation": "travel/mag_nuisance/corrected__x",
    "fusion2": "travel/solved/mag_nuisance/fusion2__x",
}


def evaluate_log(name: str, cache_root: Path) -> list[dict]:
    cache = np.load(cache_root / name / "cache" / "all.npz")
    truth = flatten(cache["travel__x"])
    truth_time = flatten(cache["travel__t"])
    predictions = {
        method: flatten(cache[key]) for method, key in METHOD_KEYS.items()
    }
    boring = np.asarray(cache["boring_mask"], dtype=bool).reshape(-1)
    valid = boring & np.isfinite(truth) & ~build_angle_bad_mask(cache, truth_time)
    for prediction in predictions.values():
        valid &= np.isfinite(prediction)

    rows: list[dict] = []
    for method, prediction in predictions.items():
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


def evaluate_high_frequency_dynamics(
    name: str,
    cache_root: Path,
    cutoff_hz: float = 5.0,
) -> dict[str, float | str]:
    cache = np.load(cache_root / name / "cache" / "all.npz")
    truth = flatten(cache["travel__x"])
    truth_time = flatten(cache["travel__t"])
    baseline = flatten(cache[METHOD_KEYS["pipeline"]])
    lifted = flatten(cache[METHOD_KEYS["delta_lifted"]])
    fusion2 = flatten(cache[METHOD_KEYS["fusion2"]])
    fs_hz = 1.0 / float(np.median(np.diff(truth_time)))
    sos = butter(4, cutoff_hz, btype="highpass", fs=fs_hz, output="sos")

    def highpass(values: np.ndarray) -> np.ndarray:
        return sosfiltfilt(sos, np.asarray(values, dtype=float))

    truth_hp = highpass(truth)
    baseline_hp = highpass(baseline)
    lifted_hp = highpass(lifted)
    fusion2_hp = highpass(fusion2)
    valid = (
        np.asarray(cache["boring_mask"], dtype=bool).reshape(-1)
        & np.isfinite(truth)
        & ~build_angle_bad_mask(cache, truth_time)
    )

    def rms(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.asarray(values)[valid] ** 2)))

    return {
        "log": name,
        "cutoff_hz": cutoff_hz,
        "baseline_hp_rms_mm": rms(baseline_hp),
        "delta_lifted_change_hp_rms_mm": rms(lifted_hp - baseline_hp),
        "fusion2_change_hp_rms_mm": rms(fusion2_hp - baseline_hp),
        "baseline_hp_error_mm": rms(baseline_hp - truth_hp),
        "delta_lifted_hp_error_mm": rms(lifted_hp - truth_hp),
        "fusion2_hp_error_mm": rms(fusion2_hp - truth_hp),
    }


def write_report(
    frame: pd.DataFrame,
    frequency_frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    aggregate_rows: list[dict] = []
    for centering in ("raw", "centered"):
        subset = frame[frame["centering"] == centering]
        baseline = subset[subset["method"] == "pipeline"].set_index("log")
        for method, group in subset.groupby("method", sort=False):
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
                    "rmse_improved_logs": int(np.sum(rmse_delta < 0.0)),
                    "mean_bin_rmse_mm": float(values["bin_rmse_mm"].mean()),
                    "median_bin_rmse_mm": float(values["bin_rmse_mm"].median()),
                    "mean_bin_delta_mm": float(bin_delta.mean()),
                    "bin_improved_logs": int(np.sum(bin_delta < 0.0)),
                }
            )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output_dir / "aggregate.csv", index=False)

    log_count = frame["log"].nunique()
    lines = [
        "# Full-rate magnetic nuisance correction",
        "",
        "All methods use identical full-rate samples, the pipeline boring mask,",
        "finite-value checks, and the standard angle-corruption exclusion. The",
        "aggregate values are arithmetic means of per-log metrics.",
        "",
        "| Centering | Method | Overall RMSE | Delta | Bin RMSE | Delta | RMSE wins | Bin wins |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.centering} | `{row.method}` | {row.mean_rmse_mm:.3f} | "
            f"{row.mean_rmse_delta_mm:+.3f} | {row.mean_bin_rmse_mm:.3f} | "
            f"{row.mean_bin_delta_mm:+.3f} | {row.rmse_improved_logs}/{log_count} | "
            f"{row.bin_improved_logs}/{log_count} |"
        )

    for centering in ("raw", "centered"):
        subset = frame[frame["centering"] == centering]
        pivot_rmse = subset.pivot(index="log", columns="method", values="rmse_mm")
        pivot_bin = subset.pivot(index="log", columns="method", values="bin_rmse_mm")
        lines.extend(
            [
                "",
                f"## Per-log {centering} errors",
                "",
                "| Log | Base RMSE | Delta-lift RMSE | Fusion2 RMSE | Base bin | Delta-lift bin | Fusion2 bin |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for log_name in pivot_rmse.index:
            lines.append(
                f"| `{log_name}` | {pivot_rmse.loc[log_name, 'pipeline']:.3f} | "
                f"{pivot_rmse.loc[log_name, 'delta_lifted']:.3f} | "
                f"{pivot_rmse.loc[log_name, 'fusion2']:.3f} | "
                f"{pivot_bin.loc[log_name, 'pipeline']:.3f} | "
                f"{pivot_bin.loc[log_name, 'delta_lifted']:.3f} | "
                f"{pivot_bin.loc[log_name, 'fusion2']:.3f} |"
            )
    lines.extend(
        [
            "",
            "`per_log.csv` contains individual travel-bin errors and sample counts,",
            "including the corrected magnetometer observation before refusion.",
            "",
            "## High-frequency preservation",
            "",
            "After a 5 Hz high-pass, the delta lift changes the baseline travel by",
            f"{frequency_frame['delta_lifted_change_hp_rms_mm'].mean():.3f} mm RMS on average. "
            "The second fusion pass changes it by "
            f"{frequency_frame['fusion2_change_hp_rms_mm'].mean():.3f} mm RMS.",
            "The mean high-frequency error versus the encoder is "
            f"{frequency_frame['baseline_hp_error_mm'].mean():.3f} mm for the baseline, "
            f"{frequency_frame['delta_lifted_hp_error_mm'].mean():.3f} mm after delta lifting, "
            f"and {frequency_frame['fusion2_hp_error_mm'].mean():.3f} mm after refusion.",
            "`frequency_dynamics.csv` contains the per-log values.",
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
            / "reports/front_mag_nuisance/observability/full_rate_correction"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    frequency_rows: list[dict] = []
    for name in args.logs:
        print(f"Evaluating {name}...", flush=True)
        rows.extend(evaluate_log(name, args.cache_root))
        frequency_rows.append(
            evaluate_high_frequency_dynamics(name, args.cache_root)
        )
    frame = pd.DataFrame(rows)
    frequency_frame = pd.DataFrame(frequency_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "per_log.csv", index=False)
    frequency_frame.to_csv(
        args.output_dir / "frequency_dynamics.csv", index=False
    )
    write_report(frame, frequency_frame, args.output_dir)
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
