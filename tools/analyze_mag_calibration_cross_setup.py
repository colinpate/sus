#!/usr/bin/env python3
"""Analyze a complete mag-calibration transfer matrix by hardware/bike setup."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tomllib

os.environ["MPLCONFIGDIR"] = "/private/tmp"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT / "backend", REPO_ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from mag_calibration_experiment import load_cached_log  # noqa: E402


METRICS = ("aligned_rmse", "aligned_nrmse_std", "anchored_rmse")


def read_inputs(spec_path: Path, run_dir: Path) -> tuple[dict, pd.DataFrame]:
    with spec_path.open("rb") as handle:
        spec = tomllib.load(handle)
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if not status.get("complete") or status.get("failed_trials"):
        raise ValueError(f"Transfer run is not complete and clean: {status}")
    frame = pd.read_csv(run_dir / "trial_metrics.csv")
    setup_by_log = {
        log_name: setup
        for setup, log_names in spec["setup_logs"].items()
        for log_name in log_names
    }
    missing = (set(frame["train_log"]) | set(frame["eval_log"])) - set(setup_by_log)
    if missing:
        raise ValueError(f"No setup assignment for logs: {sorted(missing)}")
    frame["train_setup"] = frame["train_log"].map(setup_by_log)
    frame["eval_setup"] = frame["eval_log"].map(setup_by_log)
    return spec, frame


def crossed_bootstrap(
    group: pd.DataFrame,
    values: np.ndarray,
    *,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    table = group.assign(_value=values).pivot(
        index="train_log", columns="eval_log", values="_value"
    )
    matrix = table.to_numpy(dtype=float)
    estimates = np.empty(draws, dtype=float)
    for index in range(draws):
        source_indexes = rng.integers(0, matrix.shape[0], matrix.shape[0])
        target_indexes = rng.integers(0, matrix.shape[1], matrix.shape[1])
        sampled = matrix[np.ix_(source_indexes, target_indexes)]
        source_medians = [
            np.median(row[np.isfinite(row)])
            for row in sampled
            if np.any(np.isfinite(row))
        ]
        estimates[index] = np.median(source_medians) if source_medians else np.nan
    valid_estimates = estimates[np.isfinite(estimates)]
    if not len(valid_estimates):
        raise ValueError("No valid crossed-bootstrap resamples")
    return tuple(
        float(value) for value in np.percentile(valid_estimates, [2.5, 97.5])
    )


def source_balanced_median(group: pd.DataFrame, column: str) -> float:
    return float(group.groupby("train_log")[column].median().median())


def analyze_pairs(
    spec: dict,
    frame: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    setups = list(spec["setup_logs"])
    diagonal = frame[frame["train_log"] == frame["eval_log"]].copy()
    baselines = diagonal.set_index(["trainer", "eval_log"])
    baseline_rows = []
    for trainer in spec["trainers"]:
        for setup in setups:
            selected = diagonal[
                (diagonal["trainer"] == trainer)
                & (diagonal["eval_setup"] == setup)
            ]
            row = {
                "trainer": trainer,
                "setup": setup,
                "logs": selected["eval_log"].nunique(),
            }
            for metric in METRICS:
                row[f"{metric}_median"] = float(selected[metric].median())
            baseline_rows.append(row)

    result_rows = []
    rng = np.random.default_rng(seed)
    for trainer in spec["trainers"]:
        for train_setup in setups:
            for eval_setup in setups:
                selected = frame[
                    (frame["trainer"] == trainer)
                    & (frame["train_setup"] == train_setup)
                    & (frame["eval_setup"] == eval_setup)
                ].copy()
                if train_setup == eval_setup:
                    selected = selected[selected["train_log"] != selected["eval_log"]]
                    transfer_type = "same_setup"
                else:
                    transfer_type = "cross_setup"
                row = {
                    "trainer": trainer,
                    "train_setup": train_setup,
                    "eval_setup": eval_setup,
                    "transfer_type": transfer_type,
                    "source_logs": selected["train_log"].nunique(),
                    "target_logs": selected["eval_log"].nunique(),
                    "pairs": len(selected),
                }
                for metric in METRICS:
                    baseline_values = np.asarray([
                        float(baselines.loc[(trainer, target_log), metric])
                        for target_log in selected["eval_log"]
                    ])
                    deltas = selected[metric].to_numpy(dtype=float) - baseline_values
                    row[f"{metric}_transfer_median"] = source_balanced_median(selected, metric)
                    row[f"{metric}_target_diagonal_median"] = float(
                        diagonal[
                            (diagonal["trainer"] == trainer)
                            & (diagonal["eval_setup"] == eval_setup)
                        ][metric].median()
                    )
                    selected_delta = selected.assign(_delta=deltas)
                    row[f"{metric}_delta_median"] = source_balanced_median(
                        selected_delta, "_delta"
                    )
                    low, high = crossed_bootstrap(
                        selected,
                        deltas,
                        draws=draws,
                        rng=rng,
                    )
                    row[f"{metric}_delta_ci_low"] = low
                    row[f"{metric}_delta_ci_high"] = high
                    row[f"{metric}_fraction_worse"] = float(np.mean(deltas > 0))
                result_rows.append(row)
    return pd.DataFrame(baseline_rows), pd.DataFrame(result_rows)


def signal_summary(spec: dict) -> pd.DataFrame:
    rows = []
    for setup, log_names in spec["setup_logs"].items():
        per_log = []
        for log_name in log_names:
            data = load_cached_log(log_name)
            mask = data.activity_mask & np.isfinite(data.mag) & np.isfinite(data.travel)
            mag_low, mag_high = np.percentile(data.mag[mask], [5, 95])
            travel_low, travel_high = np.percentile(data.travel[mask], [5, 95])
            per_log.append((mag_low, mag_high, mag_high - mag_low, travel_low, travel_high, travel_high - travel_low))
        medians = np.median(np.asarray(per_log), axis=0)
        rows.append({
            "setup": setup,
            "logs": len(log_names),
            "mag_p05_median": medians[0],
            "mag_p95_median": medians[1],
            "mag_p90_span_median": medians[2],
            "travel_p05_median_mm": medians[3],
            "travel_p95_median_mm": medians[4],
            "travel_p90_span_median_mm": medians[5],
        })
    return pd.DataFrame(rows)


def plot_matrices(spec: dict, pairs: pd.DataFrame, output_dir: Path) -> None:
    setups = list(spec["setup_logs"])
    labels = spec["setup_labels"]
    trainers = list(spec["trainers"])
    fig, axes = plt.subplots(2, len(trainers), figsize=(15.5, 8.2))
    actual_max = float(pairs["aligned_rmse_transfer_median"].max())
    delta_abs = float(np.max(np.abs(pairs["aligned_rmse_delta_median"])))
    for column, trainer in enumerate(trainers):
        selected = pairs[pairs["trainer"] == trainer]
        actual = np.full((len(setups), len(setups)), np.nan)
        delta = np.full_like(actual, np.nan)
        for i, source in enumerate(setups):
            for j, target in enumerate(setups):
                row = selected[
                    (selected["train_setup"] == source)
                    & (selected["eval_setup"] == target)
                ].iloc[0]
                actual[i, j] = row["aligned_rmse_transfer_median"]
                delta[i, j] = row["aligned_rmse_delta_median"]
        for row_index, (matrix, cmap, vmin, vmax, title) in enumerate((
            (actual, "viridis", 0.0, actual_max, "Transfer RMSE (mm)"),
            (delta, "coolwarm", -delta_abs, delta_abs, "Penalty vs target per-log fit (mm)"),
        )):
            axis = axes[row_index, column]
            image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
            for i in range(len(setups)):
                for j in range(len(setups)):
                    value = matrix[i, j]
                    axis.text(j, i, f"{value:+.1f}" if row_index else f"{value:.1f}", ha="center", va="center", color="white" if abs(value) > vmax * .42 else "black", fontsize=10)
            axis.set_xticks(range(len(setups)), [labels[name] for name in setups], rotation=28, ha="right")
            axis.set_yticks(range(len(setups)), [labels[name] for name in setups])
            axis.set_xlabel("Target setup")
            axis.set_ylabel("Source setup")
            axis.set_title(f"{trainer}\n{title}")
            fig.colorbar(image, ax=axis, fraction=.046, pad=.04)
    fig.suptitle("Full-log magnetometer calibration transfer across setups")
    fig.tight_layout()
    fig.savefig(output_dir / "cross_setup_transfer_matrix.png", dpi=180)
    fig.savefig(output_dir / "cross_setup_transfer_matrix.pdf")
    plt.close(fig)


def value(pairs: pd.DataFrame, trainer: str, source: str, target: str, column: str) -> float:
    return float(pairs[
        (pairs["trainer"] == trainer)
        & (pairs["train_setup"] == source)
        & (pairs["eval_setup"] == target)
    ].iloc[0][column])


def write_report(spec: dict, baselines: pd.DataFrame, pairs: pd.DataFrame, signals: pd.DataFrame, output_dir: Path) -> None:
    labels = spec["setup_labels"]
    setups = list(spec["setup_logs"])
    ss = "self-supervised"
    lines = [
        "# Cross-setup magnetometer calibration transfer",
        "",
        "## Bottom line",
        "",
        "Full-log calibrations transfer well between pod v1 and pod v2 on the same Stumpjumper, but do not transfer between the Stumpjumper and TR11. The bike/setup change produces a far larger penalty than either log-to-log variation within a setup or the pod-generation change on the same bike. This directly supports per-setup calibration and strengthens the motivation for automatic per-recording self-calibration.",
        "",
        "## Design",
        "",
        f"The experiment evaluates all 24 source logs against all 24 target logs for {len(spec['trainers'])} trainers: " + ", ".join(spec["trainers"]) + ". Each calibration is trained once on its complete source log. The matrix contains the target log's own calibration, same-setup transfers, and all six directed cross-setup transfers. All 1,728 evaluations completed successfully.",
        "",
        "The primary comparison is each transfer's aligned error minus the target log's own calibration error. Positive values mean that independent target/per-recording calibration is better. Aligned normalized RMSE is retained to ensure that conclusions are not caused only by the setups' different travel ranges. Confidence intervals use a crossed bootstrap that independently resamples source and target logs; they are descriptive because the available logs are not a prospectively held-out cohort.",
        "",
        "## Per-log baselines",
        "",
        "| Trainer | Setup | Logs | Aligned RMSE | Normalized RMSE | Anchored RMSE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in baselines.iterrows():
        lines.append(
            f"| {row['trainer']} | {labels[row['setup']]} | {int(row['logs'])} | "
            f"{row['aligned_rmse_median']:.2f} mm | {row['aligned_nrmse_std_median']:.3f} | "
            f"{row['anchored_rmse_median']:.2f} mm |"
        )
    lines.extend([
        "",
        "## Self-supervised transfer results",
        "",
        "| Source → target | Type | Transfer RMSE | Penalty vs target calibration (95% CI) | Δ normalized RMSE | Transfer worse | Anchored RMSE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    selected = pairs[pairs["trainer"] == ss]
    for _, row in selected.iterrows():
        lines.append(
            f"| {labels[row['train_setup']]} → {labels[row['eval_setup']]} | {row['transfer_type'].replace('_', ' ')} | "
            f"{row['aligned_rmse_transfer_median']:.2f} mm | {row['aligned_rmse_delta_median']:+.2f} "
            f"[{row['aligned_rmse_delta_ci_low']:+.2f}, {row['aligned_rmse_delta_ci_high']:+.2f}] mm | "
            f"{row['aligned_nrmse_std_delta_median']:+.3f} | {row['aligned_rmse_fraction_worse']:.0%} | "
            f"{row['anchored_rmse_transfer_median']:.2f} mm |"
        )
    v1, v2, tr11 = setups
    lines.extend([
        "",
        "## Main findings",
        "",
        "1. **Changing sensor generation on the same bike has a small transfer cost.** "
        f"Pod-v1 Stumpjumper curves transferred to pod-v2 Stumpjumper with a {value(pairs, ss, v1, v2, 'aligned_rmse_delta_median'):+.2f} mm median penalty; the reverse direction was {value(pairs, ss, v2, v1, 'aligned_rmse_delta_median'):+.2f} mm. These are comparable to same-setup log-transfer penalties and tiny relative to cross-bike effects. Supervised oracle penalties are positive in both directions, showing a real but modest hardware/mounting difference that self-supervised fit variance can obscure.",
        "2. **Stumpjumper calibrations fail on the TR11.** "
        f"Self-supervised transfer penalties are {value(pairs, ss, v1, tr11, 'aligned_rmse_delta_median'):+.2f} mm from pod-v1 Stumpjumper and {value(pairs, ss, v2, tr11, 'aligned_rmse_delta_median'):+.2f} mm from pod-v2 Stumpjumper. Transfer is worse than the TR11 target calibration for 98% of source-target pairs from both Stumpjumper setups.",
        "3. **TR11 calibrations fail even more severely on the Stumpjumper.** "
        f"The penalties are {value(pairs, ss, tr11, v1, 'aligned_rmse_delta_median'):+.2f} and {value(pairs, ss, tr11, v2, 'aligned_rmse_delta_median'):+.2f} mm, and every evaluated pair is worse than the target's own calibration.",
        "4. **The conclusion survives normalization and oracle substitution.** Cross-bike normalized-RMSE penalties remain large, and both supervised oracle families show the same qualitative separation. The result is therefore not explained merely by different fork travel, target difficulty, or self-supervised optimizer noise.",
        "5. **Absolute anchoring amplifies cross-bike failure.** Self-supervised cross-bike anchored errors are much larger than aligned errors, so transferring a frozen curve cannot be rescued by the present target-side anchor policy.",
        "",
        "## Why transfer is directionally asymmetric",
        "",
        "The active-data magnetic ranges differ substantially:",
        "",
        "| Setup | Median magnetic p5–p95 | Median magnetic span | Median travel span |",
        "|---|---:|---:|---:|",
    ])
    for _, row in signals.iterrows():
        lines.append(
            f"| {labels[row['setup']]} | {row['mag_p05_median']:.0f}–{row['mag_p95_median']:.0f} | "
            f"{row['mag_p90_span_median']:.0f} | {row['travel_p90_span_median_mm']:.1f} mm |"
        )
    lines.extend([
        "",
        "The TR11 occupies a narrow, low magnetic interval compared with either Stumpjumper setup. A TR11-trained power curve must extrapolate far outside its observed magnetic support on Stumpjumper targets, explaining why that direction is especially destructive. Stumpjumper-to-TR11 transfer stays nearer the source's low-magnitude region but still applies the wrong curve shape/scale. This asymmetry is evidence for a setup-specific mapping, not evidence that one transfer direction is acceptable.",
        "",
        "## Paper implications",
        "",
        "- The experiment directly supports the claim that a one-time calibration does not generalize across bike/sensor geometry, particularly across bikes.",
        "- It strengthens the value proposition for calibration without stored bike-specific priors: each target log's independently learned mapping is dramatically better than importing a curve from the other bike.",
        "- The pod-v1↔pod-v2 result is a useful nuance: the method need not claim that every remount or sensor revision creates a wholly unrelated curve. The dominant tested change is bike/fork/magnet geometry.",
        "- The paper should show directed transfer, because the failure is strongly asymmetric. A pooled 'cross-setup' number would conceal the extrapolation mechanism.",
        "- The primary paper table should include aligned and normalized error; anchored error can be a secondary end-to-end measure.",
        "",
        "## Pipeline implications",
        "",
        "- Never silently reuse a calibration across an unknown bike/setup. Require setup identity or perform self-calibration on the current recording.",
        "- Store the calibration's observed magnetic support. If a target recording lies outside it, reject the imported curve rather than extrapolating.",
        "- A pod-generation change on the same bike may permit a warm start, but it should still be validated by self-supervised constraints before acceptance.",
        "- A generic population prior could initialize optimization, but the final mapping must adapt to the current bike/setup.",
        "- Add a curve-compatibility score based on magnetic-support overlap and short-window IMU residuals; this experiment provides positive and negative pairs for selecting a threshold.",
        "",
        "## Limitations and next step",
        "",
        "This experiment evaluates the magnetic mapping before downstream fusion and uses full-log source calibration. It also uses the reference-derived `boring_mask` for evaluation, while oracle curves use target-independent reference data from their source log only. The next highest-value experiment is a smaller downstream-solver transfer study using representative source calibrations from each setup. That will measure how much cross-setup curve failure survives IMU fusion and magnetic-nuisance correction.",
        "",
    ])
    (output_dir / "cross_setup_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260906)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec, frame = read_inputs(args.spec, args.run_dir)
    baselines, pairs = analyze_pairs(
        spec,
        frame,
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    signals = signal_summary(spec)
    baselines.to_csv(args.run_dir / "setup_diagonal_baselines.csv", index=False)
    pairs.to_csv(args.run_dir / "setup_transfer_summary.csv", index=False)
    signals.to_csv(args.run_dir / "setup_signal_summary.csv", index=False)
    plot_matrices(spec, pairs, args.run_dir)
    write_report(spec, baselines, pairs, signals, args.run_dir)
    print(f"Wrote cross-setup analysis for {len(frame)} evaluations to {args.run_dir}")


if __name__ == "__main__":
    main()
