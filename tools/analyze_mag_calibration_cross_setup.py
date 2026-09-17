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
    unit_overrides = spec.get("analysis_units", {})
    unknown_units = set(unit_overrides) - set(setup_by_log)
    if unknown_units:
        raise ValueError(
            f"Analysis units reference logs outside the setup cohort: {sorted(unknown_units)}"
        )
    frame["train_setup"] = frame["train_log"].map(setup_by_log)
    frame["eval_setup"] = frame["eval_log"].map(setup_by_log)
    frame["train_unit"] = frame["train_log"].map(
        lambda log_name: unit_overrides.get(log_name, log_name)
    )
    frame["eval_unit"] = frame["eval_log"].map(
        lambda log_name: unit_overrides.get(log_name, log_name)
    )
    return spec, frame


def crossed_bootstrap(
    group: pd.DataFrame,
    values: np.ndarray,
    *,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    source_column = "train_unit" if "train_unit" in group else "train_log"
    target_column = "eval_unit" if "eval_unit" in group else "eval_log"
    table = group.assign(_value=values).pivot_table(
        index=source_column,
        columns=target_column,
        values="_value",
        aggfunc="median",
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
    source_column = "train_unit" if "train_unit" in group else "train_log"
    if "eval_unit" in group:
        unit_pairs = group.groupby(
            [source_column, "eval_unit"], as_index=False
        )[column].median()
        return float(unit_pairs.groupby(source_column)[column].median().median())
    return float(group.groupby(source_column)[column].median().median())


def unit_balanced_fraction_positive(group: pd.DataFrame, column: str) -> float:
    source_column = "train_unit" if "train_unit" in group else "train_log"
    target_column = "eval_unit" if "eval_unit" in group else "eval_log"
    unit_pairs = group.groupby([source_column, target_column])[column].median()
    return float(np.mean(unit_pairs > 0))


def unit_balanced_median(group: pd.DataFrame, column: str, unit_column: str) -> float:
    return float(group.groupby(unit_column)[column].median().median())


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
                "independent_units": selected["eval_unit"].nunique(),
            }
            for metric in METRICS:
                row[f"{metric}_median"] = unit_balanced_median(
                    selected, metric, "eval_unit"
                )
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
                    selected = selected[selected["train_unit"] != selected["eval_unit"]]
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
                    "source_units": selected["train_unit"].nunique(),
                    "target_units": selected["eval_unit"].nunique(),
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
                        unit_balanced_median(
                            diagonal[
                            (diagonal["trainer"] == trainer)
                            & (diagonal["eval_setup"] == eval_setup)
                            ],
                            metric,
                            "eval_unit",
                        )
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
                    row[f"{metric}_fraction_worse"] = unit_balanced_fraction_positive(
                        selected_delta, "_delta"
                    )
                result_rows.append(row)
    return pd.DataFrame(baseline_rows), pd.DataFrame(result_rows)


def signal_summary(spec: dict) -> pd.DataFrame:
    rows = []
    unit_overrides = spec.get("analysis_units", {})
    for setup, log_names in spec["setup_logs"].items():
        per_log = []
        for log_name in log_names:
            data = load_cached_log(log_name)
            mask = data.activity_mask & np.isfinite(data.mag) & np.isfinite(data.travel)
            mag_low, mag_high = np.percentile(data.mag[mask], [5, 95])
            travel_low, travel_high = np.percentile(data.travel[mask], [5, 95])
            per_log.append({
                "unit": unit_overrides.get(log_name, log_name),
                "mag_low": mag_low,
                "mag_high": mag_high,
                "mag_span": mag_high - mag_low,
                "travel_low": travel_low,
                "travel_high": travel_high,
                "travel_span": travel_high - travel_low,
            })
        unit_medians = pd.DataFrame(per_log).groupby("unit").median(numeric_only=True)
        medians = unit_medians.median()
        rows.append({
            "setup": setup,
            "logs": len(log_names),
            "independent_units": len(unit_medians),
            "mag_p05_median": medians["mag_low"],
            "mag_p95_median": medians["mag_high"],
            "mag_p90_span_median": medians["mag_span"],
            "travel_p05_median_mm": medians["travel_low"],
            "travel_p95_median_mm": medians["travel_high"],
            "travel_p90_span_median_mm": medians["travel_span"],
        })
    return pd.DataFrame(rows)


def plot_matrices(spec: dict, pairs: pd.DataFrame, output_dir: Path) -> None:
    setups = list(spec["setup_logs"])
    labels = spec["setup_labels"]
    trainers = list(spec["trainers"])
    fig, axes = plt.subplots(
        2,
        len(trainers),
        figsize=(max(15.5, 5.2 * len(trainers)), max(8.2, 1.6 * len(setups) + 3.0)),
    )
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
                    axis.text(j, i, f"{value:+.1f}" if row_index else f"{value:.1f}", ha="center", va="center", color="white" if abs(value) > vmax * .42 else "black", fontsize=9 if len(setups) > 4 else 10)
            axis.set_xticks(range(len(setups)), [labels[name] for name in setups], rotation=32, ha="right")
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


def write_report(spec: dict, baselines: pd.DataFrame, pairs: pd.DataFrame, signals: pd.DataFrame, output_dir: Path) -> None:
    labels = spec["setup_labels"]
    setups = list(spec["setup_logs"])
    trainers = list(spec["trainers"])
    ss = "self-supervised"
    selected = pairs[pairs["trainer"] == ss]
    cross = selected[selected["transfer_type"] == "cross_setup"]
    same = selected[selected["transfer_type"] == "same_setup"]
    worst = cross.loc[cross["aligned_rmse_delta_median"].idxmax()]
    easiest = cross.loc[cross["aligned_rmse_delta_median"].idxmin()]
    positive_cells = int((cross["aligned_rmse_delta_median"] > 0).sum())
    positive_ci_cells = int((cross["aligned_rmse_delta_ci_low"] > 0).sum())
    unit_overrides = spec.get("analysis_units", {})
    independent_units = {
        unit_overrides.get(log_name, log_name)
        for log_names in spec["setup_logs"].values()
        for log_name in log_names
    }
    log_count = len(spec["logs"])
    pairs_per_trainer = (
        log_count**2 if spec.get("include_diagonal", True) else log_count * (log_count - 1)
    )
    expected_evaluations = pairs_per_trainer * len(trainers)
    sampling_note = ""
    if "sample_seed" in spec:
        sampling_note = (
            f" The cohort is a frozen setup-stratified random sample of "
            f"{spec.get('sample_per_setup', 'the configured number of')} independent "
            f"recordings per setup using seed {spec['sample_seed']}."
        )
        if spec.get("screening_run"):
            sampling_note += (
                " The screening run was used to verify cache freshness and failures, "
                "not to select by accuracy."
            )
    lines = [
        f"# {spec['name']}: cross-setup magnetometer calibration transfer",
        "",
        "## Bottom line",
        "",
        f"The expanded experiment finds a positive self-supervised transfer penalty in {positive_cells} of {len(cross)} directed cross-setup cells; {positive_ci_cells} have a crossed-bootstrap interval entirely above zero. The penalty ranges from {easiest['aligned_rmse_delta_median']:+.2f} to {worst['aligned_rmse_delta_median']:+.2f} mm, demonstrating that transfer is strongly directional and that pooled cross-setup averages are not sufficient.",
        "",
        "## Design",
        "",
        f"The experiment evaluates all {len(spec['logs'])} source logs against all {len(spec['logs'])} target logs for {len(trainers)} trainers: " + ", ".join(trainers) + f". It covers {len(setups)} setup cohorts and {len(independent_units)} independent recording units. Each calibration is trained once on its complete source log; the matrix retains the target log's own calibration and within-setup transfers as controls. All {expected_evaluations:,} evaluations completed successfully." + sampling_note,
        "",
        "The primary comparison is each transfer's aligned error minus the target log's own calibration error from the same trainer. Positive values favor independent target/per-recording calibration. Aligned normalized RMSE controls for different travel distributions. Confidence intervals use a crossed bootstrap that independently resamples source and target recording units. Derived chunks from the same parent recording are collapsed within unit pairs and resampled as one unit.",
        "",
        "## Target-specific baselines",
        "",
        "| Trainer | Setup | Logs | Independent units | Aligned RMSE | Normalized RMSE | Anchored RMSE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in baselines.iterrows():
        lines.append(
            f"| {row['trainer']} | {labels[row['setup']]} | {int(row['logs'])} | "
            f"{int(row['independent_units'])} | {row['aligned_rmse_median']:.2f} mm | "
            f"{row['aligned_nrmse_std_median']:.3f} | {row['anchored_rmse_median']:.2f} mm |"
        )
    lines.extend([
        "",
        "## Self-supervised transfer results",
        "",
        "| Source → target | Type | Source/target units | Transfer RMSE | Penalty vs target calibration (95% CI) | Δ normalized RMSE | Unit pairs worse | Anchored RMSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in selected.iterrows():
        lines.append(
            f"| {labels[row['train_setup']]} → {labels[row['eval_setup']]} | {row['transfer_type'].replace('_', ' ')} | "
            f"{int(row['source_units'])}/{int(row['target_units'])} | {row['aligned_rmse_transfer_median']:.2f} mm | "
            f"{row['aligned_rmse_delta_median']:+.2f} [{row['aligned_rmse_delta_ci_low']:+.2f}, "
            f"{row['aligned_rmse_delta_ci_high']:+.2f}] mm | {row['aligned_nrmse_std_delta_median']:+.3f} | "
            f"{row['aligned_rmse_fraction_worse']:.0%} | {row['anchored_rmse_transfer_median']:.2f} mm |"
        )
    lines.extend([
        "",
        "## Main findings",
        "",
        f"1. **Most setup changes penalize a frozen calibration.** {positive_cells} of {len(cross)} directed self-supervised cross-setup cells have a positive median penalty, compared with a same-setup range of {same['aligned_rmse_delta_median'].min():+.2f} to {same['aligned_rmse_delta_median'].max():+.2f} mm.",
        f"2. **The strongest failure is {labels[worst['train_setup']]} → {labels[worst['eval_setup']]}.** Its median penalty is {worst['aligned_rmse_delta_median']:+.2f} mm [{worst['aligned_rmse_delta_ci_low']:+.2f}, {worst['aligned_rmse_delta_ci_high']:+.2f}], and {worst['aligned_rmse_fraction_worse']:.0%} of independent source-target unit pairs are worse than target-specific calibration.",
        f"3. **The easiest transfer direction is {labels[easiest['train_setup']]} → {labels[easiest['eval_setup']]}.** Its median penalty is {easiest['aligned_rmse_delta_median']:+.2f} mm [{easiest['aligned_rmse_delta_ci_low']:+.2f}, {easiest['aligned_rmse_delta_ci_high']:+.2f}]. This direction should be interpreted separately rather than used to justify universal transfer.",
        "4. **Oracle comparisons distinguish physical transfer mismatch from learner noise.** The oracle trainers directly observe reference travel on the source log. Agreement with the self-supervised direction therefore supports a setup-specific mapping; disagreement identifies directions where self-supervised estimation variance affects the comparison.",
        "5. **Absolute and normalized metrics remain necessary.** Normalized error checks that results are not just caused by different travel ranges, while anchored error exposes offset and mounting-reference transfer in addition to curve shape.",
        "",
        "### Oracle cross-setup summary",
        "",
        "| Trainer | Positive median penalties | CIs above zero | Penalty range |",
        "|---|---:|---:|---:|",
    ])
    for trainer in trainers:
        trainer_cross = pairs[
            (pairs["trainer"] == trainer)
            & (pairs["transfer_type"] == "cross_setup")
        ]
        lines.append(
            f"| {trainer} | {(trainer_cross['aligned_rmse_delta_median'] > 0).sum()}/{len(trainer_cross)} | "
            f"{(trainer_cross['aligned_rmse_delta_ci_low'] > 0).sum()}/{len(trainer_cross)} | "
            f"{trainer_cross['aligned_rmse_delta_median'].min():+.2f} to {trainer_cross['aligned_rmse_delta_median'].max():+.2f} mm |"
        )
    lines.extend([
        "",
        "## Magnetic and travel support",
        "",
        "| Setup | Logs | Independent units | Median magnetic p5–p95 | Median magnetic span | Median travel span |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for _, row in signals.iterrows():
        lines.append(
            f"| {labels[row['setup']]} | {int(row['logs'])} | {int(row['independent_units'])} | "
            f"{row['mag_p05_median']:.0f}–{row['mag_p95_median']:.0f} | "
            f"{row['mag_p90_span_median']:.0f} | {row['travel_p90_span_median_mm']:.1f} mm |"
        )
    lines.extend([
        "",
        "Large support differences explain some directional asymmetry: a curve trained on a narrow magnetic interval may extrapolate or clip when transferred to a wider target interval. Support overlap is not sufficient by itself, because different magnet placement and fork geometry can still produce a different curve within overlapping ranges.",
        "",
        "## Paper implications",
        "",
        "- Report the directed setup matrix rather than one pooled transfer statistic; source and target roles are not interchangeable.",
        "- Use the target-specific diagonal and within-setup transfer as controls, so target difficulty and ordinary recording variation are separated from setup mismatch.",
        "- Treat the two supervised oracles as a physical/curve-family control rather than as a deployable method.",
        "- Keep aligned and normalized error primary, with anchored error as the production-oriented secondary measure.",
        "",
        "## Pipeline implications",
        "",
        "- Do not silently reuse a calibration across an unknown setup. Require setup identity or self-calibrate on the current recording.",
        "- Store the calibration's observed magnetic support and reject unsupported extrapolation.",
        "- Validate any warm-start or population prior with current-recording IMU constraints before accepting it.",
        "- Use this matrix to develop a compatibility score from support overlap, curve parameters, and held-out self-supervised residuals.",
        "",
        "## Limitations and next step",
        "",
        "This experiment evaluates the magnetic mapping before downstream fusion and uses full-log source calibration. It uses the reference-derived `boring_mask` for evaluation; oracle curves use reference travel only from their source log. The confidence intervals are descriptive because these recordings were not collected as a prospectively held-out cohort. The next highest-value experiment is a smaller downstream-solver transfer study using representative source calibrations from each setup.",
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
