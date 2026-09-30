#!/usr/bin/env python3
"""Summarize the post-merge front calibration-duration experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sus-matplotlib-cache")
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = REPO_ROOT / "experiments" / "mag_calibration" / "runs"
RAW_RUNS = (
    "random-window-front-stumpy-postmerge-v1",
    "random-window-front-multisetup-postmerge-v1",
)
SOLVER_RUNS = (
    "solver-window-front-stumpy-postmerge-v1",
    "solver-window-front-multisetup-postmerge-v1",
)
SETUP_ORDER = (
    "stumpjumper_pod_v2",
    "tr11_pod_v2",
    "tr11_2025_pod_v2",
    "slayer_pod_v2",
)
RAW_METRICS = (
    "aligned_rmse",
    "fixed_aligned_rmse",
    "anchored_rmse",
    "aligned_nrmse_std",
    "travel_std",
    "travel_range",
    "prediction_to_travel_std",
    "bin0_fixed_rmse",
    "bin1_fixed_rmse",
    "bin2_fixed_rmse",
    "bin3_fixed_rmse",
    "training_observations",
)
SOLVER_METRICS = (
    "aligned_rmse",
    "fixed_aligned_rmse",
    "anchored_rmse",
    "aligned_nrmse_std",
    "bin0_fixed_rmse",
    "bin1_fixed_rmse",
    "bin2_fixed_rmse",
    "bin3_fixed_rmse",
)


def manifest(run_name: str) -> dict[str, Any]:
    return json.loads((RUN_ROOT / run_name / "manifest.json").read_text(encoding="utf-8"))


def setup_maps(run_name: str) -> tuple[dict[str, str], dict[str, str]]:
    spec = manifest(run_name)["spec"]
    log_to_setup = {
        log_name: setup
        for setup, logs in spec.get("setup_logs", {}).items()
        for log_name in logs
    }
    labels = dict(spec.get("setup_labels", {}))
    return log_to_setup, labels


def load_metrics(run_names: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for run_name in run_names:
        path = RUN_ROOT / run_name / "trial_metrics.csv"
        frame = pd.read_csv(path).copy()
        log_to_setup, labels = setup_maps(run_name)
        setup = frame["log"].map(log_to_setup)
        frame = frame.assign(
            run=run_name,
            setup=setup,
            setup_label=setup.map(labels),
        )
        if frame["setup"].isna().any():
            missing = sorted(frame.loc[frame["setup"].isna(), "log"].unique())
            raise ValueError(f"Missing setup mapping in {run_name}: {missing}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def log_balanced_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    available = [metric for metric in metrics if metric in frame.columns]
    numeric = frame.copy()
    for metric in available:
        numeric[metric] = pd.to_numeric(numeric[metric], errors="coerce")
    per_log = (
        numeric.groupby(group_columns + ["log"], dropna=False)[available]
        .median(numeric_only=True)
        .reset_index()
    )
    summary = (
        per_log.groupby(group_columns, dropna=False)[available]
        .median(numeric_only=True)
        .reset_index()
    )
    counts = (
        per_log.groupby(group_columns, dropna=False)["log"]
        .nunique()
        .rename("n_logs")
        .reset_index()
    )
    return summary.merge(counts, on=group_columns, how="left")


def failure_summary(run_names: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for run_name in run_names:
        schedule = pd.read_csv(RUN_ROOT / run_name / "trial_schedule.csv")
        failures_path = RUN_ROOT / run_name / "failures.csv"
        failed_ids: set[str] = set()
        if failures_path.stat().st_size:
            failures = pd.read_csv(failures_path)
            failed_ids = set(failures["trial_id"].astype(str))
        log_to_setup, labels = setup_maps(run_name)
        schedule["run"] = run_name
        schedule["setup"] = schedule["log"].map(log_to_setup)
        schedule["setup_label"] = schedule["setup"].map(labels)
        schedule["failed"] = schedule["trial_id"].astype(str).isin(failed_ids)
        frames.append(schedule)
    combined = pd.concat(frames, ignore_index=True)
    group = ["setup", "setup_label", "trainer", "duration_s"]
    result = combined.groupby(group, dropna=False).agg(
        scheduled_trials=("trial_id", "size"),
        failed_trials=("failed", "sum"),
    ).reset_index()
    result["failure_rate"] = result["failed_trials"] / result["scheduled_trials"]
    return result


def aggregate_comparison(
    old: pd.DataFrame,
    current: pd.DataFrame,
    *,
    keys: list[str],
    metric: str,
) -> pd.DataFrame:
    old = old.copy()
    current = current.copy()
    old[metric] = pd.to_numeric(old[metric], errors="coerce")
    current[metric] = pd.to_numeric(current[metric], errors="coerce")
    per_log_keys = keys + ["log"]
    old_log = old.groupby(per_log_keys, dropna=False)[metric].median().rename("premerge")
    current_log = current.groupby(per_log_keys, dropna=False)[metric].median().rename("current")
    paired = pd.concat([old_log, current_log], axis=1).dropna().reset_index()
    paired["delta_current_minus_premerge"] = paired["current"] - paired["premerge"]
    output = paired.groupby(keys, dropna=False).agg(
        premerge=("premerge", "median"),
        current=("current", "median"),
        paired_delta_median=("delta_current_minus_premerge", "median"),
        n_logs=("log", "nunique"),
    ).reset_index()
    output["delta_current_minus_premerge"] = output["current"] - output["premerge"]
    return output


def value(
    frame: pd.DataFrame,
    *,
    metric: str,
    default: float = float("nan"),
    **conditions: Any,
) -> float:
    selected = frame
    for key, expected in conditions.items():
        selected = selected[selected[key] == expected]
    if selected.empty or metric not in selected:
        return default
    return float(selected.iloc[0][metric])


def fmt(number: float, digits: int = 2) -> str:
    return "—" if not np.isfinite(number) else f"{number:.{digits}f}"


def make_plots(raw: pd.DataFrame, solver: pd.DataFrame, output_dir: Path) -> None:
    colors = dict(zip(SETUP_ORDER, ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd")))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for setup in SETUP_ORDER:
        rows = raw[
            (raw["setup"] == setup)
            & (raw["trainer"] == "self-supervised")
            & (raw["evaluation_scope"] == "full_log")
        ].sort_values("duration_s")
        if not rows.empty:
            axes[0].plot(rows["duration_s"], rows["aligned_rmse"], marker="o", color=colors[setup], label=rows.iloc[0]["setup_label"])
        solved = solver[
            (solver["setup"] == setup)
            & (solver["stage"] == "solved")
            & (solver["evaluation_scope"] == "full_log")
        ].sort_values("duration_s")
        if not solved.empty:
            axes[1].plot(solved["duration_s"], solved["aligned_rmse"], marker="o", color=colors[setup], label=solved.iloc[0]["setup_label"])
    for axis, title in zip(axes, ("Magnetic curve", "Final solved output")):
        axis.set_xscale("log")
        axis.set_xlabel("Active calibration duration (s)")
        axis.set_ylabel("Full-log aligned RMSE (mm)")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    figure.suptitle("Post-merge front calibration duration by setup")
    figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.95))
    figure.savefig(output_dir / "postmerge_duration_by_setup.png", dpi=180)
    figure.savefig(output_dir / "postmerge_duration_by_setup.pdf")
    plt.close(figure)


def write_report(
    raw: pd.DataFrame,
    solver: pd.DataFrame,
    failures: pd.DataFrame,
    raw_comparison: pd.DataFrame,
    solver_comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    lines = [
        "# Post-merge front calibration-duration study",
        "",
        "## Bottom line",
        "",
        "The current pipeline preserves the practical calibration-time result: useful accuracy is available by 10–20 active seconds, and the typical full-recording error is near its best around 40 seconds. The exact curve is setup-dependent, so 40 seconds should remain a default rather than a universal optimum. Longer accumulation is most useful as a robustness option when the calibration-quality signal remains poor.",
        "",
        "The earlier methodological explanation also survives. Same-window error can rise while full-log error improves because longer windows span more travel states. The supervised power oracle shows the same expanding-window behavior, while fixed-support and downstream results do not support the interpretation that more data simply makes the learner worse.",
        "",
        "## Design",
        "",
        "- Current merged front pipeline and production magnetic bad-mask behavior.",
        "- Eleven Stumpjumper/pod-v2 logs at 5, 10, 20, 40, 60, and 120 active seconds, with four deterministic nested centers.",
        "- Five independent recordings each from Jamaal's TR11, Harry's TR11, and Slayer at 5–60 seconds.",
        "- Full downstream solves on all Stumpjumper conditions except 60 seconds, plus two paired centers at 10, 40, and 60 seconds for every non-Stumpjumper log.",
        "- Logs are the independent analysis units; repeats are collapsed within log before setup medians are taken.",
        "- The selection/evaluation activity mask remains the reference-derived `boring_mask`.",
        "",
        "## Calibration-stage results",
        "",
        "Full-log aligned RMSE for the self-supervised magnetic curve:",
        "",
        "| Setup | 5 s | 10 s | 20 s | 40 s | 60 s | 120 s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for setup in SETUP_ORDER:
        selected = raw[(raw["setup"] == setup) & (raw["trainer"] == "self-supervised") & (raw["evaluation_scope"] == "full_log")]
        if selected.empty:
            continue
        label = str(selected.iloc[0]["setup_label"])
        cells = [fmt(value(selected, metric="aligned_rmse", duration_s=float(duration))) for duration in (5, 10, 20, 40, 60, 120)]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "Short-window failures under the production bad mask were uncommon but informative:",
        "",
        "| Setup | 5 s failure | 10 s failure | 20 s failure |",
        "|---|---:|---:|---:|",
    ])
    for setup in SETUP_ORDER:
        selected = failures[(failures["setup"] == setup) & (failures["trainer"] == "self-supervised")]
        if selected.empty:
            continue
        label = str(selected.iloc[0]["setup_label"])
        cells = [value(selected, metric="failure_rate", duration_s=float(duration)) for duration in (5, 10, 20)]
        lines.append(f"| {label} | " + " | ".join("—" if not np.isfinite(cell) else f"{cell:.1%}" for cell in cells) + " |")

    lines.extend([
        "",
        "## Final solver results",
        "",
        "Full-log aligned RMSE for the final solved output:",
        "",
        "| Setup | 5 s | 10 s | 20 s | 40 s | 60 s | 120 s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for setup in SETUP_ORDER:
        selected = solver[(solver["setup"] == setup) & (solver["stage"] == "solved") & (solver["evaluation_scope"] == "full_log")]
        if selected.empty:
            continue
        label = str(selected.iloc[0]["setup_label"])
        cells = [fmt(value(selected, metric="aligned_rmse", duration_s=float(duration))) for duration in (5, 10, 20, 40, 60, 120)]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    current_stumpy = solver[(solver["setup"] == "stumpjumper_pod_v2") & (solver["evaluation_scope"] == "full_log")]
    stumpy_training = raw[
        (raw["setup"] == "stumpjumper_pod_v2")
        & (raw["trainer"] == "self-supervised")
        & (raw["evaluation_scope"] == "training_window")
    ]
    stumpy_core = solver[
        (solver["setup"] == "stumpjumper_pod_v2")
        & (solver["evaluation_scope"] == "common_core")
    ]
    lines.extend([
        "",
        "For Stumpjumper, the final solve changes from "
        f"{fmt(value(current_stumpy, metric='aligned_rmse', stage='solved', duration_s=5.0))} mm at 5 seconds to "
        f"{fmt(value(current_stumpy, metric='aligned_rmse', stage='solved', duration_s=40.0))} mm at 40 seconds and "
        f"{fmt(value(current_stumpy, metric='aligned_rmse', stage='solved', duration_s=120.0))} mm at 120 seconds. "
        "The setup-balanced extension tests whether the same plateau appears on the other geometries rather than inferring that from pooled data.",
        "",
        "The apparent same-window reversal is still an evaluation-support effect. On Stumpjumper, raw magnetic RMSE on each window's own samples rises from "
        f"{fmt(value(stumpy_training, metric='aligned_rmse', duration_s=5.0))} mm at 5 seconds to "
        f"{fmt(value(stumpy_training, metric='aligned_rmse', duration_s=120.0))} mm at 120 seconds. "
        "When all calibrations are instead scored on the identical central 5-second core, raw magnetic RMSE changes from "
        f"{fmt(value(stumpy_core, metric='aligned_rmse', stage='mag_model', duration_s=5.0))} to "
        f"{fmt(value(stumpy_core, metric='aligned_rmse', stage='mag_model', duration_s=120.0))} mm, and final solved RMSE changes from "
        f"{fmt(value(stumpy_core, metric='aligned_rmse', stage='solved', duration_s=5.0))} to "
        f"{fmt(value(stumpy_core, metric='aligned_rmse', stage='solved', duration_s=120.0))} mm. "
        "The remaining non-monotonicity and the oracle's own-window rise still indicate a real one-dimensional curve compromise in addition to the support artifact.",
        "",
        "## Sensitivity to the pipeline revision",
        "",
        "The pre-merge and current studies use the same logs, duration labels, repeat identities, and deterministic random fractions. The activity mask changed with the pipeline, however, so their active-time coordinates do not always resolve to identical physical samples. The comparison below is therefore a cohort-level sensitivity analysis, not a pure paired code ablation.",
        "",
        "| Duration | Pre-merge mag | Current mag | Change | Pre-merge solved | Current solved | Change |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for duration in (5, 10, 20, 40, 120):
        raw_row = raw_comparison[(raw_comparison["duration_s"] == duration) & (raw_comparison["evaluation_scope"] == "full_log")]
        solved_row = solver_comparison[(solver_comparison["duration_s"] == duration) & (solver_comparison["evaluation_scope"] == "full_log") & (solver_comparison["stage"] == "solved")]
        raw_values = (float("nan"),) * 3 if raw_row.empty else tuple(float(raw_row.iloc[0][key]) for key in ("premerge", "current", "delta_current_minus_premerge"))
        solved_values = (float("nan"),) * 3 if solved_row.empty else tuple(float(solved_row.iloc[0][key]) for key in ("premerge", "current", "delta_current_minus_premerge"))
        lines.append(f"| {duration} s | {fmt(raw_values[0])} | {fmt(raw_values[1])} | {fmt(raw_values[2])} | {fmt(solved_values[0])} | {fmt(solved_values[1])} | {fmt(solved_values[2])} |")

    lines.extend([
        "",
        "## Implications",
        "",
        "1. Keep 10 active seconds as an earliest attempt, but gate release on accepted motion chunks and magnetic/travel-proxy coverage rather than time alone.",
        "2. Keep approximately 40 active seconds as the normal default. Continue collecting or retry on another block when the quality signal is weak.",
        "3. Do not select duration from own-window RMSE; use full-log/held-out behavior, fixed-support diagnostics, and final solved error.",
        "4. Report setup-stratified curves in the paper. A single pooled duration curve hides meaningful differences in both error floor and optimum.",
        "5. Treat the existing pre-merge mechanism study as supporting analysis, while using this report's current-pipeline tables for numerical recommendations.",
        "",
        "## Artifacts",
        "",
        "- `raw_setup_summary.csv`: log-balanced calibration-stage metrics by setup.",
        "- `solver_setup_summary.csv`: log-balanced stagewise solver metrics by setup.",
        "- `raw_failure_summary.csv`: planned-window failure rates, including no-chunk failures.",
        "- `raw_premerge_comparison.csv` and `solver_premerge_comparison.csv`: cohort-level sensitivity tables.",
        "- `postmerge_duration_by_setup.png`: paper-oriented setup-stratified learning curves.",
        "",
    ])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "mag_calibration" / "analysis" / "front-window-length-postmerge",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_metrics = load_metrics(RAW_RUNS)
    solver_metrics = load_metrics(SOLVER_RUNS)
    raw_summary = log_balanced_summary(
        raw_metrics,
        ["setup", "setup_label", "trainer", "duration_s", "evaluation_scope"],
        RAW_METRICS,
    )
    solver_summary = log_balanced_summary(
        solver_metrics,
        ["setup", "setup_label", "stage", "duration_s", "evaluation_scope"],
        SOLVER_METRICS,
    )
    failures = failure_summary(RAW_RUNS)

    old_raw = pd.read_csv(RUN_ROOT / "random-window-front-v1" / "trial_metrics.csv")
    old_raw = old_raw[(old_raw["trainer"] == "self-supervised") & (old_raw["repeat"] < 4)]
    new_raw = raw_metrics[(raw_metrics["setup"] == "stumpjumper_pod_v2") & (raw_metrics["trainer"] == "self-supervised")]
    raw_comparison = aggregate_comparison(
        old_raw,
        new_raw,
        keys=["duration_s", "evaluation_scope"],
        metric="aligned_rmse",
    )

    old_solver = pd.read_csv(RUN_ROOT / "solver-window-front-phase2" / "trial_metrics.csv")
    new_solver = solver_metrics[solver_metrics["setup"] == "stumpjumper_pod_v2"]
    solver_comparison = aggregate_comparison(
        old_solver,
        new_solver,
        keys=["stage", "duration_s", "evaluation_scope"],
        metric="aligned_rmse",
    )

    raw_summary.to_csv(args.output_dir / "raw_setup_summary.csv", index=False, lineterminator="\n")
    solver_summary.to_csv(args.output_dir / "solver_setup_summary.csv", index=False, lineterminator="\n")
    failures.to_csv(args.output_dir / "raw_failure_summary.csv", index=False, lineterminator="\n")
    raw_comparison.to_csv(args.output_dir / "raw_premerge_comparison.csv", index=False, lineterminator="\n")
    solver_comparison.to_csv(args.output_dir / "solver_premerge_comparison.csv", index=False, lineterminator="\n")
    make_plots(raw_summary, solver_summary, args.output_dir)
    write_report(raw_summary, solver_summary, failures, raw_comparison, solver_comparison, args.output_dir)
    print(f"Wrote post-merge duration analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
