#!/usr/bin/env python3
"""Analyze mechanisms behind mag-calibration window-length error trends."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import sys
import tomllib
from typing import Any, Iterable

os.environ["MPLCONFIGDIR"] = "/private/tmp"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT / "backend", REPO_ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from mag_calibration import MagTravelCalibration, TimeRange, resolve_window  # noqa: E402
from mag_calibration_experiment import load_cached_log, score_prediction  # noqa: E402


RUN_NAMES = (
    "random-window-front-v1-shorter",
    "random-window-front-v1",
    "random-window-front-v2",
)
TRAVEL_BIN_EDGES = np.linspace(0.0, 150.0, 6)
SETUP_ORDER = (
    "stumpjumper-pod-v2",
    "stumpjumper-pod-v1",
    "tr11-pod-v2",
    "other",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_runs() -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    base = REPO_ROOT / "experiments" / "mag_calibration" / "runs"
    for name in RUN_NAMES:
        directory = base / name
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        runs[name] = {
            "directory": directory,
            "manifest": manifest,
            "spec": manifest["spec"],
            "metrics": read_csv(directory / "trial_metrics.csv"),
            "aggregate": read_csv(directory / "aggregate_summary.csv"),
            "schedule": read_csv(directory / "trial_schedule.csv"),
        }
    return runs


def registry_setups() -> dict[str, str]:
    registry = tomllib.loads((REPO_ROOT / "logs" / "registry.toml").read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for log_name, record in registry["logs"].items():
        profiles = set(record.get("profiles", []))
        if "bike-tr11" in profiles and "front-pod-v2" in profiles:
            setup = "tr11-pod-v2"
        elif "bike-stumpjumper" in profiles and "front-pod-v1" in profiles:
            setup = "stumpjumper-pod-v1"
        elif "bike-stumpjumper" in profiles and "front-pod-v2" in profiles:
            setup = "stumpjumper-pod-v2"
        else:
            setup = "other"
        result[log_name] = setup
    return result


def finite(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def log_balanced(rows: Iterable[dict[str, Any]], metric: str) -> tuple[float, int]:
    by_log: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = finite(row, metric)
        if value is not None:
            by_log[str(row["log"])].append(value)
    medians = [float(np.median(values)) for values in by_log.values()]
    return (float(np.median(medians)), len(medians)) if medians else (float("nan"), 0)


def combined_curves(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keep = (
        "aligned_rmse_log_median",
        "aligned_nrmse_std_log_median",
        "fixed_aligned_rmse_log_median",
        "anchored_rmse_log_median",
        "travel_std_log_median",
        "travel_range_log_median",
        "prediction_to_travel_std_log_median",
        "training_observations_log_median",
        "failure_rate",
        "n_logs",
    )
    for name, run in runs.items():
        for source in run["aggregate"]:
            row: dict[str, Any] = {
                "run": name,
                "trainer": source["trainer"],
                "evaluation_scope": source["evaluation_scope"],
                "duration_s": float(source["duration_s"]),
            }
            for key in keep:
                value = finite(source, key)
                row[key] = value if value is not None else source.get(key)
            rows.append(row)
    return rows


def setup_curves(
    runs: dict[str, dict[str, Any]], setups: dict[str, str]
) -> list[dict[str, Any]]:
    source = runs["random-window-front-v2"]["metrics"]
    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, str, float, str]] = set()
    for row in source:
        keys.add((row["trainer"], row["evaluation_scope"], float(row["duration_s"]), setups.get(row["log"], "other")))
    for trainer, scope, duration, setup in sorted(keys, key=str):
        selected = [
            row for row in source
            if row["trainer"] == trainer
            and row["evaluation_scope"] == scope
            and float(row["duration_s"]) == duration
            and setups.get(row["log"], "other") == setup
        ]
        for metric in ("aligned_rmse", "aligned_nrmse_std", "fixed_aligned_rmse"):
            value, n_logs = log_balanced(selected, metric)
            rows.append({
                "run": "random-window-front-v2",
                "setup": setup,
                "trainer": trainer,
                "evaluation_scope": scope,
                "duration_s": duration,
                "metric": metric,
                "value": value,
                "n_logs": n_logs,
            })
    return rows


def fixed_bin_curves(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, run in runs.items():
        source = [row for row in run["metrics"] if row["evaluation_scope"] == "training_window"]
        keys = {(row["trainer"], float(row["duration_s"])) for row in source}
        for trainer, duration in sorted(keys, key=str):
            selected = [row for row in source if row["trainer"] == trainer and float(row["duration_s"]) == duration]
            for index, (low, high) in enumerate(zip(TRAVEL_BIN_EDGES[:-1], TRAVEL_BIN_EDGES[1:])):
                # Twenty samples is intentionally permissive for 1-second
                # windows; n_logs makes sparse support visible in the output.
                supported = [row for row in selected if float(row.get(f"bin{index}_n", 0)) >= 20]
                for metric_name, column in (
                    ("local_aligned_rmse", f"bin{index}_rmse"),
                    ("fixed_aligned_rmse", f"bin{index}_fixed_rmse"),
                    ("anchored_rmse", f"bin{index}_anchored_rmse"),
                ):
                    value, n_logs = log_balanced(supported, column)
                    rows.append({
                        "run": name,
                        "trainer": trainer,
                        "duration_s": duration,
                        "travel_bin": f"{low:g}-{high:g}",
                        "metric": metric_name,
                        "rmse_mm": value,
                        "n_logs": n_logs,
                    })
    return rows


def common_core_curves(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    data_cache: dict[str, Any] = {}
    for name, run in runs.items():
        directory: Path = run["directory"]
        durations = sorted(float(value) for value in run["spec"]["durations_s"])
        core_duration = durations[0]
        groups: dict[tuple[str, str, str], dict[float, tuple[dict[str, str], dict[str, Any]]]] = defaultdict(dict)
        for row in run["schedule"]:
            result = json.loads((directory / "trials" / f"{row['trial_id']}.json").read_text(encoding="utf-8"))
            if result.get("status") != "success":
                continue
            groups[(row["log"], row["repeat"], row["trainer"])][float(row["duration_s"])] = (row, result)

        raw_rows: list[dict[str, Any]] = []
        for (log_name, repeat, trainer), calibrations in groups.items():
            if core_duration not in calibrations:
                continue
            data = data_cache.setdefault(log_name, load_cached_log(log_name))
            core_row = calibrations[core_duration][0]
            core = resolve_window(
                data.time_s,
                TimeRange(float(core_row["start_s"]), float(core_row["stop_s"])),
                time_basis=core_row["time_basis"],
                activity_mask=data.activity_mask,
            )
            for duration, (_, result) in calibrations.items():
                calibration = MagTravelCalibration.from_dict(result["calibration"])
                prediction = calibration.predict(data.mag)
                score = score_prediction(prediction, data, core)
                raw_rows.append({
                    "run": name,
                    "log": log_name,
                    "repeat": repeat,
                    "trainer": trainer,
                    "core_duration_s": core_duration,
                    "training_duration_s": duration,
                    **score,
                })

        keys = {(row["trainer"], float(row["training_duration_s"])) for row in raw_rows}
        for trainer, duration in sorted(keys, key=str):
            selected = [row for row in raw_rows if row["trainer"] == trainer and float(row["training_duration_s"]) == duration]
            for metric in ("aligned_rmse", "aligned_nrmse_std"):
                value, n_logs = log_balanced(selected, metric)
                output.append({
                    "run": name,
                    "trainer": trainer,
                    "core_duration_s": core_duration,
                    "training_duration_s": duration,
                    "metric": metric,
                    "value": value,
                    "n_logs": n_logs,
                    "conditioned_on_core_fit_success": True,
                })
    return output


def paired_endpoint(
    rows: list[dict[str, str]],
    *,
    trainer: str,
    scope: str,
    metric: str,
    low: float,
    high: float,
) -> dict[str, Any]:
    groups: dict[tuple[str, str], dict[float, float]] = defaultdict(dict)
    for row in rows:
        if row["trainer"] != trainer or row["evaluation_scope"] != scope:
            continue
        value = finite(row, metric)
        if value is not None:
            groups[(row["log"], row["repeat"])][float(row["duration_s"])] = value
    pairs = [(values[low], values[high]) for values in groups.values() if low in values and high in values]
    differences = np.asarray([end - start for start, end in pairs], dtype=float)
    return {
        "pairs": len(pairs),
        "low_median": float(np.median([pair[0] for pair in pairs])),
        "high_median": float(np.median([pair[1] for pair in pairs])),
        "median_delta": float(np.median(differences)),
        "fraction_increased": float(np.mean(differences > 0)),
    }


def select_curve(
    rows: list[dict[str, Any]],
    *,
    run: str,
    trainer: str,
    scope: str,
) -> list[dict[str, Any]]:
    return sorted(
        (row for row in rows if row["run"] == run and row["trainer"] == trainer and row["evaluation_scope"] == scope),
        key=lambda row: float(row["duration_s"]),
    )


def make_plots(
    curves: list[dict[str, Any]],
    common_core: list[dict[str, Any]],
    bins: list[dict[str, Any]],
    setups: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    colors = {"self-supervised": "#1f77b4", "oracle-power": "#ff7f0e", "oracle-isotonic": "#2ca02c"}
    run_name = "random-window-front-v1"
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = (
        (axes[0, 0], "training_window", "aligned_rmse_log_median", "Locally centered RMSE (mm)"),
        (axes[0, 1], "training_window", "aligned_nrmse_std_log_median", "RMSE / travel standard deviation"),
        (axes[1, 0], "training_window", "fixed_aligned_rmse_log_median", "Training-window RMSE, full-log offset (mm)"),
        (axes[1, 1], "full_log", "aligned_rmse_log_median", "Full-log centered RMSE (mm)"),
    )
    for axis, scope, metric, title in panels:
        for trainer in colors:
            rows = select_curve(curves, run=run_name, trainer=trainer, scope=scope)
            if rows:
                axis.plot([row["duration_s"] for row in rows], [row[metric] for row in rows], marker="o", color=colors[trainer], label=trainer)
        axis.set_xscale("log")
        axis.set_xlabel("Active training duration (s)")
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Front v1: separating window support, centering, and generalization")
    figure.tight_layout()
    figure.savefig(output_dir / "v1_mechanisms.png", dpi=180)
    figure.savefig(output_dir / "v1_mechanisms.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    own = select_curve(curves, run=run_name, trainer="self-supervised", scope="training_window")
    core = sorted(
        (row for row in common_core if row["run"] == run_name and row["trainer"] == "self-supervised" and row["metric"] == "aligned_rmse"),
        key=lambda row: float(row["training_duration_s"]),
    )
    full = select_curve(curves, run=run_name, trainer="self-supervised", scope="full_log")
    axis.plot([row["duration_s"] for row in own], [row["aligned_rmse_log_median"] for row in own], marker="o", label="own training window")
    axis.plot([row["training_duration_s"] for row in core], [row["value"] for row in core], marker="o", label="same central 5 s core")
    axis.plot([row["duration_s"] for row in full], [row["aligned_rmse_log_median"] for row in full], marker="o", label="full log")
    axis.set_xscale("log")
    axis.set_xlabel("Active training duration (s)")
    axis.set_ylabel("Aligned RMSE (mm)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title("Front v1 self-supervised: evaluation support changes the trend")
    figure.tight_layout()
    figure.savefig(output_dir / "v1_common_core.png", dpi=180)
    figure.savefig(output_dir / "v1_common_core.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    selected = [row for row in bins if row["run"] == run_name and row["trainer"] == "self-supervised" and row["metric"] == "fixed_aligned_rmse"]
    for travel_bin in sorted({row["travel_bin"] for row in selected}, key=lambda value: float(value.split("-")[0])):
        rows = sorted((row for row in selected if row["travel_bin"] == travel_bin and row["n_logs"] >= 9), key=lambda row: float(row["duration_s"]))
        if rows:
            axis.plot([row["duration_s"] for row in rows], [row["rmse_mm"] for row in rows], marker="o", label=f"{travel_bin} mm")
    axis.set_xscale("log")
    axis.set_xlabel("Active training duration (s)")
    axis.set_ylabel("Per-bin RMSE using full-log offset (mm)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    axis.set_title("Front v1 self-supervised: longer windows shift error toward low travel")
    figure.tight_layout()
    figure.savefig(output_dir / "v1_fixed_travel_bins.png", dpi=180)
    figure.savefig(output_dir / "v1_fixed_travel_bins.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    selected = [row for row in setups if row["trainer"] == "self-supervised" and row["evaluation_scope"] == "full_log" and row["metric"] == "aligned_rmse"]
    for setup in SETUP_ORDER:
        rows = sorted((row for row in selected if row["setup"] == setup), key=lambda row: float(row["duration_s"]))
        if rows:
            axis.plot([row["duration_s"] for row in rows], [row["value"] for row in rows], marker="o", label=f"{setup} (n={rows[0]['n_logs']})")
    axis.set_xscale("log")
    axis.set_xlabel("Active training duration (s)")
    axis.set_ylabel("Full-log aligned RMSE (mm)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title("Front v2: pooled trend is strongly setup-dependent")
    figure.tight_layout()
    figure.savefig(output_dir / "v2_setup_stratified.png", dpi=180)
    figure.savefig(output_dir / "v2_setup_stratified.pdf")
    plt.close(figure)


def lookup_curve(
    curves: list[dict[str, Any]], run: str, trainer: str, scope: str, duration: float, metric: str
) -> float:
    for row in curves:
        if row["run"] == run and row["trainer"] == trainer and row["evaluation_scope"] == scope and float(row["duration_s"]) == duration:
            return float(row[metric])
    return float("nan")


def lookup_common(common: list[dict[str, Any]], run: str, trainer: str, duration: float) -> float:
    for row in common:
        if row["run"] == run and row["trainer"] == trainer and row["metric"] == "aligned_rmse" and float(row["training_duration_s"]) == duration:
            return float(row["value"])
    return float("nan")


def lookup_setup(setups: list[dict[str, Any]], setup: str, duration: float) -> float:
    for row in setups:
        if row["setup"] == setup and row["trainer"] == "self-supervised" and row["evaluation_scope"] == "full_log" and row["metric"] == "aligned_rmse" and float(row["duration_s"]) == duration:
            return float(row["value"])
    return float("nan")


def lookup_bin(bins: list[dict[str, Any]], travel_bin: str, duration: float) -> float:
    for row in bins:
        if row["run"] == "random-window-front-v1" and row["trainer"] == "self-supervised" and row["metric"] == "fixed_aligned_rmse" and row["travel_bin"] == travel_bin and float(row["duration_s"]) == duration:
            return float(row["rmse_mm"])
    return float("nan")


def write_report(
    runs: dict[str, dict[str, Any]],
    curves: list[dict[str, Any]],
    common: list[dict[str, Any]],
    bins: list[dict[str, Any]],
    setups: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    v1 = "random-window-front-v1"
    short = "random-window-front-v1-shorter"
    v2 = "random-window-front-v2"
    v1_rows = runs[v1]["metrics"]
    endpoint = paired_endpoint(v1_rows, trainer="self-supervised", scope="training_window", metric="aligned_rmse", low=5.0, high=120.0)
    endpoint_norm = paired_endpoint(v1_rows, trainer="self-supervised", scope="training_window", metric="aligned_nrmse_std", low=5.0, high=120.0)
    short_failure_1 = lookup_curve(curves, short, "self-supervised", "training_window", 1.0, "failure_rate")
    short_failure_5 = lookup_curve(curves, short, "self-supervised", "training_window", 5.0, "failure_rate")
    lines = [
        "# Front mag-calibration window-length analysis",
        "",
        "## Bottom line",
        "",
        "The rise in same-window absolute RMSE is mostly an evaluation-support and local-centering effect, not evidence that the self-supervised learner simply gets worse with more data. Longer windows contain a wider travel distribution and cannot use a highly local offset correction. On an identical central evaluation core, self-supervised curves generally improve from the shortest windows and then plateau. Full-log accuracy also improves strongly before reaching a setup-dependent optimum.",
        "",
        "There is nevertheless evidence of a real long-window compromise: the low-travel and high-travel regions move in opposite directions, and even supervised one-dimensional oracles accumulate more normalized residual as a window spans more behavior. This is consistent with hysteresis, time variation, or another missing state variable making travel not perfectly single-valued in magnetic magnitude.",
        "",
        "## 1. Travel support explains much of the apparent rise",
        "",
        "| Front v1 self-supervised | 5 s | 120 s |",
        "|---|---:|---:|",
        f"| Own-window aligned RMSE | {lookup_curve(curves, v1, 'self-supervised', 'training_window', 5, 'aligned_rmse_log_median'):.2f} mm | {lookup_curve(curves, v1, 'self-supervised', 'training_window', 120, 'aligned_rmse_log_median'):.2f} mm |",
        f"| Travel standard deviation | {lookup_curve(curves, v1, 'self-supervised', 'training_window', 5, 'travel_std_log_median'):.2f} mm | {lookup_curve(curves, v1, 'self-supervised', 'training_window', 120, 'travel_std_log_median'):.2f} mm |",
        f"| Travel range | {lookup_curve(curves, v1, 'self-supervised', 'training_window', 5, 'travel_range_log_median'):.1f} mm | {lookup_curve(curves, v1, 'self-supervised', 'training_window', 120, 'travel_range_log_median'):.1f} mm |",
        f"| RMSE / travel standard deviation | {lookup_curve(curves, v1, 'self-supervised', 'training_window', 5, 'aligned_nrmse_std_log_median'):.3f} | {lookup_curve(curves, v1, 'self-supervised', 'training_window', 120, 'aligned_nrmse_std_log_median'):.3f} |",
        "",
        f"Across {endpoint['pairs']} paired nested windows, raw RMSE increased in {endpoint['fraction_increased']:.0%} of 5-to-120-second comparisons (median change {endpoint['median_delta']:+.2f} mm). After normalization by each window's travel standard deviation, the median paired change was only {endpoint_norm['median_delta']:+.3f}, and increases occurred in {endpoint_norm['fraction_increased']:.0%}. The supervised oracles also rise on their own larger training windows, confirming that this pattern is not unique to self-supervised optimization.",
        "",
        "## 2. Local centering makes short windows look better",
        "",
        f"At 5 seconds, front-v1 self-supervised RMSE is {lookup_curve(curves, v1, 'self-supervised', 'training_window', 5, 'aligned_rmse_log_median'):.2f} mm with a separate local offset, but {lookup_curve(curves, v1, 'self-supervised', 'training_window', 5, 'fixed_aligned_rmse_log_median'):.2f} mm when the full-log offset is held fixed. At 120 seconds the two are {lookup_curve(curves, v1, 'self-supervised', 'training_window', 120, 'aligned_rmse_log_median'):.2f} and {lookup_curve(curves, v1, 'self-supervised', 'training_window', 120, 'fixed_aligned_rmse_log_median'):.2f} mm. Thus local centering contributes about half a millimeter of the apparent 5-second advantage and almost none at 120 seconds.",
        "",
        "## 3. Holding evaluation data fixed reverses the main interpretation",
        "",
        "| Training duration | Own window | Same central 5 s | Full log |",
        "|---:|---:|---:|---:|",
    ]
    for duration in (5, 10, 20, 40, 60, 120):
        lines.append(
            f"| {duration} s | {lookup_curve(curves, v1, 'self-supervised', 'training_window', duration, 'aligned_rmse_log_median'):.2f} | {lookup_common(common, v1, 'self-supervised', duration):.2f} | {lookup_curve(curves, v1, 'self-supervised', 'full_log', duration, 'aligned_rmse_log_median'):.2f} |"
        )
    lines.extend([
        "",
        "On the identical central 5-second samples, increasing training from 5 to 10–20 seconds improves RMSE rather than worsening it. The own-window curve rises because its evaluation set expands. Full-log accuracy reaches its best median around 40 seconds in this cohort, after which it is approximately flat or slightly worse.",
        "",
        "## 4. Absolute anchoring is now a larger error source than curve shape",
        "",
        f"For front v1, the existing non-reference anchoring policy gives {lookup_curve(curves, v1, 'self-supervised', 'full_log', 5, 'anchored_rmse_log_median'):.2f} mm full-log RMSE at 5 seconds, {lookup_curve(curves, v1, 'self-supervised', 'full_log', 40, 'anchored_rmse_log_median'):.2f} mm at 40 seconds, and {lookup_curve(curves, v1, 'self-supervised', 'full_log', 120, 'anchored_rmse_log_median'):.2f} mm at 120 seconds. The corresponding optimally aligned errors are {lookup_curve(curves, v1, 'self-supervised', 'full_log', 5, 'aligned_rmse_log_median'):.2f}, {lookup_curve(curves, v1, 'self-supervised', 'full_log', 40, 'aligned_rmse_log_median'):.2f}, and {lookup_curve(curves, v1, 'self-supervised', 'full_log', 120, 'aligned_rmse_log_median'):.2f} mm. Once 10–40 seconds of useful motion are available, improving the absolute reference/anchor is likely more valuable than simply accumulating more curve-training data.",
        "",
        "## 5. The long-window compromise is travel-region dependent",
        "",
        "Using the same full-log alignment offset for every training window, front-v1 self-supervised fixed-bin RMSE changes as follows:",
        "",
        "| Travel bin | 5 s training | 120 s training | Direction |",
        "|---|---:|---:|---|",
        f"| 0–30 mm | {lookup_bin(bins, '0-30', 5):.2f} | {lookup_bin(bins, '0-30', 120):.2f} | worse |",
        f"| 30–60 mm | {lookup_bin(bins, '30-60', 5):.2f} | {lookup_bin(bins, '30-60', 120):.2f} | slightly worse |",
        f"| 60–90 mm | {lookup_bin(bins, '60-90', 5):.2f} | {lookup_bin(bins, '60-90', 120):.2f} | better |",
        f"| 90–120 mm | {lookup_bin(bins, '90-120', 5):.2f} | {lookup_bin(bins, '90-120', 120):.2f} | better |",
        "",
        f"Longer training exposes high-travel motion and improves that end of the curve, while the low-travel end degrades. On the full log, the predicted-to-reference travel standard-deviation ratio moves from {lookup_curve(curves, v1, 'self-supervised', 'full_log', 5, 'prediction_to_travel_std_log_median'):.3f} at 5 seconds to {lookup_curve(curves, v1, 'self-supervised', 'full_log', 40, 'prediction_to_travel_std_log_median'):.3f} at 40 seconds and {lookup_curve(curves, v1, 'self-supervised', 'full_log', 120, 'prediction_to_travel_std_log_median'):.3f} at 120 seconds: short fits tend to under-span travel, whereas the longest fits slightly over-span it. The same low-end tendency appears in the supervised oracles, so it is unlikely to be only a self-supervised optimizer bug. A single magnetic-magnitude curve is being asked to compromise across states that are not perfectly consistent.",
        "",
        "## 6. Very short windows are information-limited",
        "",
        f"In the 1–5-second Stumpjumper sweep, self-supervised fit failures fall from {short_failure_1:.1%} at 1 second to {short_failure_5:.1%} at 5 seconds. The log-balanced median number of accepted motion chunks grows from {lookup_curve(curves, short, 'self-supervised', 'training_window', 1, 'training_observations_log_median'):.1f} to {lookup_curve(curves, short, 'self-supervised', 'training_window', 5, 'training_observations_log_median'):.1f}. Full-log RMSE falls from {lookup_curve(curves, short, 'self-supervised', 'full_log', 1, 'aligned_rmse_log_median'):.2f} to {lookup_curve(curves, short, 'self-supervised', 'full_log', 5, 'aligned_rmse_log_median'):.2f} mm. Time alone is therefore not the best readiness criterion; accepted chunks and magnetic/travel excitation should gate calibration.",
        "",
        "## 7. The all-LSM6DSO32 result is setup-dependent",
        "",
        "The completed v2 manifest contains 1, 2, 5, 10, and 40 seconds. The TOML was subsequently extended, so 60–120-second v2 points do not exist in this run and are not analyzed here.",
        "",
        "| Setup | Logs | Full-log RMSE at 10 s | Full-log RMSE at 40 s |",
        "|---|---:|---:|---:|",
        f"| Stumpjumper, pod-v2 | 11 | {lookup_setup(setups, 'stumpjumper-pod-v2', 10):.2f} | {lookup_setup(setups, 'stumpjumper-pod-v2', 40):.2f} |",
        f"| Stumpjumper, pod-v1 | 7 | {lookup_setup(setups, 'stumpjumper-pod-v1', 10):.2f} | {lookup_setup(setups, 'stumpjumper-pod-v1', 40):.2f} |",
        f"| TR11, pod-v2 | 6 | {lookup_setup(setups, 'tr11-pod-v2', 10):.2f} | {lookup_setup(setups, 'tr11-pod-v2', 40):.2f} |",
        "",
        "The pooled v2 uptick at 40 seconds is not universal: the original Stumpjumper/pod-v2 group continues improving, while the older pod-v1 and TR11 groups plateau at substantially higher error. Sensor/geometry strata should therefore be shown separately even though the algorithm receives no setup-specific prior.",
        "",
        "## Pipeline implications",
        "",
        "1. Do not optimize calibration duration using own-window absolute RMSE alone. Use full-log or held-out-block error, normalized error, fixed-bin curves, and production-anchored error together.",
        "2. Require a minimum information budget rather than only elapsed active time: accepted chunk count, magnetic span, and preferably coverage across relevant travel states.",
        "3. Ten active seconds is a defensible minimum starting point for the current front learner; 20–40 seconds is safer when the objective is whole-recording accuracy. One to two seconds is unreliable.",
        "4. Consider balancing training chunks over magnetic/travel proxy bins. The long-window curve currently trades low-travel accuracy for high-travel accuracy as high-excitation samples enter the fit.",
        "5. Treat within-log recalibration as a remaining hypothesis, not yet a conclusion. A matched-support, time-separated transfer experiment is needed to distinguish true temporal drift from hysteresis and changing travel distributions.",
        "",
        "## Method notes",
        "",
        "All aggregate curves first take the median across repeats within each log and then the median across logs. Common-core results evaluate every nested calibration on the exact smallest central window and are conditioned on that smallest self-supervised fit succeeding. Fixed-bin diagnostics require at least 20 samples in a bin and report the number of contributing logs in the accompanying CSV. `boring_mask` remains the reference-derived activity mask.",
        "",
    ])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "mag_calibration" / "analysis" / "front-window-length",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    setups = registry_setups()
    curves = combined_curves(runs)
    setup_rows = setup_curves(runs, setups)
    bin_rows = fixed_bin_curves(runs)
    common_rows = common_core_curves(runs)
    write_csv(args.output_dir / "combined_curves.csv", curves)
    write_csv(args.output_dir / "setup_curves.csv", setup_rows)
    write_csv(args.output_dir / "fixed_bin_curves.csv", bin_rows)
    write_csv(args.output_dir / "common_core_curves.csv", common_rows)
    make_plots(curves, common_rows, bin_rows, setup_rows, args.output_dir)
    write_report(runs, curves, common_rows, bin_rows, setup_rows, args.output_dir)
    print(f"Wrote window-length analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
