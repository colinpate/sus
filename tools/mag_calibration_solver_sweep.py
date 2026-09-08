#!/usr/bin/env python3
"""Run full-log front fusion for selected window-trained calibrations."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import tomllib
from typing import Any

os.environ["MPLCONFIGDIR"] = "/private/tmp"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT / "backend", REPO_ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from classes.log_config import attach_log_config  # noqa: E402
from classes.time_series import TimeSeries  # noqa: E402
from log_registry import resolve_log  # noqa: E402
from mag_calibration import MagTravelCalibration, TimeRange, resolve_window  # noqa: E402
from mag_calibration_experiment import load_cached_log, predict_calibration, score_prediction  # noqa: E402
from mag_calibration_sweep import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    git_snapshot,
    sha256_bytes,
    stable_id,
    utc_now,
    write_csv,
)
from mag_nuisance import (  # noqa: E402
    MAG_NUISANCE_SUMMARY_FIELDS,
    MagNuisanceFullRateCorrection,
    MagNuisanceTravelCorrection,
)
from travel_solver import TravelSolver  # noqa: E402


SCHEMA_VERSION = 1
STAGES = (
    "mag_model",
    "fusion1",
    "nuisance_delta_lifted",
    "nuisance_corrected_mag",
    "solved",
)
_CACHE: dict[str, tuple[Any, dict[str, Any]]] = {}


def read_spec(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    spec = tomllib.loads(raw.decode("utf-8"))
    for key in ("name", "source_run", "source_trainer", "durations_s", "logs"):
        if key not in spec:
            raise ValueError(f"Solver spec is missing {key!r}")
    if "source_repeat" not in spec and "source_repeats" not in spec:
        raise ValueError("Solver spec must define source_repeat or source_repeats")
    if "source_repeat" in spec and "source_repeats" in spec:
        raise ValueError("Define only one of source_repeat and source_repeats")
    if spec.get("experiment_type") != "downstream_solver_window":
        raise ValueError("Expected experiment_type = 'downstream_solver_window'")
    if spec.get("pipeline") != "front":
        raise ValueError("Phase 1 currently supports the front solver only")
    return spec, sha256_bytes(raw)


def source_directory(spec: dict[str, Any]) -> Path:
    path = Path(spec["source_run"])
    return path if path.is_absolute() else REPO_ROOT / path


def build_schedule(spec: dict[str, Any]) -> list[dict[str, Any]]:
    source = source_directory(spec)
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    source_rows = list(csv.DictReader((source / "trial_schedule.csv").open(encoding="utf-8")))
    wanted_durations = {float(value) for value in spec["durations_s"]}
    wanted_repeats = {
        int(value) for value in spec.get("source_repeats", [spec.get("source_repeat")])
    }
    selected = [
        row for row in source_rows
        if row["log"] in spec["logs"]
        and row["trainer"] == spec["source_trainer"]
        and int(row["repeat"]) in wanted_repeats
        and float(row["duration_s"]) in wanted_durations
    ]
    expected = len(spec["logs"]) * len(wanted_durations) * len(wanted_repeats)
    if len(selected) != expected:
        raise ValueError(f"Found {len(selected)} source trials, expected {expected}")
    rows = []
    for source_row in selected:
        result_path = source / "trials" / f"{source_row['trial_id']}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "success":
            raise ValueError(f"Source calibration {source_row['trial_id']} did not succeed")
        identity = {"experiment": spec["name"], "source_trial_id": source_row["trial_id"]}
        rows.append({
            "trial_id": stable_id(identity),
            "experiment": spec["name"],
            "pipeline": "front",
            "log": source_row["log"],
            "duration_s": float(source_row["duration_s"]),
            "repeat": int(source_row["repeat"]),
            "trainer": source_row["trainer"],
            "time_basis": source_row["time_basis"],
            "start_s": float(source_row["start_s"]),
            "stop_s": float(source_row["stop_s"]),
            "center_s": float(source_row["center_s"]),
            "source_trial_id": source_row["trial_id"],
            "source_run_fingerprint": source_manifest.get("spec_sha256"),
            "cache_fingerprint": source_row.get("cache_fingerprint"),
        })
    return sorted(
        rows,
        key=lambda row: (
            spec["logs"].index(row["log"]),
            int(row["repeat"]),
            row["duration_s"],
        ),
    )


def seed_reused_trials(spec: dict[str, Any], rows: list[dict[str, Any]], output_dir: Path) -> int:
    wanted = {row["source_trial_id"]: row for row in rows}
    seeded = 0
    for configured in spec.get("reuse_runs", []):
        prior = Path(configured)
        if not prior.is_absolute():
            prior = REPO_ROOT / prior
        prior_schedule = list(csv.DictReader((prior / "trial_schedule.csv").open(encoding="utf-8")))
        for prior_row in prior_schedule:
            current = wanted.get(prior_row["source_trial_id"])
            if current is None:
                continue
            destination = trial_path(output_dir, current["trial_id"])
            if destination.exists():
                continue
            prior_result_path = prior / "trials" / f"{prior_row['trial_id']}.json"
            if not prior_result_path.exists():
                continue
            result = json.loads(prior_result_path.read_text(encoding="utf-8"))
            if result.get("status") != "success":
                continue
            result["trial"] = current
            result["reused_from"] = str(prior_result_path)
            atomic_write_json(destination, result)
            seeded += 1
    return seeded


def prepare_run(spec_path: Path, output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec, spec_hash = read_spec(spec_path)
    manifest_path = output_dir / "manifest.json"
    schedule_path = output_dir / "trial_schedule.csv"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["spec_sha256"] != spec_hash:
            raise ValueError("Output directory belongs to another spec revision")
        if manifest["schedule_sha256"] != sha256_bytes(schedule_path.read_bytes()):
            raise ValueError("Frozen solver schedule has changed")
        rows = list(csv.DictReader(schedule_path.open(encoding="utf-8")))
        seed_reused_trials(spec, rows, output_dir)
        return spec, rows
    rows = build_schedule(spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(schedule_path, rows)
    atomic_write_json(manifest_path, {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "experiment": spec["name"],
        "spec": spec,
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": spec_hash,
        "schedule_sha256": sha256_bytes(schedule_path.read_bytes()),
        "trial_count": len(rows),
        "source_run": str(source_directory(spec)),
        "git": git_snapshot(),
        "method_note": "Each saved calibration drives both full-log fusion solves; scoring windows are applied afterward.",
    })
    seed_reused_trials(spec, rows, output_dir)
    return spec, rows


def cached_ts(cache: np.lib.npyio.NpzFile, key: str, *, units: str, frame: str) -> TimeSeries:
    t = np.asarray(cache[f"{key}__t"], dtype=float)
    x = np.asarray(cache[f"{key}__x"], dtype=float)
    fs_hz = 1.0 / float(np.median(np.diff(t)))
    return TimeSeries(t=t, x=x, units=units, frame=frame, meta={"fs_hz": fs_hz})


def load_solver_inputs(log_name: str) -> tuple[Any, dict[str, Any]]:
    if log_name in _CACHE:
        return _CACHE[log_name]
    data = load_cached_log(log_name)
    path = REPO_ROOT / "backend" / "run_artifacts" / log_name / "cache" / "all.npz"
    with np.load(path, allow_pickle=False) as cache:
        inputs = {
            "accel/lpfhp/proj": cached_ts(cache, "accel/lpfhp/proj", units="m/s^2", frame="travel"),
            "mag/norm/corr/lpf": cached_ts(cache, "mag/norm/corr/lpf", units="milli-Gauss", frame="mag"),
            "mag/lpf": cached_ts(cache, "mag/lpf", units="milli-Gauss", frame="mag"),
            "gyro/lpf/gyro1": cached_ts(cache, "gyro/lpf/gyro1", units="deg/s", frame="gyro1"),
            "mag_zv_points": np.asarray(cache["mag_zv_points"], dtype=int),
            "mag_baseline": np.asarray(cache["mag_baseline"], dtype=float),
        }
    _CACHE[log_name] = (data, inputs)
    return data, inputs


def timed_step(step: Any, ws: dict[str, Any]) -> float:
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        step.run(ws)
    return time.perf_counter() - started


def run_downstream(log_name: str, calibration: MagTravelCalibration) -> tuple[Any, dict[str, np.ndarray], dict[str, float], dict[str, float]]:
    data, cached = load_solver_inputs(log_name)
    ws = dict(cached)
    attach_log_config(ws, resolve_log(log_name).processing_config)
    with contextlib.redirect_stdout(io.StringIO()):
        raw, adjusted, offset = predict_calibration(calibration, data)
    t = data.time_s
    ws["travel/mag_model/adj"] = TimeSeries(t, adjusted, "mm", "travel", {"fs_hz": 1.0 / np.median(np.diff(t))})
    ws["mag_model_coeffs"] = np.asarray(calibration.coefficients, dtype=float)
    ws["mag_model_offset_mm"] = np.array([offset], dtype=float)
    runtimes: dict[str, float] = {}
    runtimes["fusion1"] = timed_step(TravelSolver(
        name="travel_solver",
        inputs=("accel/lpfhp/proj", "mag/norm/corr/lpf", "travel/mag_model/adj", "mag_zv_points", "mag_baseline"),
        outputs=("travel/fusion1",), max_nfev=100, verbose=0,
    ), ws)
    runtimes["nuisance"] = timed_step(MagNuisanceTravelCorrection(
        name="mag_nuisance_correction",
        inputs=("mag/lpf", "gyro/lpf/gyro1", "mag/norm/corr/lpf", "mag_model_coeffs", "mag_model_offset_mm", "travel/fusion1"),
        outputs=("travel/solved/mag_nuisance/10hz", "mag/nuisance/body/10hz", "mag/nuisance/world/10hz", "mag/nuisance/xyz_path", "mag/nuisance/summary"),
    ), ws)
    runtimes["nuisance_full_rate"] = timed_step(MagNuisanceFullRateCorrection(
        name="mag_nuisance_full_rate",
        inputs=("mag/lpf", "gyro/lpf/gyro1", "travel/fusion1", "travel/mag_model/adj", "travel/solved/mag_nuisance/10hz", "mag/nuisance/body/10hz", "mag/nuisance/world/10hz", "mag/nuisance/xyz_path"),
        outputs=("travel/solved/mag_nuisance/delta_lifted", "travel/mag_nuisance/corrected"),
    ), ws)
    runtimes["fusion2"] = timed_step(TravelSolver(
        name="travel_solver_mag_nuisance",
        inputs=("accel/lpfhp/proj", "mag/norm/corr/lpf", "travel/mag_nuisance/corrected", "mag_zv_points", "mag_baseline"),
        outputs=("travel/solved",), max_nfev=100, verbose=0,
    ), ws)
    signals = {
        "mag_model": adjusted,
        "fusion1": ws["travel/fusion1"].x[:, 0],
        "nuisance_delta_lifted": ws["travel/solved/mag_nuisance/delta_lifted"].x[:, 0],
        "nuisance_corrected_mag": ws["travel/mag_nuisance/corrected"].x[:, 0],
        "solved": ws["travel/solved"].x[:, 0],
    }
    summary_values = np.asarray(ws["mag/nuisance/summary"], dtype=float)
    nuisance = dict(zip(MAG_NUISANCE_SUMMARY_FIELDS, summary_values.tolist()))
    return data, signals, runtimes, nuisance


def scoped_metrics(data: Any, signals: dict[str, np.ndarray], row: dict[str, Any], spec: dict[str, Any]) -> list[dict[str, Any]]:
    training = resolve_window(data.time_s, TimeRange(float(row["start_s"]), float(row["stop_s"])), time_basis=row["time_basis"], activity_mask=data.activity_mask)
    full = resolve_window(data.time_s, TimeRange(), time_basis=row["time_basis"], activity_mask=data.activity_mask)
    core_s = float(spec.get("common_core_s", min(spec["durations_s"])))
    center = float(row["center_s"])
    core = resolve_window(data.time_s, TimeRange(center - core_s / 2, center + core_s / 2), time_basis=row["time_basis"], activity_mask=data.activity_mask)
    train_mask = training.sample_mask(len(data.time_s))
    scopes = {
        "training_window": (training, None),
        "common_core": (core, None),
        "full_log": (full, None),
        "full_log_excluding_training": (full, train_mask),
    }
    output = []
    for stage, prediction in signals.items():
        full_offset = score_prediction(prediction, data, full)["aligned_offset_mm"]
        for scope in spec["evaluation_scopes"]:
            resolved, exclude = scopes[scope]
            score = score_prediction(prediction, data, resolved, exclude_mask=exclude, fixed_alignment_offset_mm=full_offset)
            output.append({"stage": stage, "evaluation_scope": scope, **score})
    by_scope = {(item["stage"], item["evaluation_scope"]): item for item in output}
    for item in output:
        if item["stage"] == "mag_model":
            continue
        base = by_scope[("mag_model", item["evaluation_scope"])]
        for prefix in ("aligned", "anchored", "fixed_aligned"):
            item[f"{prefix}_mse_improvement_vs_mag"] = base[f"{prefix}_rmse"] ** 2 - item[f"{prefix}_rmse"] ** 2
        for index in range(5):
            for kind in ("rmse", "anchored_rmse", "fixed_rmse"):
                key = f"bin{index}_{kind}"
                item[f"{key}_mse_improvement_vs_mag"] = base[key] ** 2 - item[key] ** 2
    return output


def atomic_save_predictions(path: Path, time_s: np.ndarray, signals: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}-", suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, time_s=time_s, **signals)
    os.replace(temporary, path)


def execute(row: dict[str, Any], spec: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        source = source_directory(spec)
        source_result = json.loads((source / "trials" / f"{row['source_trial_id']}.json").read_text(encoding="utf-8"))
        calibration = MagTravelCalibration.from_dict(source_result["calibration"])
        data, signals, runtimes, nuisance = run_downstream(row["log"], calibration)
        metrics = scoped_metrics(data, signals, row, spec)
        if spec.get("save_predictions", True):
            atomic_save_predictions(output_dir / "predictions" / f"{row['trial_id']}.npz", data.time_s, signals)
        return {"status": "success", "metrics_version": 2, "trial": row, "runtime_s": {**runtimes, "total": time.perf_counter() - started}, "nuisance": nuisance, "metrics": metrics}
    except Exception as error:
        return {"status": "failed", "trial": row, "runtime_s": {"total": time.perf_counter() - started}, "error_type": type(error).__name__, "error": str(error)}


def trial_path(output_dir: Path, trial_id: str) -> Path:
    return output_dir / "trials" / f"{trial_id}.json"


def run_trials(
    rows: list[dict[str, Any]],
    spec: dict[str, Any],
    output_dir: Path,
    max_trials: int | None,
    workers: int,
) -> tuple[int, int]:
    pending = [row for row in rows if not trial_path(output_dir, row["trial_id"]).exists()]
    if max_trials is not None:
        pending = pending[:max_trials]
    failures = 0
    if workers <= 1:
        completed = ((row, execute(row, spec, output_dir)) for row in pending)
        for index, (row, result) in enumerate(completed, 1):
            atomic_write_json(trial_path(output_dir, row["trial_id"]), result)
            failures += result["status"] == "failed"
            print(f"[{index}/{len(pending)}] {row['log']} r{row['repeat']} {row['duration_s']}s: {result['status']} ({result['runtime_s']['total']:.1f}s)", flush=True)
        return len(pending), failures

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute, row, spec, output_dir): row for row in pending}
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            result = future.result()
            atomic_write_json(trial_path(output_dir, row["trial_id"]), result)
            failures += result["status"] == "failed"
            print(f"[{index}/{len(pending)}] {row['log']} r{row['repeat']} {row['duration_s']}s: {result['status']} ({result['runtime_s']['total']:.1f}s)", flush=True)
    return len(pending), failures


def summarize(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    metrics, failures, runtimes = [], [], []
    completed = 0
    for row in rows:
        path = trial_path(output_dir, row["trial_id"])
        if not path.exists():
            continue
        completed += 1
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["status"] != "success":
            failures.append({**row, "error": result.get("error"), "error_type": result.get("error_type")})
            continue
        runtimes.append({**row, **result["runtime_s"], **{f"nuisance_{k}": v for k, v in result["nuisance"].items()}})
        metrics.extend({**row, **item} for item in result["metrics"])
    write_csv(output_dir / "trial_metrics.csv", metrics)
    write_csv(output_dir / "runtime_metrics.csv", runtimes)
    write_csv(output_dir / "failures.csv", failures)
    aggregate = []
    for duration in sorted({float(row["duration_s"]) for row in metrics}):
        for stage in STAGES:
            for scope in sorted({row["evaluation_scope"] for row in metrics}):
                selected = [row for row in metrics if float(row["duration_s"]) == duration and row["stage"] == stage and row["evaluation_scope"] == scope]
                if not selected:
                    continue
                out = {
                    "duration_s": duration,
                    "stage": stage,
                    "evaluation_scope": scope,
                    "n_logs": len({row["log"] for row in selected}),
                    "n_trials": len(selected),
                }
                for key in ("aligned_rmse", "anchored_rmse", "fixed_aligned_rmse", "aligned_nrmse_std", "bin0_fixed_rmse", "bin0_fixed_mean_error", "aligned_mse_improvement_vs_mag", "bin0_fixed_rmse_mse_improvement_vs_mag"):
                    values = np.asarray([
                        np.median([
                            float(row[key]) for row in selected
                            if row["log"] == log_name
                            and key in row
                            and np.isfinite(float(row[key]))
                        ])
                        for log_name in sorted({row["log"] for row in selected})
                        if any(
                            row["log"] == log_name
                            and key in row
                            and np.isfinite(float(row[key]))
                            for row in selected
                        )
                    ])
                    out[f"{key}_median"] = float(np.median(values)) if len(values) else float("nan")
                aggregate.append(out)
    write_csv(output_dir / "aggregate_summary.csv", aggregate)
    status = {"updated_at": utc_now(), "scheduled_trials": len(rows), "completed_trials": completed, "successful_trials": completed - len(failures), "failed_trials": len(failures), "remaining_trials": len(rows) - completed, "complete": completed == len(rows)}
    create_plot(aggregate, output_dir, completed < len(rows), output_dir.name)
    write_report(metrics, runtimes, status, output_dir)
    atomic_write_json(output_dir / "status.json", status)
    return status


def create_plot(rows: list[dict[str, Any]], output_dir: Path, partial: bool, experiment_name: str) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = (("full_log", "aligned_rmse_median", "Full-log aligned RMSE"), ("training_window", "fixed_aligned_rmse_median", "Training-window RMSE, full-log offset"), ("training_window", "bin0_fixed_rmse_median", "Training-window 0–30 mm RMSE, full-log offset"))
    for axis, (scope, metric, title) in zip(axes, panels):
        for stage in ("mag_model", "fusion1", "nuisance_corrected_mag", "solved"):
            selected = sorted((row for row in rows if row["evaluation_scope"] == scope and row["stage"] == stage), key=lambda row: row["duration_s"])
            if selected:
                axis.plot([row["duration_s"] for row in selected], [row[metric] for row in selected], marker="o", label=stage)
        axis.set_xscale("log"); axis.set_xlabel("Training duration (s)"); axis.set_ylabel("RMSE (mm)"); axis.set_title(title); axis.grid(alpha=.25)
    axes[0].legend(frameon=False)
    fig.suptitle(experiment_name.replace("-", " ").title() + (" (partial)" if partial else "")); fig.tight_layout()
    fig.savefig(output_dir / "solver_learning_curve.png", dpi=180); fig.savefig(output_dir / "solver_learning_curve.pdf"); plt.close(fig)


def endpoint_change(
    rows: list[dict[str, Any]],
    *,
    scope: str,
    stage: str,
    metric: str,
    start_duration: float | None = None,
    stop_duration: float | None = None,
) -> dict[str, float]:
    selected = [
        row for row in rows
        if row["evaluation_scope"] == scope and row["stage"] == stage
    ]
    durations = sorted({float(row["duration_s"]) for row in selected})
    start = durations[0] if start_duration is None else float(start_duration)
    stop = durations[-1] if stop_duration is None else float(stop_duration)
    by_key = {
        (row["log"], int(row["repeat"]), float(row["duration_s"])): float(row[metric])
        for row in selected
    }
    log_deltas = []
    for log_name in sorted({row["log"] for row in selected}):
        repeated = [
            by_key[(log_name, repeat, stop)]
            - by_key[(log_name, repeat, start)]
            for repeat in sorted({int(row["repeat"]) for row in selected if row["log"] == log_name})
            if (log_name, repeat, start) in by_key
            and (log_name, repeat, stop) in by_key
        ]
        finite_repeated = np.asarray(repeated)[np.isfinite(repeated)]
        if len(finite_repeated):
            log_deltas.append(float(np.median(finite_repeated)))
    finite = np.asarray(log_deltas)
    if not len(finite):
        return {"median": float("nan"), "fraction_improved": float("nan"), "p": float("nan"), "n": 0}
    try:
        p_value = float(wilcoxon(finite).pvalue)
    except ValueError:
        p_value = float("nan")
    return {
        "median": float(np.median(finite)),
        "fraction_improved": float(np.mean(finite < 0)),
        "p": p_value,
        "n": int(len(finite)),
    }


def median_metric(
    rows: list[dict[str, Any]],
    *,
    duration: float,
    scope: str,
    stage: str,
    metric: str,
) -> float:
    selected = [
        row for row in rows
        if float(row["duration_s"]) == duration
        and row["evaluation_scope"] == scope
        and row["stage"] == stage
        and metric in row
        and np.isfinite(float(row[metric]))
    ]
    values = np.asarray([
        np.median([float(row[metric]) for row in selected if row["log"] == log_name])
        for log_name in sorted({row["log"] for row in selected})
    ])
    return float(np.nanmedian(values)) if len(values) else float("nan")


def within_log_repeat_iqr(
    rows: list[dict[str, Any]],
    *,
    duration: float,
    scope: str,
    stage: str,
    metric: str,
) -> float:
    selected = [
        row for row in rows
        if float(row["duration_s"]) == duration
        and row["evaluation_scope"] == scope
        and row["stage"] == stage
        and metric in row
        and np.isfinite(float(row[metric]))
    ]
    spreads = []
    for log_name in sorted({row["log"] for row in selected}):
        values = np.asarray([float(row[metric]) for row in selected if row["log"] == log_name])
        spreads.append(float(np.percentile(values, 75) - np.percentile(values, 25)))
    return float(np.median(spreads)) if spreads else float("nan")


def trial_quantile(
    rows: list[dict[str, Any]],
    *,
    duration: float,
    scope: str,
    stage: str,
    metric: str,
    quantile: float,
) -> float:
    values = np.asarray([
        float(row[metric]) for row in rows
        if float(row["duration_s"]) == duration
        and row["evaluation_scope"] == scope
        and row["stage"] == stage
        and metric in row
        and np.isfinite(float(row[metric]))
    ])
    return float(np.quantile(values, quantile)) if len(values) else float("nan")


def fmt(value: float, digits: int = 2) -> str:
    return "—" if not np.isfinite(value) else f"{value:.{digits}f}"


def write_report(
    metrics: list[dict[str, Any]],
    runtimes: list[dict[str, Any]],
    status: dict[str, Any],
    output_dir: Path,
) -> None:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    spec = manifest["spec"]
    durations = sorted(float(value) for value in spec["durations_s"])
    if not metrics:
        atomic_write_text(
            output_dir / "report.md",
            f"# {spec['name']}\n\nNo successful trials yet.\n",
        )
        return
    available_durations = sorted({float(row["duration_s"]) for row in metrics})
    short, long = available_durations[0], available_durations[-1]
    repeats = sorted({int(row["repeat"]) for row in metrics})
    repeat_text = ", ".join(str(value) for value in repeats)
    full_mag = endpoint_change(metrics, scope="full_log", stage="mag_model", metric="aligned_rmse")
    full_solved = endpoint_change(metrics, scope="full_log", stage="solved", metric="aligned_rmse")
    solved_20_40 = endpoint_change(
        metrics,
        scope="full_log",
        stage="solved",
        metric="aligned_rmse",
        start_duration=20,
        stop_duration=40,
    ) if {20.0, 40.0}.issubset(available_durations) else None
    solved_40_120 = endpoint_change(
        metrics,
        scope="full_log",
        stage="solved",
        metric="aligned_rmse",
        start_duration=40,
        stop_duration=120,
    ) if {40.0, 120.0}.issubset(available_durations) else None
    train_bin_mag = endpoint_change(metrics, scope="training_window", stage="mag_model", metric="bin0_fixed_rmse")
    train_bin_solved = endpoint_change(metrics, scope="training_window", stage="solved", metric="bin0_fixed_rmse")
    core_bin_mag = endpoint_change(metrics, scope="common_core", stage="mag_model", metric="bin0_fixed_rmse")
    core_bin_solved = endpoint_change(metrics, scope="common_core", stage="solved", metric="bin0_fixed_rmse")
    total_runtime = np.asarray([float(row["total"]) for row in runtimes])

    lines = [
        f"# {spec['name']}",
        "",
        f"Status: **{status['successful_trials']}/{status['scheduled_trials']} successful** "
        f"({status['failed_trials']} failed).",
        "",
        "## Design",
        "",
        f"- Front pipeline; {len(spec['logs'])} Stumpjumper/pod-v2 logs.",
        f"- {len(repeats)} deterministic nested window(s) per log (source repeats {repeat_text}) at "
        + ", ".join(f"{duration:g} s" for duration in durations) + ".",
        "- The saved self-supervised curve is injected before the two full-log fusion solves.",
        "- Scoring is applied afterward on the training window, a fixed centered common core, "
        "the full log, and the full log excluding training.",
        "- Fixed-offset window metrics use the corresponding stage's full-log alignment, so "
        "window-local recentering cannot hide bias.",
        "",
        "## Main result",
        "",
        f"On the full log, median raw magnetic-model RMSE changes from "
        f"{fmt(median_metric(metrics, duration=short, scope='full_log', stage='mag_model', metric='aligned_rmse'))} "
        f"to {fmt(median_metric(metrics, duration=long, scope='full_log', stage='mag_model', metric='aligned_rmse'))} mm; "
        f"the paired {long:g}-{short:g} s change is {fmt(full_mag['median'])} mm "
        f"({full_mag['fraction_improved']:.0%} of logs improve, exploratory Wilcoxon p={fmt(full_mag['p'], 3)}).",
        f"For the final solved output, median full-log RMSE changes from "
        f"{fmt(median_metric(metrics, duration=short, scope='full_log', stage='solved', metric='aligned_rmse'))} "
        f"to {fmt(median_metric(metrics, duration=long, scope='full_log', stage='solved', metric='aligned_rmse'))} mm; "
        f"the paired change is {fmt(full_solved['median'])} mm "
        f"({full_solved['fraction_improved']:.0%} improve, p={fmt(full_solved['p'], 3)}).",
        "",
        "The suspicious low-travel trend is strongly attenuated downstream. On the varying "
        f"training windows, raw 0–30 mm fixed-offset RMSE has a paired change of "
        f"{fmt(train_bin_mag['median'])} mm from {short:g} to {long:g} s, while the final solved "
        f"output changes by {fmt(train_bin_solved['median'])} mm. On the identical centered "
        f"{spec.get('common_core_s', short):g} s core, the corresponding changes are "
        f"{fmt(core_bin_mag['median'])} and {fmt(core_bin_solved['median'])} mm. This supports "
        "the interpretation that fusion/correction removes much of the raw curve's apparent "
        "low-travel degradation rather than propagating it to final travel.",
        "",
        "## Practical duration",
        "",
        *(
            [
                f"The typical paired full-log solved improvement is {fmt(-solved_20_40['median'])} mm "
                f"from 20 to 40 s (p={fmt(solved_20_40['p'], 3)}) and only "
                f"{fmt(-solved_40_120['median'])} mm from 40 to 120 s "
                f"(p={fmt(solved_40_120['p'], 3)}). The central-error curve therefore has a "
                "practical elbow around 20–40 active seconds.",
                f"Longer calibration still improves repeatability and tail risk. Median within-log "
                f"repeat IQR falls from {fmt(within_log_repeat_iqr(metrics, duration=20, scope='full_log', stage='solved', metric='aligned_rmse'))} mm at 20 s "
                f"to {fmt(within_log_repeat_iqr(metrics, duration=40, scope='full_log', stage='solved', metric='aligned_rmse'))} mm at 40 s and "
                f"{fmt(within_log_repeat_iqr(metrics, duration=120, scope='full_log', stage='solved', metric='aligned_rmse'))} mm at 120 s. "
                f"The across-condition 90th percentile is {fmt(trial_quantile(metrics, duration=20, scope='full_log', stage='solved', metric='aligned_rmse', quantile=.9))}, "
                f"{fmt(trial_quantile(metrics, duration=40, scope='full_log', stage='solved', metric='aligned_rmse', quantile=.9))}, and "
                f"{fmt(trial_quantile(metrics, duration=120, scope='full_log', stage='solved', metric='aligned_rmse', quantile=.9))} mm, respectively. "
                "This makes 40 s a reasonable default, while 120 s is preferable when robustness "
                "to an unlucky calibration window matters more than calibration latency.",
                "",
            ]
            if solved_20_40 is not None and solved_40_120 is not None
            else []
        ),
        "## Median RMSE by duration",
        "",
        "| Training | Full log: mag | Full log: solved | Training window: mag | Training window: solved | 0–30 mm training: mag | 0–30 mm training: solved |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for duration in durations:
        values = [
            median_metric(metrics, duration=duration, scope="full_log", stage="mag_model", metric="aligned_rmse"),
            median_metric(metrics, duration=duration, scope="full_log", stage="solved", metric="aligned_rmse"),
            median_metric(metrics, duration=duration, scope="training_window", stage="mag_model", metric="fixed_aligned_rmse"),
            median_metric(metrics, duration=duration, scope="training_window", stage="solved", metric="fixed_aligned_rmse"),
            median_metric(metrics, duration=duration, scope="training_window", stage="mag_model", metric="bin0_fixed_rmse"),
            median_metric(metrics, duration=duration, scope="training_window", stage="solved", metric="bin0_fixed_rmse"),
        ]
        lines.append(f"| {duration:g} s | " + " | ".join(fmt(value) for value in values) + " |")
    lines.extend([
        "",
        "All values are millimetres. Repeats are first collapsed within each log, then logs are "
        "weighted equally. Negative paired changes mean "
        "lower error at the longest duration. The p-values are descriptive only: this "
        "run remains an exploratory rather than confirmatory significance study.",
        "",
        "## Runtime and artifacts",
        "",
        f"Median runtime was {fmt(float(np.median(total_runtime)), 1)} s per full-log condition "
        f"({fmt(float(np.sum(total_runtime)) / 60.0, 1)} solver-minutes total).",
        "Raw per-trial metrics, " + ("predictions, " if spec.get("save_predictions", True) else "") + "aggregate tables, and the learning-curve figure are "
        "stored beside this report. The frozen schedule links every solve to its exact source "
        "calibration and cached-input fingerprint.",
        "",
    ])
    atomic_write_text(output_dir / "report.md", "\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("schedule", "run", "summarize"):
        sub = subs.add_parser(command); sub.add_argument("spec", type=Path); sub.add_argument("--output-dir", type=Path, required=True)
        if command == "run":
            sub.add_argument("--max-trials", type=int)
            sub.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); spec, rows = prepare_run(args.spec, args.output_dir)
    if args.command == "schedule": print(f"Scheduled {len(rows)} full-log solver runs"); return
    if args.command == "run":
        count, failures = run_trials(rows, spec, args.output_dir, args.max_trials, args.workers); print(f"Executed {count} solver runs ({failures} failed)")
    print(json.dumps(summarize(rows, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
