#!/usr/bin/env python3
"""Deterministic, resumable random-window mag-calibration experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
import tomllib
from typing import Any, Iterable

os.environ["MPLCONFIGDIR"] = "/private/tmp"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
TOOLS_DIR = REPO_ROOT / "tools"
for directory in (BACKEND_DIR, TOOLS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from mag_calibration import (  # noqa: E402
    MagTravelCalibration,
    RecordingWindow,
    TimeRange,
    resolve_window,
    sample_durations,
)
from mag_calibration_experiment import (  # noqa: E402
    calibration_columns,
    fit_calibration,
    load_cached_log,
    predict_calibration,
    score_prediction,
)


SCHEMA_VERSION = 1
SUPPORTED_TRAINERS = {
    "self-supervised",
    "oracle-power",
    "oracle-isotonic",
    "oracle-binned-median",
}
SUPPORTED_SCOPES = {
    "training_window",
    "full_log",
    "full_log_excluding_training",
}
SUMMARY_METRICS = (
    "aligned_rmse",
    "aligned_mae",
    "aligned_nrmse_std",
    "aligned_nrmse_p90",
    "anchored_rmse",
    "anchored_mae",
    "anchored_nrmse_std",
    "fixed_aligned_rmse",
    "fixed_aligned_nrmse_std",
    "bin_rmse",
    "bin_occupied_rmse",
    "correlation",
    "travel_std",
    "travel_range",
    "travel_p90_span",
    "prediction_std",
    "prediction_to_travel_std",
    "training_observations",
)
_WORKER_CACHE: dict[str, Any] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_uniform(*parts: object) -> float:
    payload = json.dumps(parts, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / 2**64


def stable_id(value: object, *, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write_text(path, "")
        return
    preferred = [
        "trial_id", "pipeline", "log", "repeat", "duration_s", "trainer",
        "evaluation_scope", "status", "aligned_rmse", "anchored_rmse",
    ]
    keys = set().union(*(row.keys() for row in rows))
    fields = [key for key in preferred if key in keys]
    fields.extend(sorted(keys - set(fields)))
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_spec(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    spec = tomllib.loads(raw.decode("utf-8"))
    required = ("name", "seed", "logs", "durations_s", "repeats", "trainers")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"Experiment spec is missing: {', '.join(missing)}")
    if int(spec.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported experiment schema {spec.get('schema_version')!r}")
    if not spec["logs"]:
        raise ValueError("Experiment spec must contain at least one log")
    durations = [float(value) for value in spec["durations_s"]]
    if not durations or any(value <= 0 for value in durations):
        raise ValueError("durations_s must contain positive values")
    if len(set(durations)) != len(durations):
        raise ValueError("durations_s must not contain duplicates")
    if int(spec["repeats"]) <= 0:
        raise ValueError("repeats must be positive")
    unknown_trainers = set(spec["trainers"]) - SUPPORTED_TRAINERS
    if unknown_trainers:
        raise ValueError(f"Unknown trainers: {sorted(unknown_trainers)}")
    scopes = set(spec.get("evaluation_scopes", sorted(SUPPORTED_SCOPES)))
    unknown_scopes = scopes - SUPPORTED_SCOPES
    if unknown_scopes:
        raise ValueError(f"Unknown evaluation scopes: {sorted(unknown_scopes)}")
    if spec.get("time_basis", "active") not in ("active", "elapsed"):
        raise ValueError("time_basis must be 'active' or 'elapsed'")
    if spec.get("sampling", "nested-center") != "nested-center":
        raise ValueError("Only nested-center sampling is currently supported")
    return spec, sha256_bytes(raw)


def available_duration(data: Any, time_basis: str) -> float:
    durations = sample_durations(data.time_s)
    if time_basis == "active":
        return float(np.sum(durations * data.activity_mask))
    return float(data.time_s[-1] + durations[-1] - data.time_s[0])


def nested_windows(
    available_s: float,
    durations_s: Iterable[float],
    *,
    random_fraction: float,
) -> tuple[float, list[tuple[float, float, float]]]:
    durations = [float(value) for value in durations_s]
    maximum = max(durations)
    if available_s + 1e-9 < maximum:
        raise ValueError(f"Available duration {available_s:.3f}s is shorter than {maximum:.3f}s")
    half_maximum = maximum / 2.0
    low = half_maximum
    high = available_s - half_maximum
    center = low if high <= low else low + random_fraction * (high - low)
    windows = [(duration, center - duration / 2.0, center + duration / 2.0) for duration in durations]
    return center, windows


def git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run(
        "status", "--short", "--", "backend", "tools/mag_calibration_experiment.py",
        "tools/mag_calibration_sweep.py", "experiments/mag_calibration",
    )
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def build_schedule(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    durations = [float(value) for value in spec["durations_s"]]
    maximum = max(durations)
    seed = int(spec["seed"])
    time_basis = str(spec.get("time_basis", "active"))
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    expected_pipeline = spec.get("pipeline")

    for log_name in spec["logs"]:
        try:
            data = load_cached_log(str(log_name))
            available_s = available_duration(data, time_basis)
            if expected_pipeline and data.pipeline != expected_pipeline:
                raise ValueError(
                    f"cache is {data.pipeline!r}, expected {expected_pipeline!r}"
                )
            if available_s + 1e-9 < maximum:
                excluded.append({
                    "log": log_name,
                    "reason": "insufficient_duration",
                    "available_s": available_s,
                    "required_s": maximum,
                })
                continue
        except Exception as error:
            excluded.append({
                "log": log_name,
                "reason": "cache_error",
                "error_type": type(error).__name__,
                "error": str(error),
            })
            continue

        for repeat in range(int(spec["repeats"])):
            fraction = stable_uniform(SCHEMA_VERSION, seed, log_name, repeat)
            center, windows = nested_windows(
                available_s, durations, random_fraction=fraction
            )
            for duration, start, stop in windows:
                for trainer in spec["trainers"]:
                    identity = {
                        "schema_version": SCHEMA_VERSION,
                        "experiment": spec["name"],
                        "seed": seed,
                        "log": log_name,
                        "repeat": repeat,
                        "duration_s": duration,
                        "trainer": trainer,
                        "start_s": start,
                        "stop_s": stop,
                        "time_basis": time_basis,
                    }
                    rows.append({
                        "trial_id": stable_id(identity),
                        "experiment": spec["name"],
                        "pipeline": data.pipeline,
                        "log": log_name,
                        "repeat": repeat,
                        "duration_s": duration,
                        "trainer": trainer,
                        "time_basis": time_basis,
                        "random_fraction": fraction,
                        "center_s": center,
                        "start_s": start,
                        "stop_s": stop,
                        "available_s": available_s,
                        "cache_fingerprint": data.source_fingerprint,
                    })
    return rows, excluded


def prepare_run(spec_path: Path, output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec, spec_hash = read_spec(spec_path)
    manifest_path = output_dir / "manifest.json"
    schedule_path = output_dir / "trial_schedule.csv"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("spec_sha256") != spec_hash:
            raise ValueError(
                f"{output_dir} belongs to a different version of the experiment spec"
            )
        schedule_hash = sha256_bytes(schedule_path.read_bytes())
        if manifest.get("schedule_sha256") not in (None, schedule_hash):
            raise ValueError(f"The frozen trial schedule in {output_dir} has changed")
        if manifest.get("schedule_sha256") is None:
            manifest["schedule_sha256"] = schedule_hash
            atomic_write_json(manifest_path, manifest)
        schedule = list(csv.DictReader(schedule_path.open(encoding="utf-8")))
        schedule = [coerce_schedule_row(row) for row in schedule]
        return spec, schedule

    schedule, excluded = build_schedule(spec)
    if not schedule:
        raise ValueError("No eligible logs produced any scheduled trials")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(schedule_path, schedule)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "experiment": spec["name"],
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": spec_hash,
        "schedule_sha256": sha256_bytes(schedule_path.read_bytes()),
        "spec": spec,
        "trial_count": len(schedule),
        "eligible_logs": sorted({row["log"] for row in schedule}),
        "excluded_logs": excluded,
        "git": git_snapshot(),
        "activity_mask_note": (
            "Active time and evaluation currently use boring_mask, which is derived from "
            "reference travel and is therefore experimental infrastructure rather than a "
            "production activity detector."
        ),
    }
    atomic_write_json(manifest_path, manifest)
    return spec, schedule


def load_frozen_run(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = output_dir / "manifest.json"
    schedule_path = output_dir / "trial_schedule.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schedule_hash = sha256_bytes(schedule_path.read_bytes())
    expected_hash = manifest.get("schedule_sha256")
    if expected_hash not in (None, schedule_hash):
        raise ValueError(f"The frozen trial schedule in {output_dir} has changed")
    if expected_hash is None:
        manifest["schedule_sha256"] = schedule_hash
        atomic_write_json(manifest_path, manifest)
    schedule = [
        coerce_schedule_row(row)
        for row in csv.DictReader(schedule_path.open(encoding="utf-8"))
    ]
    return dict(manifest["spec"]), schedule


def coerce_schedule_row(row: dict[str, Any]) -> dict[str, Any]:
    converted = dict(row)
    converted["repeat"] = int(converted["repeat"])
    for key in ("duration_s", "random_fraction", "center_s", "start_s", "stop_s", "available_s"):
        converted[key] = float(converted[key])
    if converted.get("cache_fingerprint") == "":
        converted["cache_fingerprint"] = None
    return converted


def worker_data(log_name: str) -> Any:
    if log_name not in _WORKER_CACHE:
        _WORKER_CACHE[log_name] = load_cached_log(log_name)
    return _WORKER_CACHE[log_name]


def compute_trial_metrics(
    data: Any,
    row: dict[str, Any],
    calibration: MagTravelCalibration,
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    training = resolve_window(
        data.time_s,
        TimeRange(float(row["start_s"]), float(row["stop_s"])),
        time_basis=str(row["time_basis"]),
        activity_mask=data.activity_mask,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        _, prediction, anchor_offset = predict_calibration(calibration, data)
    full = resolve_window(
        data.time_s,
        TimeRange(),
        time_basis=str(row["time_basis"]),
        activity_mask=data.activity_mask,
    )
    training_mask = training.sample_mask(len(data.time_s))
    scope_parameters = {
        "training_window": (training, None),
        "full_log": (full, None),
        "full_log_excluding_training": (full, training_mask),
    }

    # Every scope also receives the single alignment offset estimated on the
    # full log. Comparing this with its locally centered RMSE quantifies how
    # much short-window centering itself helps.
    full_score = score_prediction(prediction, data, full)
    full_alignment_offset = float(full_score["aligned_offset_mm"])
    metrics: list[dict[str, Any]] = []
    for scope in spec.get("evaluation_scopes", sorted(SUPPORTED_SCOPES)):
        resolved, exclude = scope_parameters[scope]
        scored = score_prediction(
            prediction,
            data,
            resolved,
            exclude_mask=exclude,
            fixed_alignment_offset_mm=full_alignment_offset,
        )
        metrics.append({"evaluation_scope": scope, **scored})

    diagnostic_mask = (
        training_mask
        & data.activity_mask
        & np.isfinite(data.mag)
        & np.isfinite(data.travel)
    )
    diagnostics = {
        "requested_active_s": float(row["duration_s"]),
        "resolved_active_s": training.active_duration_s,
        "wall_start_s": training.wall_start_s,
        "wall_stop_s": training.wall_stop_s,
        "mag_min": float(np.min(data.mag[diagnostic_mask])),
        "mag_max": float(np.max(data.mag[diagnostic_mask])),
        "travel_min": float(np.min(data.travel[diagnostic_mask])),
        "travel_max": float(np.max(data.travel[diagnostic_mask])),
        "travel_range": float(np.ptp(data.travel[diagnostic_mask])),
        "travel_std": float(np.std(data.travel[diagnostic_mask])),
    }
    return metrics, anchor_offset, diagnostics


def execute_trial(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    started = utc_now()
    try:
        data = worker_data(str(row["log"]))
        window = RecordingWindow(
            log_name=str(row["log"]),
            time_range=TimeRange(float(row["start_s"]), float(row["stop_s"])),
            time_basis=str(row["time_basis"]),
        )
        calibration, training = fit_calibration(
            data,
            window,
            trainer=str(row["trainer"]),
            oracle_bins=int(spec.get("oracle_bins", 100)),
            verbose=False,
        )
        metrics, anchor_offset, diagnostics = compute_trial_metrics(
            data, row, calibration, spec
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "metrics_version": 2,
            "status": "success",
            "started_at": started,
            "finished_at": utc_now(),
            "trial": row,
            "calibration": calibration.to_dict(),
            "calibration_columns": calibration_columns(calibration),
            "target_anchor_offset_mm": anchor_offset,
            "training_window": diagnostics,
            "metrics": metrics,
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "started_at": started,
            "finished_at": utc_now(),
            "trial": row,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }


def trial_path(output_dir: Path, trial_id: str) -> Path:
    return output_dir / "trials" / f"{trial_id}.json"


def backfill_metrics(
    spec: dict[str, Any],
    schedule: list[dict[str, Any]],
    output_dir: Path,
    *,
    force: bool,
) -> tuple[int, list[dict[str, Any]]]:
    updated = 0
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(schedule, start=1):
        path = trial_path(output_dir, str(row["trial_id"]))
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "success":
            continue
        if not force and int(result.get("metrics_version", 1)) >= 2:
            continue
        try:
            data = worker_data(str(row["log"]))
            calibration = MagTravelCalibration.from_dict(result["calibration"])
            metrics, anchor_offset, diagnostics = compute_trial_metrics(
                data, row, calibration, spec
            )
            result["metrics_version"] = 2
            result["metrics_updated_at"] = utc_now()
            result["metrics"] = metrics
            result["target_anchor_offset_mm"] = anchor_offset
            result["training_window"] = diagnostics
            atomic_write_json(path, result)
            updated += 1
        except Exception as error:
            errors.append({
                "trial_id": row["trial_id"],
                "log": row["log"],
                "error_type": type(error).__name__,
                "error": str(error),
            })
        if index % 100 == 0:
            print(
                f"Scanned {index}/{len(schedule)} trials; updated {updated}, errors {len(errors)}",
                flush=True,
            )
    write_csv(output_dir / "metric_backfill_failures.csv", errors)
    return updated, errors


def run_trials(
    spec: dict[str, Any],
    schedule: list[dict[str, Any]],
    output_dir: Path,
    *,
    workers: int,
    max_trials: int | None,
    max_trials_per_log: int | None,
    retry_failures: bool,
) -> tuple[int, int]:
    pending: list[dict[str, Any]] = []
    for row in schedule:
        path = trial_path(output_dir, str(row["trial_id"]))
        if path.exists():
            if not retry_failures:
                continue
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("status") != "failed":
                continue
        pending.append(row)
    if max_trials_per_log is not None:
        counts: dict[str, int] = {}
        selected: list[dict[str, Any]] = []
        for row in pending:
            log_name = str(row["log"])
            if counts.get(log_name, 0) >= max_trials_per_log:
                continue
            selected.append(row)
            counts[log_name] = counts.get(log_name, 0) + 1
        pending = selected
    elif max_trials is not None:
        pending = pending[:max_trials]
    if not pending:
        return 0, 0

    completed = 0
    failed = 0
    if workers == 1:
        iterator = ((row, execute_trial(row, spec)) for row in pending)
        for row, result in iterator:
            atomic_write_json(trial_path(output_dir, str(row["trial_id"])), result)
            completed += 1
            failed += result["status"] == "failed"
            print(f"[{completed}/{len(pending)}] {row['trial_id']} {result['status']}", flush=True)
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(execute_trial, row, spec): row for row in pending}
                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        result = future.result()
                    except BaseException as error:
                        result = {
                            "schema_version": SCHEMA_VERSION,
                            "status": "failed",
                            "started_at": None,
                            "finished_at": utc_now(),
                            "trial": row,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        }
                    atomic_write_json(trial_path(output_dir, str(row["trial_id"])), result)
                    completed += 1
                    failed += result["status"] == "failed"
                    print(f"[{completed}/{len(pending)}] {row['trial_id']} {result['status']}", flush=True)
        except PermissionError as error:
            if completed:
                raise
            print(
                f"Process workers unavailable ({error}); falling back to one worker.",
                flush=True,
            )
            for row in pending:
                result = execute_trial(row, spec)
                atomic_write_json(trial_path(output_dir, str(row["trial_id"])), result)
                completed += 1
                failed += result["status"] == "failed"
                print(f"[{completed}/{len(pending)}] {row['trial_id']} {result['status']}", flush=True)
    return completed, failed


def finite_values(rows: Iterable[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def grouped(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    return groups


def bootstrap_ci(values: np.ndarray, *, seed_parts: tuple[object, ...], draws: int) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    seed = int(stable_uniform("bootstrap", *seed_parts) * (2**63 - 1))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    distribution = np.median(values[indices], axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return float(low), float(high)


def collect_results(
    output_dir: Path,
    schedule: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    metric_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    completed = 0
    for scheduled in schedule:
        path = trial_path(output_dir, str(scheduled["trial_id"]))
        if not path.exists():
            continue
        completed += 1
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "success":
            failures.append({
                **scheduled,
                "status": "failed",
                "error_type": result.get("error_type"),
                "error": result.get("error"),
            })
            continue
        base = {
            **scheduled,
            "status": "success",
            **result.get("calibration_columns", {}),
            "target_anchor_offset_mm": result.get("target_anchor_offset_mm"),
            **{f"window_{key}": value for key, value in result.get("training_window", {}).items()},
        }
        for metric in result.get("metrics", []):
            metric_rows.append({**base, **metric})
    return metric_rows, failures, completed


def summarize_results(
    spec: dict[str, Any],
    schedule: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    metric_rows, failures, completed = collect_results(output_dir, schedule)
    write_csv(output_dir / "trial_metrics.csv", metric_rows)
    write_csv(output_dir / "failures.csv", failures)

    scheduled_counts: dict[tuple[Any, ...], int] = {}
    for row in schedule:
        key = (row["pipeline"], row["trainer"], float(row["duration_s"]), row["log"])
        scheduled_counts[key] = scheduled_counts.get(key, 0) + 1
    successful_ids: dict[tuple[Any, ...], set[str]] = {}
    for row in metric_rows:
        key = (row["pipeline"], row["trainer"], float(row["duration_s"]), row["log"])
        successful_ids.setdefault(key, set()).add(str(row["trial_id"]))
    failed_ids: dict[tuple[Any, ...], set[str]] = {}
    for row in failures:
        key = (row["pipeline"], row["trainer"], float(row["duration_s"]), row["log"])
        failed_ids.setdefault(key, set()).add(str(row["trial_id"]))

    log_keys = ("pipeline", "trainer", "duration_s", "evaluation_scope", "log")
    per_log: list[dict[str, Any]] = []
    metric_groups = grouped(metric_rows, log_keys)
    result_group_keys = set(metric_groups)
    for pipeline, trainer, duration, log_name in failed_ids:
        for scope in spec.get("evaluation_scopes", sorted(SUPPORTED_SCOPES)):
            result_group_keys.add((pipeline, trainer, duration, scope, log_name))
    for key in sorted(result_group_keys, key=str):
        rows = metric_groups.get(key, [])
        pipeline, trainer, duration, scope, log_name = key
        count_key = (pipeline, trainer, float(duration), log_name)
        scheduled_count = scheduled_counts[count_key]
        successful_count = len(successful_ids.get(count_key, set()))
        failed_count = len(failed_ids.get(count_key, set()))
        completed_count = successful_count + failed_count
        summary: dict[str, Any] = {
            "pipeline": pipeline,
            "trainer": trainer,
            "duration_s": float(duration),
            "evaluation_scope": scope,
            "log": log_name,
            "scheduled_trials": scheduled_count,
            "completed_trials": completed_count,
            "successful_trials": successful_count,
            "failed_trials": failed_count,
            "completion_rate": completed_count / scheduled_count,
            "failure_rate": failed_count / completed_count if completed_count else float("nan"),
        }
        for metric in SUMMARY_METRICS:
            values = finite_values(rows, metric)
            summary[f"{metric}_median"] = float(np.median(values)) if len(values) else float("nan")
            summary[f"{metric}_mean"] = float(np.mean(values)) if len(values) else float("nan")
            summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        per_log.append(summary)
    write_csv(output_dir / "per_log_summary.csv", per_log)

    aggregate_keys = ("pipeline", "trainer", "duration_s", "evaluation_scope")
    aggregate: list[dict[str, Any]] = []
    draws = int(spec.get("bootstrap_draws", 2000))
    for key, rows in sorted(grouped(per_log, aggregate_keys).items(), key=lambda item: str(item[0])):
        pipeline, trainer, duration, scope = key
        summary = {
            "pipeline": pipeline,
            "trainer": trainer,
            "duration_s": float(duration),
            "evaluation_scope": scope,
            "n_logs": len(rows),
            "successful_trials": int(sum(int(row["successful_trials"]) for row in rows)),
            "failed_trials": int(sum(int(row["failed_trials"]) for row in rows)),
            "completed_trials": int(sum(int(row["completed_trials"]) for row in rows)),
            "scheduled_trials": int(sum(int(row["scheduled_trials"]) for row in rows)),
        }
        summary["completion_rate"] = summary["completed_trials"] / summary["scheduled_trials"]
        summary["failure_rate"] = (
            summary["failed_trials"] / summary["completed_trials"]
            if summary["completed_trials"] else float("nan")
        )
        for metric in SUMMARY_METRICS:
            values = finite_values(rows, f"{metric}_median")
            median = float(np.median(values)) if len(values) else float("nan")
            mean = float(np.mean(values)) if len(values) else float("nan")
            low, high = bootstrap_ci(
                values,
                seed_parts=(spec["seed"], pipeline, trainer, duration, scope, metric),
                draws=draws,
            )
            summary[f"{metric}_log_median"] = median
            summary[f"{metric}_log_mean"] = mean
            summary[f"{metric}_ci95_low"] = low
            summary[f"{metric}_ci95_high"] = high
        aggregate.append(summary)
    write_csv(output_dir / "aggregate_summary.csv", aggregate)
    create_plots(aggregate, output_dir)
    write_report(spec, schedule, completed, failures, aggregate, output_dir)
    status = {
        "updated_at": utc_now(),
        "scheduled_trials": len(schedule),
        "completed_trials": completed,
        "successful_trials": completed - len(failures),
        "failed_trials": len(failures),
        "remaining_trials": len(schedule) - completed,
        "complete": completed == len(schedule),
    }
    atomic_write_json(output_dir / "status.json", status)
    return status


def create_plots(aggregate: list[dict[str, Any]], output_dir: Path) -> None:
    if not aggregate:
        return
    scopes = [
        scope for scope in (
            "training_window", "full_log", "full_log_excluding_training"
        ) if any(row["evaluation_scope"] == scope for row in aggregate)
    ]
    trainers = sorted({str(row["trainer"]) for row in aggregate})
    colors = plt.get_cmap("tab10")
    figure, axes = plt.subplots(1, len(scopes), figsize=(5.2 * len(scopes), 4.2), squeeze=False)
    for axis, scope in zip(axes[0], scopes):
        for index, trainer in enumerate(trainers):
            rows = sorted(
                (row for row in aggregate if row["evaluation_scope"] == scope and row["trainer"] == trainer),
                key=lambda row: float(row["duration_s"]),
            )
            if not rows:
                continue
            x = np.asarray([float(row["duration_s"]) for row in rows])
            y = np.asarray([float(row["aligned_rmse_log_median"]) for row in rows])
            low = np.asarray([float(row["aligned_rmse_ci95_low"]) for row in rows])
            high = np.asarray([float(row["aligned_rmse_ci95_high"]) for row in rows])
            axis.plot(x, y, marker="o", label=trainer, color=colors(index))
            axis.fill_between(x, low, high, alpha=0.16, color=colors(index))
        axis.set_xscale("log")
        axis.set_xlabel("Training-window active duration (s)")
        axis.set_ylabel("Aligned RMSE (mm)")
        axis.set_title(scope.replace("_", " ").title())
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False)
    partial = any(float(row["completion_rate"]) < 1.0 for row in aggregate)
    suffix = " (partial run; descriptive only)" if partial else ""
    figure.suptitle(f"Mag-to-travel calibration learning curve{suffix}")
    figure.tight_layout()
    figure.savefig(output_dir / "learning_curve.png", dpi=180)
    figure.savefig(output_dir / "learning_curve.pdf")
    plt.close(figure)


def format_number(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "—"


def write_report(
    spec: dict[str, Any],
    schedule: list[dict[str, Any]],
    completed: int,
    failures: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    lines = [
        f"# {spec['name']}",
        "",
        f"Generated: {utc_now()}",
        "",
        f"Completed {completed} of {len(schedule)} scheduled fits; {len(failures)} failed.",
        "",
        *(
            ["**Partial smoke output: do not interpret these values as experiment results.**", ""]
            if completed < len(schedule)
            else []
        ),
        "Windows are deterministic, randomly centered, and nested across durations within each log/repeat. "
        "Each trainer receives the identical window. Active time and scoring currently use `boring_mask`.",
        "",
    ]
    for scope in ("training_window", "full_log", "full_log_excluding_training"):
        rows = [row for row in aggregate if row["evaluation_scope"] == scope]
        if not rows:
            continue
        lines.extend([
            f"## {scope.replace('_', ' ').title()}",
            "",
            "| Trainer | Duration (s) | Logs | Aligned RMSE median (mm) | 95% log-bootstrap CI | Complete | Failure rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in sorted(rows, key=lambda item: (str(item["trainer"]), float(item["duration_s"]))):
            lines.append(
                "| {trainer} | {duration:g} | {logs} | {rmse} | [{low}, {high}] | {completion:.1%} | {failure:.1%} |".format(
                    trainer=row["trainer"],
                    duration=float(row["duration_s"]),
                    logs=row["n_logs"],
                    rmse=format_number(row["aligned_rmse_log_median"]),
                    low=format_number(row["aligned_rmse_ci95_low"]),
                    high=format_number(row["aligned_rmse_ci95_high"]),
                    completion=float(row["completion_rate"]),
                    failure=float(row["failure_rate"]),
                )
            )
        lines.append("")
    lines.extend([
        "## Interpretation notes",
        "",
        "- `training_window` measures local reconstruction on the same self-supervised data block.",
        "- `full_log` measures how much data is needed for a calibration that represents the recording as a whole.",
        "- `full_log_excluding_training` removes direct sample overlap while staying within the same recording.",
        "- `oracle-power` uses reference travel but the production power-curve family; `oracle-isotonic` is a more flexible ceiling.",
        "- Aggregates first take a median across repeats within each log, then weight logs equally.",
        "",
    ])
    atomic_write_text(output_dir / "report.md", "\n".join(lines))


def default_output_dir(spec: dict[str, Any]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "experiments" / "mag_calibration" / "runs" / f"{stamp}-{spec['name']}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("schedule", "run"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("spec", type=Path)
        subparser.add_argument("--output-dir", type=Path)
        if command == "run":
            subparser.add_argument("--workers", type=int, default=1)
            limit = subparser.add_mutually_exclusive_group()
            limit.add_argument("--max-trials", type=int)
            limit.add_argument("--max-trials-per-log", type=int)
            subparser.add_argument("--retry-failures", action="store_true")
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("spec", type=Path)
    summarize.add_argument("--output-dir", type=Path, required=True)
    refresh = subparsers.add_parser(
        "refresh",
        help="Backfill current metrics and summaries from a run's frozen manifest",
    )
    refresh.add_argument("--output-dir", type=Path, required=True)
    refresh.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "refresh":
        spec, schedule = load_frozen_run(args.output_dir)
        updated, errors = backfill_metrics(
            spec, schedule, args.output_dir, force=args.force
        )
        status = summarize_results(spec, schedule, args.output_dir)
        print(f"Updated metrics for {updated} trials ({len(errors)} backfill errors)")
        print(json.dumps(status, indent=2))
        return
    spec, _ = read_spec(args.spec)
    output_dir = args.output_dir or default_output_dir(spec)
    spec, schedule = prepare_run(args.spec, output_dir)
    if args.command == "schedule":
        print(f"Scheduled {len(schedule)} trials in {output_dir}")
        return
    if args.command == "run":
        if args.workers < 1:
            raise ValueError("--workers must be at least 1")
        if args.max_trials is not None and args.max_trials < 1:
            raise ValueError("--max-trials must be at least 1")
        if args.max_trials_per_log is not None and args.max_trials_per_log < 1:
            raise ValueError("--max-trials-per-log must be at least 1")
        completed, failed = run_trials(
            spec,
            schedule,
            output_dir,
            workers=args.workers,
            max_trials=args.max_trials,
            max_trials_per_log=args.max_trials_per_log,
            retry_failures=args.retry_failures,
        )
        print(f"Executed {completed} fits ({failed} failed)")
    status = summarize_results(spec, schedule, output_dir)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
