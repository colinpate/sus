#!/usr/bin/env python3
"""Resumable, spec-driven cross-log mag-calibration transfer experiment."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tomllib
from typing import Any

os.environ["MPLCONFIGDIR"] = "/private/tmp"

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT / "backend", REPO_ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from mag_calibration import MagTravelCalibration, RecordingWindow, TimeRange, resolve_window  # noqa: E402
from mag_calibration_experiment import (  # noqa: E402
    calibration_columns,
    fit_calibration,
    load_cached_log,
    predict_calibration,
    score_prediction,
)
from mag_calibration_sweep import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    git_snapshot,
    sha256_bytes,
    stable_id,
    utc_now,
    write_csv,
)


SCHEMA_VERSION = 1
SUPPORTED_TRAINERS = {
    "self-supervised",
    "oracle-power",
    "oracle-isotonic",
    "oracle-binned-median",
}
_DATA_CACHE: dict[str, Any] = {}


def load_data(log_name: str) -> Any:
    if log_name not in _DATA_CACHE:
        _DATA_CACHE[log_name] = load_cached_log(log_name)
    return _DATA_CACHE[log_name]


def read_spec(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    spec = tomllib.loads(raw.decode("utf-8"))
    required = ("name", "pipeline", "logs", "trainers")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"Transfer spec is missing: {', '.join(missing)}")
    if int(spec.get("schema_version", 1)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported transfer schema {spec.get('schema_version')!r}")
    if spec.get("experiment_type") != "cross_log":
        raise ValueError("Transfer specs require experiment_type = 'cross_log'")
    if spec.get("training_window", "full_log") != "full_log":
        raise ValueError("Only full-log calibration training is currently supported")
    if spec.get("evaluation_window", "full_log") != "full_log":
        raise ValueError("Only full-log transfer evaluation is currently supported")
    if len(spec["logs"]) < 2:
        raise ValueError("Cross-log transfer requires at least two logs")
    unknown = set(spec["trainers"]) - SUPPORTED_TRAINERS
    if unknown:
        raise ValueError(f"Unknown trainers: {sorted(unknown)}")
    return spec, sha256_bytes(raw)


def build_schedule(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_logs: list[str] = []
    excluded: list[dict[str, Any]] = []
    for log_name in spec["logs"]:
        try:
            data = load_data(str(log_name))
            if data.pipeline != spec["pipeline"]:
                raise ValueError(f"cache pipeline is {data.pipeline!r}")
            valid_logs.append(str(log_name))
        except Exception as error:
            excluded.append({
                "log": log_name,
                "error_type": type(error).__name__,
                "error": str(error),
            })
    if len(valid_logs) < 2:
        raise ValueError("Fewer than two compatible logs remain")

    rows: list[dict[str, Any]] = []
    include_diagonal = bool(spec.get("include_diagonal", True))
    for trainer in spec["trainers"]:
        for train_log in valid_logs:
            calibration_id = stable_id({
                "experiment": spec["name"],
                "trainer": trainer,
                "train_log": train_log,
                "training_window": "full_log",
            })
            train_data = load_data(train_log)
            for eval_log in valid_logs:
                if not include_diagonal and train_log == eval_log:
                    continue
                eval_data = load_data(eval_log)
                identity = {
                    "experiment": spec["name"],
                    "trainer": trainer,
                    "train_log": train_log,
                    "eval_log": eval_log,
                }
                rows.append({
                    "trial_id": stable_id(identity),
                    "calibration_id": calibration_id,
                    "experiment": spec["name"],
                    "pipeline": spec["pipeline"],
                    "trainer": trainer,
                    "train_log": train_log,
                    "eval_log": eval_log,
                    "pair_type": "diagonal" if train_log == eval_log else "transfer",
                    "time_basis": spec.get("time_basis", "active"),
                    "train_cache_fingerprint": train_data.source_fingerprint,
                    "eval_cache_fingerprint": eval_data.source_fingerprint,
                })
    return rows, excluded


def schedule_hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def prepare_run(spec_path: Path, output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec, spec_hash = read_spec(spec_path)
    manifest_path = output_dir / "manifest.json"
    schedule_path = output_dir / "trial_schedule.csv"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("spec_sha256") != spec_hash:
            raise ValueError(f"{output_dir} belongs to a different spec revision")
        if manifest.get("schedule_sha256") != schedule_hash(schedule_path):
            raise ValueError("Frozen transfer schedule has changed")
        return spec, list(csv.DictReader(schedule_path.open(encoding="utf-8")))

    rows, excluded = build_schedule(spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(schedule_path, rows)
    atomic_write_json(output_dir / "manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "experiment": spec["name"],
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": spec_hash,
        "schedule_sha256": schedule_hash(schedule_path),
        "spec": spec,
        "trial_count": len(rows),
        "calibration_count": len({row["calibration_id"] for row in rows}),
        "eligible_logs": sorted({row["train_log"] for row in rows}),
        "excluded_logs": excluded,
        "git": git_snapshot(),
        "activity_mask_note": "Evaluation uses the reference-derived boring_mask.",
    })
    return spec, rows


def calibration_path(output_dir: Path, calibration_id: str) -> Path:
    return output_dir / "calibrations" / f"{calibration_id}.json"


def fit_or_load_calibration(
    row: dict[str, Any], spec: dict[str, Any], output_dir: Path
) -> MagTravelCalibration:
    path = calibration_path(output_dir, row["calibration_id"])
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "success":
            raise ValueError(payload.get("error", "Calibration fit previously failed"))
        return MagTravelCalibration.from_dict(payload["calibration"])

    data = load_data(row["train_log"])
    window = RecordingWindow(
        log_name=row["train_log"],
        time_range=TimeRange(),
        time_basis=row["time_basis"],
    )
    try:
        calibration, resolved = fit_calibration(
            data,
            window,
            trainer=row["trainer"],
            oracle_bins=int(spec.get("oracle_bins", 100)),
            verbose=False,
        )
        payload = {
            "status": "success",
            "created_at": utc_now(),
            "calibration_id": row["calibration_id"],
            "trainer": row["trainer"],
            "train_log": row["train_log"],
            "train_active_s": resolved.active_duration_s,
            "calibration": calibration.to_dict(),
            "calibration_columns": calibration_columns(calibration),
        }
        atomic_write_json(path, payload)
        return calibration
    except Exception as error:
        atomic_write_json(path, {
            "status": "failed",
            "created_at": utc_now(),
            "calibration_id": row["calibration_id"],
            "trainer": row["trainer"],
            "train_log": row["train_log"],
            "error_type": type(error).__name__,
            "error": str(error),
        })
        raise


def execute_trial(
    row: dict[str, Any], spec: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    try:
        calibration = fit_or_load_calibration(row, spec, output_dir)
        target = load_data(row["eval_log"])
        with contextlib.redirect_stdout(io.StringIO()):
            _, prediction, anchor_offset = predict_calibration(calibration, target)
        resolved = resolve_window(
            target.time_s,
            TimeRange(),
            time_basis=row["time_basis"],
            activity_mask=target.activity_mask,
        )
        metrics = score_prediction(prediction, target, resolved)
        return {
            "schema_version": SCHEMA_VERSION,
            "metrics_version": 2,
            "status": "success",
            "finished_at": utc_now(),
            "trial": row,
            "target_anchor_offset_mm": anchor_offset,
            "calibration_columns": calibration_columns(calibration),
            "metrics": metrics,
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "finished_at": utc_now(),
            "trial": row,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def trial_path(output_dir: Path, trial_id: str) -> Path:
    return output_dir / "trials" / f"{trial_id}.json"


def run_trials(
    rows: list[dict[str, Any]],
    spec: dict[str, Any],
    output_dir: Path,
    max_trials: int | None,
) -> tuple[int, int]:
    pending = [row for row in rows if not trial_path(output_dir, row["trial_id"]).exists()]
    if max_trials is not None:
        pending = pending[:max_trials]
    failed = 0
    for index, row in enumerate(pending, start=1):
        result = execute_trial(row, spec, output_dir)
        atomic_write_json(trial_path(output_dir, row["trial_id"]), result)
        failed += result["status"] == "failed"
        print(
            f"[{index}/{len(pending)}] {row['trainer']} {row['train_log']} -> "
            f"{row['eval_log']}: {result['status']}",
            flush=True,
        )
    return len(pending), failed


def finite(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def summarize(
    rows: list[dict[str, Any]], spec: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    completed = 0
    for row in rows:
        path = trial_path(output_dir, row["trial_id"])
        if not path.exists():
            continue
        completed += 1
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "success":
            failures.append({**row, "error_type": result.get("error_type"), "error": result.get("error")})
            continue
        metrics.append({
            **row,
            **result.get("calibration_columns", {}),
            "target_anchor_offset_mm": result.get("target_anchor_offset_mm"),
            **result["metrics"],
        })
    write_csv(output_dir / "trial_metrics.csv", metrics)
    write_csv(output_dir / "failures.csv", failures)

    source_rows: list[dict[str, Any]] = []
    for trainer in spec["trainers"]:
        for train_log in spec["logs"]:
            for pair_type in ("diagonal", "transfer"):
                selected = [row for row in metrics if row["trainer"] == trainer and row["train_log"] == train_log and row["pair_type"] == pair_type]
                if not selected:
                    continue
                source_rows.append({
                    "trainer": trainer,
                    "train_log": train_log,
                    "pair_type": pair_type,
                    "evaluation_logs": len(selected),
                    "aligned_rmse_median": float(np.median(finite(selected, "aligned_rmse"))),
                    "anchored_rmse_median": float(np.median(finite(selected, "anchored_rmse"))),
                    "aligned_nrmse_std_median": float(np.median(finite(selected, "aligned_nrmse_std"))),
                })
    write_csv(output_dir / "source_summary.csv", source_rows)

    aggregate: list[dict[str, Any]] = []
    for trainer in spec["trainers"]:
        for pair_type in ("diagonal", "transfer"):
            selected = [row for row in source_rows if row["trainer"] == trainer and row["pair_type"] == pair_type]
            if not selected:
                continue
            aggregate.append({
                "trainer": trainer,
                "pair_type": pair_type,
                "source_logs": len(selected),
                "aligned_rmse_source_median": float(np.median(finite(selected, "aligned_rmse_median"))),
                "anchored_rmse_source_median": float(np.median(finite(selected, "anchored_rmse_median"))),
                "aligned_nrmse_std_source_median": float(np.median(finite(selected, "aligned_nrmse_std_median"))),
            })
    write_csv(output_dir / "aggregate_summary.csv", aggregate)

    # Compare every off-diagonal shared calibration with the target log's own
    # self-supervised diagonal. Positive deltas favor per-log calibration.
    baselines = {
        (row["trainer"], row["eval_log"]): float(row["aligned_rmse"])
        for row in metrics
        if row["pair_type"] == "diagonal"
    }
    comparisons: list[dict[str, Any]] = []
    for trainer in spec["trainers"]:
        for baseline_trainer in dict.fromkeys((trainer, "self-supervised")):
            selected = [
                row for row in metrics
                if row["trainer"] == trainer
                and row["pair_type"] == "transfer"
                and (baseline_trainer, row["eval_log"]) in baselines
            ]
            deltas = np.asarray([
                float(row["aligned_rmse"])
                - baselines[(baseline_trainer, row["eval_log"])]
                for row in selected
            ])
            if len(deltas):
                comparisons.append({
                    "transfer_trainer": trainer,
                    "baseline_trainer": baseline_trainer,
                    "comparison": f"transfer minus target {baseline_trainer} diagonal",
                    "pairs": len(deltas),
                    "median_delta_mm": float(np.median(deltas)),
                    "fraction_baseline_better": float(np.mean(deltas > 0)),
                })
    write_csv(output_dir / "comparison_summary.csv", comparisons)

    complete = completed == len(rows)
    lines = [
        f"# {spec['name']}",
        "",
        f"Completed {completed} of {len(rows)} train/evaluation pairs; {len(failures)} failed.",
        "",
    ]
    if not complete:
        lines.extend(["**Partial smoke output: do not interpret as final experiment results.**", ""])
    lines.extend([
        "Each source calibration is trained once on its complete source log. Diagonal pairs are the per-log baseline; off-diagonal pairs test transfer to another log. Aggregates first summarize target logs within each source calibration, then weight source logs equally.",
        "",
        "| Trainer | Pair type | Source logs | Aligned RMSE (mm) | Anchored RMSE (mm) |",
        "|---|---|---:|---:|---:|",
    ])
    for row in aggregate:
        lines.append(
            f"| {row['trainer']} | {row['pair_type']} | {row['source_logs']} | "
            f"{row['aligned_rmse_source_median']:.3f} | {row['anchored_rmse_source_median']:.3f} |"
        )
    lines.extend([
        "",
        "The diagonal must be retained: it controls for each log's intrinsic difficulty. Use `comparison_summary.csv` for the direct shared-calibration versus target per-log self-supervised comparison.",
        "",
        "`boring_mask` is still used for evaluation, and oracle trainers use reference travel only from their source training log.",
        "",
    ])
    atomic_write_text(output_dir / "report.md", "\n".join(lines))
    status = {
        "updated_at": utc_now(),
        "scheduled_trials": len(rows),
        "completed_trials": completed,
        "successful_trials": completed - len(failures),
        "failed_trials": len(failures),
        "remaining_trials": len(rows) - completed,
        "complete": complete,
    }
    atomic_write_json(output_dir / "status.json", status)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("schedule", "run", "summarize"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("spec", type=Path)
        subparser.add_argument("--output-dir", type=Path, required=True)
        if command == "run":
            subparser.add_argument("--max-trials", type=int)
            subparser.add_argument("--trainer", choices=sorted(SUPPORTED_TRAINERS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec, rows = prepare_run(args.spec, args.output_dir)
    if args.command == "schedule":
        print(
            f"Scheduled {len(rows)} transfer evaluations using "
            f"{len({row['calibration_id'] for row in rows})} source calibrations"
        )
        return
    if args.command == "run":
        run_rows = rows if args.trainer is None else [
            row for row in rows if row["trainer"] == args.trainer
        ]
        completed, failed = run_trials(run_rows, spec, args.output_dir, args.max_trials)
        print(f"Executed {completed} evaluations ({failed} failed)")
    print(json.dumps(summarize(rows, spec, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
