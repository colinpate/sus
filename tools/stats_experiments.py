"""Durable experiment records and cache-freshness checks for pipeline statistics."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np

from backend.log_registry import ResolvedLog
from backend.run_provenance import RunProvenance, build_run_provenance, sha256_json


SCHEMA_VERSION = 2
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "stats"
MANIFEST_FILENAME = "experiment.json"
METRICS_FILENAME = "metrics.csv"
LOGS_FILENAME = "logs.csv"
REPORT_FILENAME = "report.txt"
MetricKey = tuple[str, str, str, str, str]


@dataclass(frozen=True)
class CacheInspection:
    log: ResolvedLog
    status: str
    reason: str
    expected: RunProvenance | None
    recorded: dict[str, Any] | None
    cache_fingerprint: str | None

    @property
    def fresh(self) -> bool:
        return self.status == "fresh"

    def manifest_record(self) -> dict[str, Any]:
        provenance = self.recorded or (self.expected.to_dict() if self.expected else {})
        input_info = provenance.get("input", {})
        return {
            "log": self.log.log_id,
            "pipeline": self.log.pipeline,
            "status": self.status,
            "reason": self.reason,
            "run_fingerprint": provenance.get("run_fingerprint"),
            "input_sha256": input_info.get("sha256"),
            "resolved_config_sha256": provenance.get("resolved_config_sha256"),
            "pipeline_code_sha256": provenance.get("pipeline_code_sha256"),
            "assets_sha256": provenance.get("assets_sha256"),
            "environment_sha256": provenance.get("environment_sha256"),
            "processed_at": provenance.get("generated_at"),
            "profiles": list(self.log.profiles),
            "sets": list(self.log.sets),
            "tags": list(self.log.tags),
            "metadata": self.log.metadata,
            "git": provenance.get("git", {}),
        }


def cache_paths(log_id: str, cache_root: Path) -> tuple[Path, Path]:
    run_dir = cache_root / log_id
    return run_dir / "run.json", run_dir / "cache" / "all.npz"


def inspect_cache(
    log: ResolvedLog,
    cache_root: Path,
    *,
    expected: RunProvenance | None = None,
) -> CacheInspection:
    """Verify the run manifest, all-cache, and current input/config/code fingerprint agree."""

    try:
        expected = expected or build_run_provenance(log, pipeline=log.pipeline)
    except Exception as exc:
        return CacheInspection(log, "unverifiable", str(exc), None, None, None)

    run_path, cache_path = cache_paths(log.log_id, cache_root)
    if not run_path.is_file():
        return CacheInspection(log, "missing-run", f"missing {run_path}", expected, None, None)

    try:
        recorded = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return CacheInspection(log, "invalid-run", f"cannot read {run_path}: {exc}", expected, None, None)

    if recorded.get("status") != "success":
        return CacheInspection(log, "failed-run", f"run status is {recorded.get('status')!r}", expected, recorded, None)
    if recorded.get("log") != log.log_id or recorded.get("pipeline") != log.pipeline:
        return CacheInspection(log, "wrong-run", "run manifest identifies a different log or pipeline", expected, recorded, None)
    if recorded.get("run_fingerprint") != expected.run_fingerprint:
        return CacheInspection(
            log,
            "stale",
            "input, processing config, pipeline code/assets, or runtime differs from the cached run",
            expected,
            recorded,
            None,
        )
    if not cache_path.is_file():
        return CacheInspection(log, "missing-cache", f"missing {cache_path}", expected, recorded, None)

    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            cache_fingerprint = str(cache["__run_fingerprint"].item()) if "__run_fingerprint" in cache else None
    except Exception as exc:
        return CacheInspection(log, "invalid-cache", f"cannot read {cache_path}: {exc}", expected, recorded, None)

    if cache_fingerprint is None:
        return CacheInspection(log, "unversioned-cache", "cache has no run fingerprint", expected, recorded, None)
    if cache_fingerprint != expected.run_fingerprint:
        return CacheInspection(log, "cache-mismatch", "cache fingerprint does not match run manifest", expected, recorded, cache_fingerprint)
    return CacheInspection(log, "fresh", "matches current input, config, code/assets, and runtime", expected, recorded, cache_fingerprint)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "experiment"


def make_experiment_id(name: str, *, now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{slugify(name)}"


def experiment_fingerprint(
    *,
    name: str,
    selection: Mapping[str, Any],
    stats_config: Mapping[str, Any],
    logs: Iterable[Mapping[str, Any]],
) -> str:
    return sha256_json(
        {
            "schema_version": SCHEMA_VERSION,
            "name": name,
            "selection": selection,
            "stats": stats_config,
            "logs": [
                {
                    "log": record.get("log"),
                    "run_fingerprint": record.get("run_fingerprint"),
                }
                for record in logs
            ],
        }
    )


def write_metrics(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    fieldnames = ["centering", "section", "log", "comparison", "metric", "value"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_log_records(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    fieldnames = [
        "log",
        "pipeline",
        "status",
        "run_fingerprint",
        "input_sha256",
        "resolved_config_sha256",
        "pipeline_code_sha256",
        "assets_sha256",
        "environment_sha256",
        "processed_at",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_metrics(experiment_dir: Path) -> dict[MetricKey, float]:
    path = experiment_dir / METRICS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Expected experiment metrics at {path}")
    metrics: dict[MetricKey, float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                row.get("centering", ""),
                row.get("section", ""),
                row.get("log", ""),
                row.get("comparison", ""),
                row.get("metric", ""),
            )
            metrics[key] = float(row.get("value", "nan"))
    return metrics


def iter_experiments(root: Path = DEFAULT_EXPERIMENT_ROOT) -> list[tuple[Path, dict[str, Any]]]:
    experiments: list[tuple[Path, dict[str, Any]]] = []
    if not root.is_dir():
        return experiments
    for manifest_path in root.glob(f"*/{MANIFEST_FILENAME}"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        experiments.append((manifest_path.parent, manifest))
    experiments.sort(key=lambda item: str(item[1].get("created_at", "")), reverse=True)
    return experiments


def resolve_experiment(reference: str | Path, root: Path = DEFAULT_EXPERIMENT_ROOT) -> tuple[Path, dict[str, Any]]:
    candidate = Path(reference)
    if candidate.is_dir() and (candidate / MANIFEST_FILENAME).is_file():
        path = candidate
        return path, json.loads((path / MANIFEST_FILENAME).read_text(encoding="utf-8"))

    text = str(reference)
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path, manifest in iter_experiments(root):
        experiment_id = str(manifest.get("id", path.name))
        name = str(manifest.get("name", ""))
        if text in {experiment_id, path.name, name} or experiment_id.startswith(text):
            matches.append((path, manifest))
    if not matches:
        raise FileNotFoundError(f"No saved experiment matches {text!r} under {root}")
    return matches[0]


def unique_values(records: Iterable[Mapping[str, Any]], key: str) -> list[str]:
    return sorted({str(record[key]) for record in records if record.get(key)})


def primary_metric_values(
    metrics: Mapping[MetricKey, float],
    *,
    centering: str,
    section: str = "error",
    comparison: str = "travel/solved",
    metric: str = "rmse",
) -> dict[str, float]:
    values: dict[str, float] = {}
    for (row_centering, row_section, log_id, row_comparison, row_metric), value in metrics.items():
        if (
            row_centering == centering
            and row_section == section
            and row_comparison == comparison
            and row_metric == metric
            and log_id
        ):
            values[log_id] = value
    return values


def aggregate(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(array) == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
    }
