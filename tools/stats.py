#!/usr/bin/env python3
"""Create, catalog, inspect, and compare versioned pipeline-stat experiments."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.log_registry import DEFAULT_REGISTRY_PATH, LogRegistry, ResolvedLog, load_registry  # noqa: E402
from backend.run_provenance import sha256_file  # noqa: E402
from tools import stats_aggregator as metrics_engine  # noqa: E402
from tools.stats_experiments import (  # noqa: E402
    DEFAULT_EXPERIMENT_ROOT,
    LOGS_FILENAME,
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    REPORT_FILENAME,
    SCHEMA_VERSION,
    CacheInspection,
    aggregate,
    experiment_fingerprint,
    inspect_cache,
    iter_experiments,
    load_metrics,
    make_experiment_id,
    primary_metric_values,
    resolve_experiment,
    unique_values,
    write_log_records,
    write_metrics,
)


DEFAULT_CACHE_ROOT = REPO_ROOT / "backend" / "run_artifacts"
CENTERING_MODES = ("uncentered", "centered")


def project_python() -> Path:
    candidate = REPO_ROOT / "venv" / "bin" / "python3"
    return candidate if candidate.is_file() else Path(sys.executable)


def ensure_project_python() -> None:
    """Keep pipeline execution and freshness fingerprints in the same Python environment."""

    candidate = project_python().absolute()
    if candidate != Path(sys.executable).absolute():
        os.execv(str(candidate), [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]])


def parse_filters(values: Iterable[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected KEY=VALUE for --where, got {value!r}")
        key, expected = value.split("=", 1)
        if not key:
            raise ValueError(f"Expected a non-empty key in {value!r}")
        filters[key] = expected
    return filters


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def select_logs(args: argparse.Namespace, registry: LogRegistry) -> tuple[list[ResolvedLog], dict[str, Any]]:
    filters = parse_filters(args.where)
    logs = registry.select(
        log_ids=args.logs,
        set_name=args.set_name,
        filters=filters,
        usable_only=not args.all_statuses,
    )
    selection = {
        "registry": relative_path(registry.path),
        "registry_sha256": sha256_file(registry.path),
        "explicit_logs": list(args.logs),
        "set": args.set_name,
        "where": filters,
        "usable_only": not args.all_statuses,
        "selected_logs": [log.log_id for log in logs],
    }
    return logs, selection


def pipeline_script(log: ResolvedLog) -> Path:
    if log.pipeline == "front":
        return REPO_ROOT / "backend" / "pipeline.py"
    if log.pipeline == "rear":
        return REPO_ROOT / "backend" / "pipeline_rear.py"
    raise ValueError(f"{log.log_id}: unsupported pipeline {log.pipeline!r}")


def process_log(log: ResolvedLog) -> None:
    matplotlib_dir = Path(tempfile.gettempdir()) / "sus-matplotlib-cache"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
    subprocess.run(
        [str(project_python()), str(pipeline_script(log)), log.log_id],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )


def inspect_logs(logs: Iterable[ResolvedLog], cache_root: Path) -> list[CacheInspection]:
    inspections: list[CacheInspection] = []
    for log in logs:
        print(f"Checking {log.log_id}...", flush=True)
        inspections.append(inspect_cache(log, cache_root))
    return inspections


def print_nonfresh(inspections: Iterable[CacheInspection]) -> None:
    rows = [inspection for inspection in inspections if not inspection.fresh]
    if not rows:
        return
    print("Caches that cannot be used:")
    for inspection in rows:
        print(f"  {inspection.log.log_id:<24} {inspection.status:<18} {inspection.reason}")


def collect_both_modes(
    log_ids: list[str],
    cache_root: Path,
    *,
    error_threshold: float | None,
    deep_dive: bool,
    sort_key: str,
) -> tuple[dict[str, metrics_engine.AggregatedReport], list[dict[str, Any]], str]:
    reports: dict[str, metrics_engine.AggregatedReport] = {}
    metric_rows: list[dict[str, Any]] = []
    report_parts: list[str] = []

    for centering in CENTERING_MODES:
        centered = centering == "centered"
        report = metrics_engine.collect_report(
            log_ids,
            cache_root,
            center_errors=centered,
            error_threshold=error_threshold,
            include_diagnostics=deep_dive,
        )
        reports[centering] = report
        metric_rows.extend(
            {"centering": centering, **row}
            for row in metrics_engine.tidy_metric_rows(report)
        )
        title = "CENTERED ERRORS" if centered else "UNCENTERED ERRORS"
        body = metrics_engine.render_report(report, center_errors=centered, sort_key=sort_key)
        report_parts.append(f"{'=' * 24} {title} {'=' * 24}\n\n{body}")

    return reports, metric_rows, "\n".join(report_parts)


def report_failures(reports: Mapping[str, metrics_engine.AggregatedReport]) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for centering, report in reports.items():
        failures.extend(
            {"centering": centering, "log": log_id, "error": str(exc)}
            for log_id, exc in report.failures
        )
    return failures


def save_tables(
    root: Path,
    reports: Mapping[str, metrics_engine.AggregatedReport],
) -> list[str]:
    written: list[str] = []
    for centering, report in reports.items():
        table_dir = root / "tables" / centering
        table_dir.mkdir(parents=True, exist_ok=True)
        for filename, rows, fieldnames in metrics_engine.wide_report_tables(report):
            path = table_dir / filename
            metrics_engine.write_csv(path, rows, fieldnames)
            written.append(str(path.relative_to(root)))
    return written


def build_report_header(manifest: Mapping[str, Any]) -> str:
    tags = ", ".join(manifest.get("tags", [])) or "none"
    notes = manifest.get("notes") or "(none)"
    selection = manifest["selection"]
    group = selection.get("set") or ", ".join(selection.get("explicit_logs", [])) or "registry query/all usable"
    versions = manifest["versions"]
    code_versions = ", ".join(value[:12] for value in versions.get("pipeline_code_sha256", []))
    return (
        f"Experiment: {manifest['name']}\n"
        f"ID: {manifest['id']}\n"
        f"Created: {manifest['created_at']}\n"
        f"Group: {group}\n"
        f"Tags: {tags}\n"
        f"Pipeline code: {code_versions}\n"
        f"Experiment fingerprint: {manifest['experiment_fingerprint']}\n"
        f"Notes: {notes}\n\n"
    )


def save_experiment(
    args: argparse.Namespace,
    *,
    selection: dict[str, Any],
    used: list[CacheInspection],
    excluded: list[CacheInspection],
    reports: Mapping[str, metrics_engine.AggregatedReport],
    metric_rows: list[dict[str, Any]],
    report_body: str,
) -> tuple[Path, dict[str, Any]]:
    root = args.store.resolve()
    root.mkdir(parents=True, exist_ok=True)
    experiment_id = make_experiment_id(args.name)
    destination = root / experiment_id
    suffix = 2
    while destination.exists():
        destination = root / f"{experiment_id}-{suffix}"
        suffix += 1
    experiment_id = destination.name

    log_records = [inspection.manifest_record() for inspection in used]
    excluded_records = [inspection.manifest_record() for inspection in excluded]
    stats_config = {
        "centering": list(CENTERING_MODES),
        "error_threshold": args.error_threshold,
        "deep_dive": bool(args.deep_dive),
        "sort_key": args.sort_key,
        "metrics_schema": 1,
    }
    versions = {
        "run_fingerprints": unique_values(log_records, "run_fingerprint"),
        "pipeline_code_sha256": unique_values(log_records, "pipeline_code_sha256"),
        "resolved_config_sha256": unique_values(log_records, "resolved_config_sha256"),
        "input_sha256": unique_values(log_records, "input_sha256"),
        "assets_sha256": unique_values(log_records, "assets_sha256"),
        "environment_sha256": unique_values(log_records, "environment_sha256"),
        "git_commits": sorted(
            {
                str(record.get("git", {}).get("commit"))
                for record in log_records
                if record.get("git", {}).get("commit")
            }
        ),
        "any_dirty": any(record.get("git", {}).get("dirty") for record in log_records),
    }
    fingerprint = experiment_fingerprint(
        name=args.name,
        selection=selection,
        stats_config=stats_config,
        logs=log_records,
    )
    failures = report_failures(reports)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "id": experiment_id,
        "name": args.name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "notes": args.notes or "",
        "tags": sorted(set(args.tags)),
        "experiment_fingerprint": fingerprint,
        "selection": selection,
        "stats": stats_config,
        "versions": versions,
        "results": {
            "requested_logs": len(selection["selected_logs"]),
            "included_logs": len(used),
            "excluded_logs": len(excluded),
            "metric_rows": len(metric_rows),
            "failures": failures,
        },
        "logs": log_records,
        "excluded": excluded_records,
    }

    temporary = Path(tempfile.mkdtemp(prefix=f".{experiment_id}-", dir=root))
    try:
        write_metrics(temporary / METRICS_FILENAME, metric_rows)
        write_log_records(temporary / LOGS_FILENAME, [*log_records, *excluded_records])
        table_files = save_tables(temporary, reports)
        files = [MANIFEST_FILENAME, METRICS_FILENAME, LOGS_FILENAME, REPORT_FILENAME, *table_files]
        manifest["files"] = files
        (temporary / REPORT_FILENAME).write_text(build_report_header(manifest) + report_body, encoding="utf-8")
        (temporary / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    except Exception:
        for path in sorted(temporary.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        temporary.rmdir()
        raise
    return destination, manifest


def command_run(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    logs, selection = select_logs(args, registry)
    if not logs:
        raise SystemExit("The registry query selected no logs.")

    inspections = inspect_logs(logs, args.cache_root)
    if args.process:
        stale = [inspection.log for inspection in inspections if not inspection.fresh]
        if stale:
            print(f"Processing {len(stale)} stale or missing log(s) with the current pipeline...")
        for log in stale:
            print(f"Running {log.pipeline} pipeline for {log.log_id}...", flush=True)
            process_log(log)
        inspections = inspect_logs(logs, args.cache_root)

    nonfresh = [inspection for inspection in inspections if not inspection.fresh]
    if nonfresh:
        print_nonfresh(inspections)
        if not args.allow_partial:
            raise SystemExit(
                "Refusing to create a partial/mixed experiment. Re-run with --process to refresh these "
                "caches, or explicitly use --allow-partial to record and exclude them."
            )

    used = [inspection for inspection in inspections if inspection.fresh]
    if not used:
        raise SystemExit("No fresh caches are available for this experiment.")

    reports, metric_rows, report_body = collect_both_modes(
        [inspection.log.log_id for inspection in used],
        args.cache_root,
        error_threshold=args.error_threshold,
        deep_dive=args.deep_dive,
        sort_key=args.sort_key,
    )
    failures = report_failures(reports)
    if failures and not args.allow_partial:
        for failure in failures:
            print(f"  {failure['centering']:<10} {failure['log']}: {failure['error']}")
        raise SystemExit("Refusing to save an incomplete stats experiment; use --allow-partial to override.")

    destination, manifest = save_experiment(
        args,
        selection=selection,
        used=used,
        excluded=nonfresh,
        reports=reports,
        metric_rows=metric_rows,
        report_body=report_body,
    )
    print(f"Saved {manifest['name']!r} as {manifest['id']}")
    print(f"  {destination}")
    print(f"  {len(used)} logs, both uncentered and centered stats, {len(metric_rows)} metrics")

    if args.baseline:
        print()
        return print_comparison(args.baseline, str(destination), args.store, "both", "error", "travel/solved", "rmse", args.compare_top)
    return 0


def group_label(manifest: Mapping[str, Any]) -> str:
    selection = manifest.get("selection", {})
    if selection.get("set"):
        return f"set:{selection['set']}"
    explicit = selection.get("explicit_logs", [])
    if explicit:
        return ",".join(explicit[:2]) + ("..." if len(explicit) > 2 else "")
    filters = selection.get("where", {})
    if filters:
        return ",".join(f"{key}={value}" for key, value in filters.items())
    return "all usable"


def command_list(args: argparse.Namespace) -> int:
    rows = []
    for path, manifest in iter_experiments(args.store):
        if args.tag and args.tag not in manifest.get("tags", []):
            continue
        if args.contains_log and args.contains_log not in {row.get("log") for row in manifest.get("logs", [])}:
            continue
        if args.code:
            versions = manifest.get("versions", {}).get("pipeline_code_sha256", [])
            if not any(str(value).startswith(args.code) for value in versions):
                continue
        rows.append((path, manifest))

    if not rows:
        print(f"No saved experiments under {args.store}")
        return 0
    print(f"{'created':<25} {'name':<28} {'logs':>4}  {'code':<12}  {'group':<24} tags")
    for _, manifest in rows:
        versions = manifest.get("versions", {}).get("pipeline_code_sha256", [])
        code = str(versions[0])[:10] if len(versions) == 1 else f"mixed:{len(versions)}"
        tags = ",".join(manifest.get("tags", []))
        print(
            f"{str(manifest.get('created_at', '')):<25} {str(manifest.get('name', ''))[:28]:<28} "
            f"{int(manifest.get('results', {}).get('included_logs', 0)):>4}  {code:<12}  "
            f"{group_label(manifest)[:24]:<24} {tags}"
        )
        if args.verbose and manifest.get("notes"):
            print(f"  {manifest['id']}: {manifest['notes']}")
    return 0


def format_stat(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.4f}"


def command_show(args: argparse.Namespace) -> int:
    path, manifest = resolve_experiment(args.reference, args.store)
    print(f"{manifest['name']} ({manifest['id']})")
    print(f"Created: {manifest['created_at']}")
    print(f"Path: {path}")
    print(f"Group: {group_label(manifest)}")
    print(f"Logs: {manifest['results']['included_logs']} included, {manifest['results']['excluded_logs']} excluded")
    print(f"Tags: {', '.join(manifest.get('tags', [])) or '(none)'}")
    print(f"Notes: {manifest.get('notes') or '(none)'}")
    versions = manifest.get("versions", {})
    print(f"Pipeline code: {', '.join(value[:12] for value in versions.get('pipeline_code_sha256', []))}")
    print(f"Git commit: {', '.join(value[:12] for value in versions.get('git_commits', [])) or '(unknown)'}")
    print(f"Run recorded dirty tree: {'yes' if versions.get('any_dirty') else 'no'}")
    print(f"Experiment fingerprint: {manifest['experiment_fingerprint']}")

    metrics = load_metrics(path)
    print("\ntravel/solved RMSE")
    for centering in CENTERING_MODES:
        summary = aggregate(primary_metric_values(metrics, centering=centering).values())
        print(
            f"  {centering:<10} n={summary['n']:<3} mean={format_stat(summary['mean'])}  "
            f"median={format_stat(summary['median'])}  std={format_stat(summary['std'])}"
        )
    failures = manifest.get("results", {}).get("failures", [])
    if failures:
        print(f"\nStats failures: {len(failures)} (see experiment.json)")
    return 0


def selected_centerings(value: str) -> tuple[str, ...]:
    return CENTERING_MODES if value == "both" else (value,)


def print_comparison(
    baseline_ref: str,
    current_ref: str,
    store: Path,
    centering: str,
    section: str,
    comparison: str,
    metric: str,
    top: int,
) -> int:
    baseline_path, baseline_manifest = resolve_experiment(baseline_ref, store)
    current_path, current_manifest = resolve_experiment(current_ref, store)
    baseline_metrics = load_metrics(baseline_path)
    current_metrics = load_metrics(current_path)

    baseline_code = baseline_manifest.get("versions", {}).get("pipeline_code_sha256", [])
    current_code = current_manifest.get("versions", {}).get("pipeline_code_sha256", [])

    print(f"Baseline: {baseline_manifest['name']} ({baseline_manifest['id']})")
    print(f"          code {', '.join(value[:12] for value in baseline_code) or '(unknown)'}")
    print(f"Current:  {current_manifest['name']} ({current_manifest['id']})")
    print(f"          code {', '.join(value[:12] for value in current_code) or '(unknown)'}")
    print(f"Metric: {section} / {comparison} / {metric}; delta = current - baseline")

    baseline_stats = baseline_manifest.get("stats", {})
    current_stats = current_manifest.get("stats", {})
    if baseline_stats.get("error_threshold") != current_stats.get("error_threshold"):
        print("WARNING: experiments used different error thresholds.")

    baseline_inputs = {
        str(record.get("log")): record.get("input_sha256")
        for record in baseline_manifest.get("logs", [])
    }
    current_inputs = {
        str(record.get("log")): record.get("input_sha256")
        for record in current_manifest.get("logs", [])
    }
    changed_inputs = {
        log_id
        for log_id in set(baseline_inputs) & set(current_inputs)
        if not baseline_inputs[log_id] or baseline_inputs[log_id] != current_inputs[log_id]
    }
    if changed_inputs:
        print(
            f"WARNING: excluding {len(changed_inputs)} overlapping log ID(s) whose input checksum changed: "
            + ", ".join(sorted(changed_inputs))
        )

    for mode in selected_centerings(centering):
        before = primary_metric_values(
            baseline_metrics,
            centering=mode,
            section=section,
            comparison=comparison,
            metric=metric,
        )
        after = primary_metric_values(
            current_metrics,
            centering=mode,
            section=section,
            comparison=comparison,
            metric=metric,
        )
        overlap = sorted((set(before) & set(after)) - changed_inputs)
        only_before = sorted(set(before) - set(after))
        only_after = sorted(set(after) - set(before))
        print(f"\n{mode}: {len(overlap)} overlapping logs ({len(only_before)} baseline-only, {len(only_after)} current-only)")
        if not overlap:
            continue

        rows = [
            {
                "log": log_id,
                "before": before[log_id],
                "after": after[log_id],
                "delta": after[log_id] - before[log_id],
            }
            for log_id in overlap
        ]
        finite_rows = [row for row in rows if np.isfinite(row["delta"])]
        before_summary = aggregate(row["before"] for row in finite_rows)
        after_summary = aggregate(row["after"] for row in finite_rows)
        delta_summary = aggregate(row["delta"] for row in finite_rows)
        print(
            f"  overlap mean:   {format_stat(before_summary['mean'])} -> {format_stat(after_summary['mean'])}  "
            f"delta {format_stat(delta_summary['mean'])}"
        )
        print(
            f"  overlap median: {format_stat(before_summary['median'])} -> {format_stat(after_summary['median'])}  "
            f"delta {format_stat(delta_summary['median'])}"
        )
        finite_rows.sort(key=lambda row: abs(row["delta"]), reverse=True)
        print(f"  {'log':<25} {'before':>10} {'after':>10} {'delta':>10}")
        for row in finite_rows[: max(1, top)]:
            print(
                f"  {row['log']:<25} {format_stat(row['before']):>10} "
                f"{format_stat(row['after']):>10} {format_stat(row['delta']):>10}"
            )
    return 0


def command_compare(args: argparse.Namespace) -> int:
    return print_comparison(
        args.baseline,
        args.current,
        args.store,
        args.centering,
        args.section,
        args.comparison,
        args.metric,
        args.top,
    )


def add_store_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store", type=Path, default=DEFAULT_EXPERIMENT_ROOT, help="Saved experiment root")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Create a versioned stats experiment")
    run_parser.add_argument("name", help="Memorable experiment name")
    run_parser.add_argument("logs", nargs="*", help="Explicit registry log IDs; defaults to the selected set/query or all usable logs")
    run_parser.add_argument("--set", dest="set_name", help="Select a named registry set")
    run_parser.add_argument("--where", action="append", default=[], metavar="KEY=VALUE", help="Filter resolved log metadata")
    run_parser.add_argument("--all-statuses", action="store_true", help="Include non-usable logs intentionally")
    run_parser.add_argument("--process", action="store_true", help="Run the current pipeline for stale or missing selected caches")
    run_parser.add_argument("--allow-partial", action="store_true", help="Explicitly exclude stale or failed logs and save a partial experiment")
    run_parser.add_argument("--notes", help="What changed, hypothesis, or result to remember")
    run_parser.add_argument("--tag", dest="tags", action="append", default=[], help="Catalog tag (repeatable)")
    run_parser.add_argument("--baseline", help="Compare the new experiment to a saved experiment after creation")
    run_parser.add_argument("--compare-top", type=int, default=20)
    run_parser.add_argument("--deep-dive", action="store_true", help="Include detailed stage/condition/bin diagnostics")
    run_parser.add_argument("--error-threshold", type=float)
    run_parser.add_argument("--sort-key", default="log")
    run_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    run_parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    add_store_argument(run_parser)
    run_parser.set_defaults(func=command_run)

    list_parser = subparsers.add_parser("list", help="Browse the experiment catalog")
    list_parser.add_argument("--tag")
    list_parser.add_argument("--contains-log")
    list_parser.add_argument("--code", help="Pipeline-code fingerprint prefix")
    list_parser.add_argument("--verbose", action="store_true")
    add_store_argument(list_parser)
    list_parser.set_defaults(func=command_list)

    show_parser = subparsers.add_parser("show", help="Show one saved experiment")
    show_parser.add_argument("reference", help="Experiment ID/prefix, name, or directory")
    add_store_argument(show_parser)
    show_parser.set_defaults(func=command_show)

    compare_parser = subparsers.add_parser("compare", help="Compare saved experiments on overlapping logs")
    compare_parser.add_argument("baseline")
    compare_parser.add_argument("current")
    compare_parser.add_argument("--centering", choices=["both", *CENTERING_MODES], default="both")
    compare_parser.add_argument("--section", default="error")
    compare_parser.add_argument("--comparison", default="travel/solved")
    compare_parser.add_argument("--metric", default="rmse")
    compare_parser.add_argument("--top", type=int, default=20)
    add_store_argument(compare_parser)
    compare_parser.set_defaults(func=command_compare)
    return parser


def main() -> int:
    ensure_project_python()
    args = build_parser().parse_args()
    try:
        return int(args.func(args))
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Pipeline failed with exit code {exc.returncode}; no experiment was saved.") from exc
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
