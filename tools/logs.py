#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.log_registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    LogRegistry,
    RegistryError,
    deep_merge,
    load_registry,
)
from read_binary import FORMATS, ConversionResult, convert, detect_formats, sha256_file  # noqa: E402


TR11_LOGS = {"log-0078", "log-0078-valid", "log-0079", "log-0080", "log-0080-valid", "log-0081"}
LEGACY_FRONT_DEFAULTS = {
    "log022", "log029", "log030", "log031", "log038", "log056_ccdh", "log060_upperpred",
    "log078", "log079", "log080", "log085", "log088", "log091", "log096", "log098",
    "log099", "log103", "log104", "log106", "log107", "log109", "log110", "log112",
}


def parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_key_values(values: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise RegistryError(f"Expected KEY=VALUE, got {value!r}")
        key, raw = value.split("=", 1)
        if not key:
            raise RegistryError(f"Expected a non-empty key in {value!r}")
        result[key] = parse_value(raw)
    return result


def relative_to_registry(path: Path, registry: LogRegistry) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(registry.path.parent))
    except ValueError:
        return str(resolved)


def preset_record(registry: LogRegistry, preset_name: str) -> dict[str, Any]:
    if preset_name not in registry.presets:
        choices = ", ".join(sorted(registry.presets))
        raise RegistryError(f"Unknown preset {preset_name!r}; choose one of: {choices}")
    preset = deepcopy(registry.presets[preset_name])
    return {
        key: value
        for key, value in preset.items()
        if key in {"profiles", "pipeline", "record_format", "status", "sets", "tags", "metadata", "overrides"}
    }


def resolved_profile_defaults(registry: LogRegistry, profiles: Iterable[str]) -> tuple[str | None, str | None]:
    pipeline: str | None = None
    record_format: str | None = None
    for name in profiles:
        profile = registry.profiles.get(name)
        if profile is None:
            raise RegistryError(f"Unknown profile {name!r}")
        if profile.get("pipeline"):
            value = str(profile["pipeline"])
            if pipeline is not None and pipeline != value:
                raise RegistryError(f"Profiles select conflicting pipelines: {pipeline} and {value}")
            pipeline = value
        if profile.get("record_format"):
            value = str(profile["record_format"])
            if record_format is not None and record_format != value:
                raise RegistryError(f"Profiles select conflicting record formats: {record_format} and {value}")
            record_format = value
    return pipeline, record_format


def copy_source(source: Path, destination: Path) -> Path:
    source = source.resolve()
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source == destination:
        return destination
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise FileExistsError(f"Refusing to replace different source file {destination}")
        return destination
    temporary = destination.with_suffix(destination.suffix + ".partial")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    return destination


def ingest_one(
    registry: LogRegistry,
    source: Path,
    *,
    log_id: str,
    base_record: Mapping[str, Any],
    metadata: Mapping[str, Any],
    status: str | None,
    sets: Iterable[str],
    tags: Iterable[str],
    record_format: str | None,
) -> ConversionResult:
    if log_id in registry.logs:
        raise RegistryError(f"{log_id} already exists in the registry")
    if not source.is_file():
        raise FileNotFoundError(source)

    record = deepcopy(dict(base_record))
    profiles = [str(value) for value in record.get("profiles", [])]
    profile_pipeline, profile_format = resolved_profile_defaults(registry, profiles)
    effective_format = record_format or record.get("record_format") or profile_format
    if effective_format is None:
        matches = detect_formats(source)
        if len(matches) != 1:
            raise RegistryError(
                f"{source}: record format is ambiguous ({', '.join(matches) or 'no matches'}); "
                "use a preset or --record-format"
            )
        effective_format = matches[0]
    if effective_format not in FORMATS:
        raise RegistryError(f"Unknown record format {effective_format!r}")

    raw_destination = registry.path.parent / "raw" / f"{log_id}.bin"
    csv_destination = registry.path.parent / "converted" / f"{log_id}.csv"
    raw_destination = copy_source(source, raw_destination)
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=csv_destination.parent, prefix=f".{log_id}-", suffix=".csv.partial", delete=False
    ) as handle:
        temporary_csv = Path(handle.name)
    try:
        result = convert(raw_destination, temporary_csv, fmt=str(effective_format))
        if result.records == 0:
            raise RegistryError(f"{source}: conversion produced no records")
        os.replace(temporary_csv, csv_destination)
    except Exception:
        temporary_csv.unlink(missing_ok=True)
        raise

    record["file"] = relative_to_registry(csv_destination, registry)
    record["source"] = relative_to_registry(raw_destination, registry)
    record["pipeline"] = str(record.get("pipeline") or profile_pipeline or "")
    record["record_format"] = str(effective_format)
    record["status"] = status or str(record.get("status", "pending-review"))
    record["sets"] = sorted(set(record.get("sets", [])) | set(sets))
    record["tags"] = sorted(set(record.get("tags", [])) | set(tags))
    record["metadata"] = deep_merge(record.get("metadata", {}), metadata)
    record["import"] = {
        "records": result.records,
        "duration_s": result.duration_s,
        "sequence_gaps": result.sequence_gaps,
        "source_sha256": result.source_sha256,
        "output_sha256": result.output_sha256,
    }
    registry.set_record(log_id, record)
    registry.save()
    return result


def command_ingest(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    if args.copy_from and args.preset:
        raise RegistryError("Use either --copy-from or --preset, not both")
    if args.copy_from:
        if args.copy_from not in registry.logs:
            raise RegistryError(f"Unknown source log {args.copy_from!r}")
        source_record = registry.logs[args.copy_from]
        base_record = {
            key: deepcopy(source_record[key])
            for key in ("profiles", "pipeline", "record_format", "sets", "tags", "metadata", "overrides")
            if key in source_record
        }
        copied_metadata = base_record.get("metadata", {})
        for capture_specific_key in ("trail", "notes", "conditions"):
            copied_metadata.pop(capture_specific_key, None)
        base_record["status"] = "pending-review"
    elif args.preset:
        base_record = preset_record(registry, args.preset)
    else:
        raise RegistryError("Ingestion requires --preset or --copy-from")

    metadata = parse_key_values(args.metadata)
    for key in ("trail", "frame_model", "fork_model", "shock_model", "notes"):
        value = getattr(args, key)
        if value is not None:
            metadata[key] = value

    if args.id and len(args.inputs) != 1:
        raise RegistryError("--id can only be used with one input file")

    imported: list[str] = []
    for source in args.inputs:
        log_id = args.id or source.stem.removesuffix(".flash")
        result = ingest_one(
            registry,
            source,
            log_id=log_id,
            base_record=base_record,
            metadata=metadata,
            status=args.status,
            sets=args.sets,
            tags=args.tags,
            record_format=args.record_format,
        )
        imported.append(log_id)
        print(
            f"Imported {log_id}: {result.records} records, {result.duration_s:.1f}s, "
            f"sequence gaps={result.sequence_gaps}, status={registry.logs[log_id]['status']}"
        )

    if args.process:
        return process_ids(registry, imported, include_nonusable=True)
    return 0


def update_record_metadata(record: dict[str, Any], updates: Mapping[str, Any]) -> None:
    metadata = record.setdefault("metadata", {})
    metadata.update(updates)


def command_annotate(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    updates = parse_key_values(args.metadata)
    for key in ("trail", "frame_model", "fork_model", "shock_model", "notes"):
        value = getattr(args, key)
        if value is not None:
            updates[key] = value
    for log_id in args.logs:
        if log_id not in registry.logs:
            raise RegistryError(f"Unknown log {log_id!r}")
        record = registry.logs[log_id]
        update_record_metadata(record, updates)
        if args.status is not None:
            record["status"] = args.status
        if args.reason is not None:
            record["reason"] = args.reason
        record["tags"] = sorted((set(record.get("tags", [])) | set(args.tags)) - set(args.remove_tags))
        record["sets"] = sorted((set(record.get("sets", [])) | set(args.sets)) - set(args.remove_sets))
    registry.save()
    print(f"Updated {len(args.logs)} log(s)")
    return 0


def command_mark(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    for log_id in args.logs:
        if log_id not in registry.logs:
            raise RegistryError(f"Unknown log {log_id!r}")
        registry.logs[log_id]["status"] = args.status
        if args.reason:
            registry.logs[log_id]["reason"] = args.reason
    registry.save()
    print(f"Marked {len(args.logs)} log(s) as {args.status}")
    return 0


def filter_dict(values: Iterable[str]) -> dict[str, str]:
    return {key: str(value) for key, value in parse_key_values(values).items()}


def command_list(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    logs = registry.select(
        log_ids=args.logs,
        set_name=args.set_name,
        filters=filter_dict(args.where),
        usable_only=not args.all_statuses,
    )
    print(f"{'log':<24} {'status':<15} {'pipeline':<8} {'pod':<5} {'bike':<28} trail")
    for log in logs:
        print(
            f"{log.log_id:<24} {log.status:<15} {log.pipeline:<8} "
            f"{str(log.metadata.get('pod_version', '')):<5} "
            f"{str(log.metadata.get('bike_model', '')):<28} {log.metadata.get('trail', '')}"
        )
    print(f"{len(logs)} log(s)")
    return 0


def command_show(args: argparse.Namespace) -> int:
    log = load_registry(args.registry).resolve(args.log)
    print(json.dumps({
        "log": log.log_id,
        "file": str(log.csv_path) if log.csv_path else None,
        "source": str(log.source_path) if log.source_path else None,
        "pipeline": log.pipeline,
        "status": log.status,
        "profiles": log.profiles,
        "sets": log.sets,
        "tags": log.tags,
        "metadata": log.metadata,
        "processing_config": log.processing_config,
        "record_format": log.record_format,
    }, indent=2))
    return 0


def command_validate(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    errors = registry.validate(check_files=not args.no_files)
    if errors:
        print(f"Registry validation failed with {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"Registry is valid: {len(registry.logs)} logs, {len(registry.profiles)} profiles")
    return 0


def pipeline_python() -> str:
    candidate = REPO_ROOT / "venv" / "bin" / "python3"
    return str(candidate) if candidate.exists() else sys.executable


def process_ids(registry: LogRegistry, log_ids: Iterable[str], *, include_nonusable: bool) -> int:
    for log_id in log_ids:
        log = registry.resolve(log_id)
        if log.status != "usable" and not include_nonusable:
            print(f"Skipping {log_id}: status={log.status}")
            continue
        script = REPO_ROOT / "backend" / ("pipeline_rear.py" if log.pipeline == "rear" else "pipeline.py")
        print(f"Processing {log_id} with {log.pipeline} pipeline")
        subprocess.run([pipeline_python(), str(script), log_id], cwd=REPO_ROOT, check=True)
    return 0


def command_process(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    logs = registry.select(
        log_ids=args.logs,
        set_name=args.set_name,
        filters=filter_dict(args.where),
        usable_only=not args.include_nonusable,
    )
    if not logs:
        raise RegistryError("No logs matched")
    return process_ids(registry, [log.log_id for log in logs], include_nonusable=args.include_nonusable)


def clean_empty_mappings(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_empty_mappings(child) for key, child in value.items() if child != {}}
    if isinstance(value, list):
        return [clean_empty_mappings(child) for child in value]
    return value


def config_for_profiles(registry: LogRegistry, profiles: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in profiles:
        result = deep_merge(result, registry.profiles[name].get("config", {}))
    return result


def profiles_for_legacy(
    log_id: str,
    *,
    position: str,
    pod_version: int | None,
    sidecar: Mapping[str, Any],
) -> list[str]:
    profiles: list[str] = []
    if position == "rear":
        profiles.extend([f"rear-pod-v{pod_version if pod_version is not None else 0}", "bike-stumpjumper", "rear-stumpjumper-150"])
        linkage = sidecar.get("steps", {}).get("linkage_angle_to_travel", {}).get("linkage_path")
        if linkage == "foo.json":
            profiles.append("rear-linkage-original")
        elif linkage == "foo_ss.json":
            profiles.append("rear-linkage-current")
        z_rotation = sidecar.get("steps", {}).get("diff_mag", {}).get("z_rotation_deg")
        if z_rotation == -90:
            profiles.append("rear-mag-z-minus90")
        elif z_rotation == 0:
            profiles.append("rear-mag-z-zero")
        if sidecar.get("signals", {}).get("mag", {}).get("orientation_preset") == "identity":
            profiles.append("dual-mag-opposed")
        return profiles

    if log_id in TR11_LOGS:
        profiles.extend(["front-pod-v2", "bike-tr11", "fork-boxxer-200"])
    else:
        if pod_version in (1, 2):
            profiles.append(f"front-pod-v{pod_version}")
        profiles.extend(["bike-stumpjumper", "fork-fox36-160"])
    angle = sidecar.get("steps", {}).get("angle_to_travel", {})
    signature = (angle.get("hypotenuse"), angle.get("top_adjacent"), angle.get("angle_sign"))
    if signature == (125.0, 121.0, None):
        profiles.append("front-angle-stumpjumper-v1")
    elif signature == (125.0, 113.5, -1.0):
        profiles.append("front-angle-stumpjumper-v2")
    elif signature == (150.0, 136.0, -1.0):
        profiles.append("front-angle-tr11-boxxer")
    return profiles


def read_catalog(path: Path, available_names: Iterable[str]) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    case_map = {name.casefold(): name for name in available_names}
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            raw_id = (row.get("log") or "").strip()
            if not raw_id:
                continue
            log_id = case_map.get(raw_id.casefold(), raw_id.lower() if raw_id.startswith("Log-") else raw_id)
            rows[log_id] = {str(key): (value or "").strip() for key, value in row.items() if key}
    return rows


def status_from_catalog(value: str) -> tuple[str, str | None]:
    normalized = value.strip().casefold()
    if normalized == "yes":
        return "usable", None
    if normalized == "partial":
        return "partial", "Marked partial in the legacy log list"
    if normalized in {"no", "false"}:
        return "corrupt", "Marked invalid in the legacy log list"
    return "pending-review", None


def command_migrate_legacy(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    logs_dir = registry.path.parent
    csv_paths = {
        path.stem: path
        for path in logs_dir.glob("log*.csv")
        if not path.stem.endswith("_filtered") and "_sst" not in path.stem
    }
    bin_paths = {path.stem: path for path in logs_dir.glob("log*.bin") if not path.name.endswith(".flash.bin")}
    sidecars = {path.name.removesuffix(".meta.json"): path for path in logs_dir.glob("log*.meta.json")}
    available = set(csv_paths) | set(bin_paths) | set(sidecars)
    catalog = read_catalog(args.catalog, available)
    candidates = sorted(set(csv_paths) | set(sidecars) | set(catalog))

    added = 0
    for log_id in candidates:
        if log_id in registry.logs and not args.replace:
            continue
        sidecar: dict[str, Any] = {}
        if log_id in sidecars:
            value = json.loads(sidecars[log_id].read_text(encoding="utf-8"))
            if isinstance(value, dict):
                sidecar = value
        row = catalog.get(log_id, {})
        position = row.get("wheel") or ("rear" if "rear" in log_id else "front")
        pod_text = row.get("pod", "")
        pod_version = int(pod_text) if pod_text.isdigit() else (0 if position == "rear" else None)
        profiles = profiles_for_legacy(
            log_id,
            position=position,
            pod_version=pod_version,
            sidecar=sidecar,
        )
        status, reason = status_from_catalog(row.get("Valid", ""))
        if not row and (sidecar or log_id in LEGACY_FRONT_DEFAULTS):
            status = "usable"

        record: dict[str, Any] = {
            "pipeline": position,
            "status": status,
            "profiles": profiles,
            "sets": [],
            "tags": [],
        }
        if log_id in csv_paths:
            record["file"] = csv_paths[log_id].name
        source = bin_paths.get(log_id)
        if source is None and log_id.endswith("-valid"):
            source = bin_paths.get(log_id.removesuffix("-valid"))
        if source is not None:
            record["source"] = source.name
        if pod_version is not None or profiles:
            record["record_format"] = "dual_mag"
        if reason:
            record["reason"] = reason
        metadata: dict[str, Any] = {}
        if row.get("trail"):
            metadata["trail"] = row["trail"]
        if row.get("Setup/bike"):
            metadata["legacy_setup_note"] = row["Setup/bike"]
        if row.get("Travel"):
            metadata["legacy_travel_mm"] = parse_value(row["Travel"])
        if metadata:
            record["metadata"] = metadata
        if row:
            record["sets"].append("catalog")
            if status == "usable":
                record["sets"].append(f"{position}-default")
        elif log_id in LEGACY_FRONT_DEFAULTS:
            record["sets"].append("front-default")

        known_config = clean_empty_mappings(config_for_profiles(registry, profiles))
        old_config = clean_empty_mappings(sidecar)
        if old_config and old_config != known_config:
            record["overrides"] = sidecar
        registry.set_record(log_id, record)
        added += 1

    registry.save()
    print(f"Migrated {added} legacy log entries into {registry.path} ({len(registry.logs)} total)")
    return 0


def add_common_annotation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trail")
    parser.add_argument("--frame-model")
    parser.add_argument("--fork-model")
    parser.add_argument("--shock-model")
    parser.add_argument("--notes")
    parser.add_argument("--metadata", action="append", default=[], metavar="KEY=VALUE")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import, describe, select, and process suspension logs")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Import and convert one or more binary logs")
    ingest_parser.add_argument("inputs", nargs="+", type=Path)
    ingest_parser.add_argument("--id", help="Registry ID; defaults to input filename")
    ingest_parser.add_argument("--preset")
    ingest_parser.add_argument("--copy-from")
    ingest_parser.add_argument("--record-format", choices=sorted(FORMATS))
    ingest_parser.add_argument("--status", choices=["pending-review", "usable", "partial", "corrupt", "excluded"])
    ingest_parser.add_argument("--set", dest="sets", action="append", default=[])
    ingest_parser.add_argument("--tag", dest="tags", action="append", default=[])
    ingest_parser.add_argument("--process", action="store_true")
    add_common_annotation_args(ingest_parser)
    ingest_parser.set_defaults(func=command_ingest)

    annotate_parser = subparsers.add_parser("annotate", help="Update notes, tags, sets, or quality status")
    annotate_parser.add_argument("logs", nargs="+")
    annotate_parser.add_argument("--status", choices=["pending-review", "usable", "partial", "corrupt", "excluded"])
    annotate_parser.add_argument("--reason")
    annotate_parser.add_argument("--tag", dest="tags", action="append", default=[])
    annotate_parser.add_argument("--remove-tag", dest="remove_tags", action="append", default=[])
    annotate_parser.add_argument("--set", dest="sets", action="append", default=[])
    annotate_parser.add_argument("--remove-set", dest="remove_sets", action="append", default=[])
    add_common_annotation_args(annotate_parser)
    annotate_parser.set_defaults(func=command_annotate)

    mark_parser = subparsers.add_parser("mark", help="Set log quality status")
    mark_parser.add_argument("logs", nargs="+")
    mark_parser.add_argument("status", choices=["pending-review", "usable", "partial", "corrupt", "excluded"])
    mark_parser.add_argument("--reason")
    mark_parser.set_defaults(func=command_mark)

    list_parser = subparsers.add_parser("list", help="List registry logs")
    list_parser.add_argument("logs", nargs="*")
    list_parser.add_argument("--set", dest="set_name")
    list_parser.add_argument("--where", action="append", default=[], metavar="KEY=VALUE")
    list_parser.add_argument("--all-statuses", action="store_true")
    list_parser.set_defaults(func=command_list)

    show_parser = subparsers.add_parser("show", help="Show resolved metadata and processing config")
    show_parser.add_argument("log")
    show_parser.set_defaults(func=command_show)

    validate_parser = subparsers.add_parser("validate", help="Validate registry structure and file references")
    validate_parser.add_argument("--no-files", action="store_true")
    validate_parser.set_defaults(func=command_validate)

    process_parser = subparsers.add_parser("process", help="Run the configured pipeline for selected logs")
    process_parser.add_argument("logs", nargs="*")
    process_parser.add_argument("--set", dest="set_name")
    process_parser.add_argument("--where", action="append", default=[], metavar="KEY=VALUE")
    process_parser.add_argument("--include-nonusable", action="store_true")
    process_parser.set_defaults(func=command_process)

    migrate_parser = subparsers.add_parser("migrate-legacy", help="Import the old list and sidecar metadata")
    migrate_parser.add_argument("--catalog", type=Path, default=REPO_ROOT / "logs" / "lists" / "logs.csv")
    migrate_parser.add_argument("--replace", action="store_true")
    migrate_parser.set_defaults(func=command_migrate_legacy)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.func(args))
    except (RegistryError, FileNotFoundError, KeyError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
