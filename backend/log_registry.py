from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
import tomllib
from typing import Any, Iterable, Mapping
import warnings


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "logs" / "registry.toml"
VALID_PIPELINES = {"front", "rear"}
VALID_STATUSES = {"pending-review", "usable", "partial", "corrupt", "excluded"}
VALID_RECORD_FORMATS = {"legacy", "imu_gyro", "dual_mag"}


class RegistryError(ValueError):
    pass


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


@dataclass(frozen=True)
class ResolvedLog:
    log_id: str
    registry_path: Path
    csv_path: Path | None
    source_path: Path | None
    pipeline: str
    status: str
    profiles: tuple[str, ...]
    sets: tuple[str, ...]
    tags: tuple[str, ...]
    metadata: dict[str, Any]
    processing_config: dict[str, Any]
    record_format: str | None
    legacy: bool = False

    def require_csv(self) -> Path:
        if self.csv_path is None:
            raise RegistryError(f"{self.log_id} has no converted CSV path")
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)
        return self.csv_path


class LogRegistry:
    def __init__(self, path: Path, data: Mapping[str, Any]) -> None:
        self.path = path.resolve()
        self.data: dict[str, Any] = deepcopy(dict(data))
        self.data.setdefault("schema_version", 1)
        self.data.setdefault("profiles", {})
        self.data.setdefault("presets", {})
        self.data.setdefault("logs", {})

    @classmethod
    def load(cls, path: Path = DEFAULT_REGISTRY_PATH) -> "LogRegistry":
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        return cls(path, data)

    @property
    def profiles(self) -> dict[str, dict[str, Any]]:
        return self.data["profiles"]

    @property
    def presets(self) -> dict[str, dict[str, Any]]:
        return self.data["presets"]

    @property
    def logs(self) -> dict[str, dict[str, Any]]:
        return self.data["logs"]

    def _data_path(self, value: object) -> Path | None:
        if value in (None, ""):
            return None
        path = Path(str(value))
        if path.is_absolute():
            return path
        return (self.path.parent / path).resolve()

    def resolve(self, log_id: str) -> ResolvedLog:
        if log_id not in self.logs:
            raise KeyError(f"Unknown log {log_id!r} in {self.path}")
        record = self.logs[log_id]
        profile_names = tuple(str(value) for value in record.get("profiles", []))

        metadata: dict[str, Any] = {}
        processing_config: dict[str, Any] = {}
        pipeline: str | None = None
        record_format: str | None = None
        for profile_name in profile_names:
            if profile_name not in self.profiles:
                raise RegistryError(f"{log_id}: unknown profile {profile_name!r}")
            profile = self.profiles[profile_name]
            metadata = deep_merge(metadata, profile.get("metadata", {}))
            processing_config = deep_merge(processing_config, profile.get("config", {}))
            if "pipeline" in profile:
                profile_pipeline = str(profile["pipeline"])
                if pipeline is not None and pipeline != profile_pipeline:
                    raise RegistryError(
                        f"{log_id}: profiles select both {pipeline!r} and {profile_pipeline!r} pipelines"
                    )
                pipeline = profile_pipeline
            if "record_format" in profile:
                profile_format = str(profile["record_format"])
                if record_format is not None and record_format != profile_format:
                    raise RegistryError(
                        f"{log_id}: profiles select both {record_format!r} and {profile_format!r} formats"
                    )
                record_format = profile_format

        metadata = deep_merge(metadata, record.get("metadata", {}))
        processing_config = deep_merge(processing_config, record.get("overrides", {}))
        pipeline = str(record.get("pipeline", pipeline or ""))
        record_format_value = record.get("record_format", record_format)
        record_format = str(record_format_value) if record_format_value else None

        return ResolvedLog(
            log_id=log_id,
            registry_path=self.path,
            csv_path=self._data_path(record.get("file")),
            source_path=self._data_path(record.get("source")),
            pipeline=pipeline,
            status=str(record.get("status", "pending-review")),
            profiles=profile_names,
            sets=tuple(str(value) for value in record.get("sets", [])),
            tags=tuple(str(value) for value in record.get("tags", [])),
            metadata=metadata,
            processing_config=processing_config,
            record_format=record_format,
        )

    def validate(self, *, check_files: bool = True) -> list[str]:
        errors: list[str] = []
        if self.data.get("schema_version") != 1:
            errors.append(f"unsupported schema_version={self.data.get('schema_version')!r}")
        if not isinstance(self.profiles, dict) or not isinstance(self.logs, dict):
            return errors + ["profiles and logs must be TOML tables"]

        seen_files: dict[Path, str] = {}
        seen_ids: dict[str, str] = {}
        for log_id in self.logs:
            try:
                log = self.resolve(log_id)
            except Exception as exc:
                errors.append(str(exc))
                continue
            if not log_id or log_id != log_id.strip():
                errors.append(f"invalid log id {log_id!r}")
            previous_id = seen_ids.get(log_id.casefold())
            if previous_id is not None:
                errors.append(f"log IDs differ only by case: {previous_id!r} and {log_id!r}")
            seen_ids[log_id.casefold()] = log_id
            if log.pipeline not in VALID_PIPELINES:
                errors.append(f"{log_id}: invalid or missing pipeline {log.pipeline!r}")
            if log.status not in VALID_STATUSES:
                errors.append(f"{log_id}: invalid status {log.status!r}")
            if log.record_format is not None and log.record_format not in VALID_RECORD_FORMATS:
                errors.append(f"{log_id}: invalid record format {log.record_format!r}")
            if log.csv_path is not None:
                previous = seen_files.get(log.csv_path)
                if previous is not None:
                    errors.append(f"{log_id}: CSV is also used by {previous}: {log.csv_path}")
                seen_files[log.csv_path] = log_id
                if check_files and not log.csv_path.exists():
                    errors.append(f"{log_id}: missing CSV {log.csv_path}")
            elif log.status == "usable":
                errors.append(f"{log_id}: usable log has no CSV")
            if check_files and log.source_path is not None and not log.source_path.exists():
                errors.append(f"{log_id}: missing source {log.source_path}")
            if check_files:
                for asset in config_asset_paths(log.processing_config):
                    path = asset if asset.is_absolute() else REPO_ROOT / asset
                    if not path.is_file():
                        errors.append(f"{log_id}: missing processing asset {path}")

        for preset_name, preset in self.presets.items():
            for profile_name in preset.get("profiles", []):
                if profile_name not in self.profiles:
                    errors.append(f"preset {preset_name}: unknown profile {profile_name!r}")
        return errors

    def select(
        self,
        *,
        log_ids: Iterable[str] = (),
        set_name: str | None = None,
        filters: Mapping[str, str] | None = None,
        usable_only: bool = True,
    ) -> list[ResolvedLog]:
        requested = list(log_ids)
        candidates = requested if requested else list(self.logs)
        selected: list[ResolvedLog] = []
        for log_id in candidates:
            log = self.resolve(log_id)
            if usable_only and log.status != "usable":
                continue
            if set_name is not None and set_name not in log.sets:
                continue
            values = {
                "id": log.log_id,
                "pipeline": log.pipeline,
                "status": log.status,
                **{key: str(value) for key, value in log.metadata.items()},
            }
            if filters and any(values.get(key) != value for key, value in filters.items()):
                continue
            selected.append(log)
        return selected

    def set_record(self, log_id: str, record: Mapping[str, Any]) -> None:
        self.logs[log_id] = deepcopy(dict(record))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = dump_toml(self.data)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.path)


def load_registry(path: Path = DEFAULT_REGISTRY_PATH) -> LogRegistry:
    return LogRegistry.load(path)


def resolve_log(
    log_id: str,
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    allow_legacy: bool = True,
) -> ResolvedLog:
    try:
        return load_registry(registry_path).resolve(log_id)
    except (FileNotFoundError, KeyError):
        if not allow_legacy:
            raise

    csv_path = REPO_ROOT / "logs" / f"{log_id}.csv"
    if not csv_path.exists():
        raise KeyError(f"Unknown log {log_id!r}; no registry entry or legacy CSV exists")
    sidecar = csv_path.with_suffix(".meta.json")
    config: dict[str, Any] = {}
    if sidecar.exists():
        with sidecar.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict):
            config = value
    warnings.warn(
        f"{log_id} is not in {registry_path}; using legacy filename/sidecar discovery",
        DeprecationWarning,
        stacklevel=2,
    )


def config_asset_paths(config: Mapping[str, Any]) -> list[Path]:
    paths: list[Path] = []

    def visit(value: object, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and (key.endswith("_path") or key.endswith("_file")):
            paths.append(Path(value))

    visit(config)
    return paths
    return ResolvedLog(
        log_id=log_id,
        registry_path=registry_path,
        csv_path=csv_path,
        source_path=(REPO_ROOT / "logs" / f"{log_id}.bin"),
        pipeline="rear" if "rear" in log_id else "front",
        status="usable",
        profiles=(),
        sets=(),
        tags=(),
        metadata={},
        processing_config=config,
        record_format=None,
        legacy=True,
    )


_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _toml_key(value: str) -> str:
    return value if _BARE_KEY.fullmatch(value) else json.dumps(value, ensure_ascii=False)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        if value != value:
            return "nan"
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Cannot encode {type(value).__name__} as TOML")


def dump_toml(data: Mapping[str, Any]) -> str:
    lines: list[str] = []

    def emit_table(path: tuple[str, ...], table: Mapping[str, Any]) -> None:
        scalar_items = [(key, value) for key, value in table.items() if not isinstance(value, Mapping)]
        child_items = [(key, value) for key, value in table.items() if isinstance(value, Mapping)]
        if path and scalar_items:
            if lines and lines[-1] != "":
                lines.append("")
            lines.append("[" + ".".join(_toml_key(part) for part in path) + "]")
        for key, value in scalar_items:
            lines.append(f"{_toml_key(str(key))} = {_toml_value(value)}")
        for key, value in child_items:
            emit_table((*path, str(key)), value)

    emit_table((), data)
    return "\n".join(lines).rstrip() + "\n"
