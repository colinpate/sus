from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

try:
    from log_registry import REPO_ROOT, ResolvedLog
except ModuleNotFoundError:  # Imported as backend.run_provenance from repository-root tools/tests.
    from backend.log_registry import REPO_ROOT, ResolvedLog


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_tree_manifest(root: Path) -> list[dict[str, str]]:
    files: list[dict[str, str]] = []
    for path in sorted((root / "backend").rglob("*.py")):
        if "__pycache__" in path.parts or "run_artifacts" in path.parts or "tests" in path.parts:
            continue
        files.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path)})
    return files


def referenced_assets(config: Mapping[str, Any], root: Path) -> list[dict[str, str]]:
    found: set[Path] = set()

    def visit(value: object, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and (key.endswith("_path") or key.endswith("_file")):
            path = Path(value)
            if not path.is_absolute():
                path = root / path
            if path.is_file():
                found.add(path.resolve())

    visit(config)
    return [
        {"path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path), "sha256": sha256_file(path)}
        for path in sorted(found)
    ]


def runtime_versions() -> dict[str, str]:
    result = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
    }
    for package in ("numpy", "scipy", "pandas"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "missing"
    return result


@dataclass(frozen=True)
class RunProvenance:
    log_id: str
    pipeline: str
    input_path: str
    input_sha256: str
    resolved_config_sha256: str
    pipeline_code_sha256: str
    pipeline_source: tuple[dict[str, str], ...]
    assets: tuple[dict[str, str], ...]
    assets_sha256: str
    environment: dict[str, str]
    environment_sha256: str
    run_fingerprint: str
    profiles: tuple[str, ...]
    descriptive_metadata: dict[str, Any]
    git: dict[str, Any]

    def to_dict(self, *, status: str = "success") -> dict[str, Any]:
        return {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": status,
            "log": self.log_id,
            "pipeline": self.pipeline,
            "input": {"path": self.input_path, "sha256": self.input_sha256},
            "profiles": list(self.profiles),
            "descriptive_metadata": self.descriptive_metadata,
            "resolved_config_sha256": self.resolved_config_sha256,
            "pipeline_code_sha256": self.pipeline_code_sha256,
            "pipeline_source": list(self.pipeline_source),
            "assets": list(self.assets),
            "assets_sha256": self.assets_sha256,
            "environment": self.environment,
            "environment_sha256": self.environment_sha256,
            "run_fingerprint": self.run_fingerprint,
            "git": self.git,
        }

    def write(self, output_directory: Path) -> Path:
        output_directory.mkdir(parents=True, exist_ok=True)
        path = output_directory / "run.json"
        temporary = output_directory / ".run.json.tmp"
        temporary.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return path


def build_run_provenance(
    log: ResolvedLog,
    *,
    pipeline: str,
    root: Path = REPO_ROOT,
) -> RunProvenance:
    input_path = log.require_csv().resolve()
    config_hash = sha256_json(log.processing_config)
    source_manifest = source_tree_manifest(root)
    code_hash = sha256_json(source_manifest)
    assets = tuple(referenced_assets(log.processing_config, root))
    assets_hash = sha256_json(assets)
    environment = runtime_versions()
    environment_hash = sha256_json(environment)
    input_hash = sha256_file(input_path)
    git = git_snapshot(root)
    fingerprint = sha256_json(
        {
            "schema_version": 1,
            "pipeline": pipeline,
            "input_sha256": input_hash,
            "resolved_config_sha256": config_hash,
            "pipeline_code_sha256": code_hash,
            "assets_sha256": assets_hash,
            "environment_sha256": environment_hash,
        }
    )
    return RunProvenance(
        log_id=log.log_id,
        pipeline=pipeline,
        input_path=str(input_path.relative_to(root)) if input_path.is_relative_to(root) else str(input_path),
        input_sha256=input_hash,
        resolved_config_sha256=config_hash,
        pipeline_code_sha256=code_hash,
        pipeline_source=tuple(source_manifest),
        assets=assets,
        assets_sha256=assets_hash,
        environment=environment,
        environment_sha256=environment_hash,
        run_fingerprint=fingerprint,
        profiles=log.profiles,
        descriptive_metadata=log.metadata,
        git=git,
    )


def git_snapshot(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--short", "--", "backend")
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }
