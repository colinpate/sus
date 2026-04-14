from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping


LOG_CONFIG_KEY = "__log_config__"
LogConfig = Dict[str, Any]
EMPTY_LOG_CONFIG_HASH = hashlib.sha256(b"{}").hexdigest()


def get_log_config_path(log_path: Path) -> Path:
    return log_path.with_suffix(".meta.json")


def load_log_config(log_path: Path) -> LogConfig:
    config_path = get_log_config_path(log_path)
    if not config_path.exists():
        print(f"No log config found at {config_path}, using empty config")
        return {}

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Expected top-level object in {config_path}")

    return config


def attach_log_config(ws: Dict[str, Any], config: Mapping[str, Any]) -> None:
    ws[LOG_CONFIG_KEY] = dict(config)


def get_log_config(ws: Mapping[str, Any]) -> LogConfig:
    config = ws.get(LOG_CONFIG_KEY, {})
    if isinstance(config, dict):
        return config
    return {}


def get_log_config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_signal_config(config: Mapping[str, Any], signal_name: str) -> Dict[str, Any]:
    signals = config.get("signals", {})
    if not isinstance(signals, dict):
        return {}

    signal_config = signals.get(signal_name, {})
    if isinstance(signal_config, dict):
        return signal_config
    return {}


def get_step_config(ws: Mapping[str, Any], step_name: str, step_type: str) -> Dict[str, Any]:
    config = get_log_config(ws)

    step_types = config.get("step_types", {})
    steps = config.get("steps", {})

    merged: Dict[str, Any] = {}
    if isinstance(step_types, dict):
        type_config = step_types.get(step_type, {})
        if isinstance(type_config, dict):
            merged.update(type_config)

    if isinstance(steps, dict):
        step_config = steps.get(step_name, {})
        if isinstance(step_config, dict):
            merged.update(step_config)

    return merged
