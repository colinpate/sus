from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mag_to_travel_model_core import MagToTravelModel


PipelineKind = Literal["front", "rear"]
TimeBasis = Literal["elapsed", "active"]
CalibrationMethod = Literal[
    "self_supervised_power",
    "oracle_power",
    "oracle_isotonic",
    "oracle_binned_median",
]


@dataclass(frozen=True)
class TimeRange:
    """A half-open range expressed in elapsed or accumulated active seconds."""

    start_s: float | None = None
    stop_s: float | None = None

    def __post_init__(self):
        if self.start_s is not None and self.start_s < 0:
            raise ValueError("start_s must be non-negative")
        if self.stop_s is not None and self.stop_s < 0:
            raise ValueError("stop_s must be non-negative")
        if self.start_s is not None and self.stop_s is not None and self.stop_s <= self.start_s:
            raise ValueError("stop_s must be greater than start_s")


@dataclass(frozen=True)
class RecordingWindow:
    log_name: str
    time_range: TimeRange = field(default_factory=TimeRange)
    time_basis: TimeBasis = "elapsed"


@dataclass(frozen=True)
class ResolvedWindow:
    """A requested time range resolved to a half-open sample range."""

    sample_start: int
    sample_stop: int
    wall_start_s: float
    wall_stop_s: float
    active_duration_s: float

    @property
    def sample_range(self) -> tuple[int, int]:
        return self.sample_start, self.sample_stop

    def sample_mask(self, sample_count: int) -> np.ndarray:
        mask = np.zeros(sample_count, dtype=bool)
        mask[self.sample_start:self.sample_stop] = True
        return mask


def sample_durations(time_s: np.ndarray) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    if len(time_s) == 0:
        return np.empty(0, dtype=float)
    if len(time_s) == 1:
        return np.ones(1, dtype=float)
    differences = np.diff(time_s)
    positive = differences[np.isfinite(differences) & (differences > 0)]
    if len(positive) == 0:
        raise ValueError("Time values must contain at least one positive interval")
    fallback = float(np.median(positive))
    return np.r_[np.where(differences > 0, differences, fallback), fallback]


def resolve_window(
    time_s: np.ndarray,
    requested: TimeRange,
    *,
    time_basis: TimeBasis = "elapsed",
    activity_mask: np.ndarray | None = None,
) -> ResolvedWindow:
    """Resolve elapsed or accumulated-active seconds to sample boundaries.

    Active ranges follow the wall-clock span containing the requested amount
    of active data.  Inactive gaps inside that span remain available to signal
    preprocessing, while scoring can still intersect the resolved range with
    the activity mask.
    """
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    if len(time_s) == 0:
        raise ValueError("Cannot resolve a window on an empty time series")
    if np.any(~np.isfinite(time_s)) or np.any(np.diff(time_s) < 0):
        raise ValueError("Time values must be finite and nondecreasing")

    durations = sample_durations(time_s)
    if activity_mask is None:
        activity = np.ones(len(time_s), dtype=bool)
    else:
        activity = np.asarray(activity_mask, dtype=bool).reshape(-1)
        if activity.shape != time_s.shape:
            raise ValueError(f"Activity mask shape {activity.shape} does not match time shape {time_s.shape}")

    if time_basis == "elapsed":
        origin = float(time_s[0])
        start_value = 0.0 if requested.start_s is None else float(requested.start_s)
        stop_value = None if requested.stop_s is None else float(requested.stop_s)
        start = int(np.searchsorted(time_s, origin + start_value, side="left"))
        stop = len(time_s) if stop_value is None else int(np.searchsorted(time_s, origin + stop_value, side="left"))
    elif time_basis == "active":
        if requested.start_s is None and requested.stop_s is None:
            start, stop = 0, len(time_s)
        else:
            active_indices = np.flatnonzero(activity)
            if len(active_indices) == 0:
                raise ValueError("Cannot resolve an active-time window because the activity mask is empty")
            active_weights = durations[active_indices]
            active_ends = np.cumsum(active_weights)
            active_starts = active_ends - active_weights
            total_active = float(active_ends[-1])
            start_value = 0.0 if requested.start_s is None else float(requested.start_s)
            stop_value = total_active if requested.stop_s is None else float(requested.stop_s)
            if start_value >= total_active:
                raise ValueError(
                    f"Active window starts at {start_value:.3f}s but the recording has only {total_active:.3f}s"
                )
            # Assign every active sample according to the accumulated active
            # time at the start of its sample interval.  Adjacent requested
            # ranges therefore resolve to disjoint sample ranges even when a
            # boundary is not exactly representable at the sample cadence.
            start_pos = int(np.searchsorted(active_starts, start_value, side="left"))
            stop_pos = int(np.searchsorted(active_starts, stop_value, side="left"))
            if start_pos >= len(active_indices):
                raise ValueError("Active window starts beyond the final active sample")
            stop_pos = min(max(stop_pos, start_pos + 1), len(active_indices))
            start = int(active_indices[start_pos])
            stop = int(active_indices[stop_pos - 1]) + 1
    else:
        raise ValueError(f"Unknown time basis {time_basis!r}")

    start = max(0, min(start, len(time_s)))
    stop = max(start, min(stop, len(time_s)))
    if stop <= start:
        raise ValueError(f"Requested window resolves to an empty sample range [{start}, {stop})")

    active_duration = float(np.sum(durations[start:stop] * activity[start:stop]))
    wall_stop = float(time_s[stop - 1] + durations[stop - 1])
    return ResolvedWindow(
        sample_start=start,
        sample_stop=stop,
        wall_start_s=float(time_s[start]),
        wall_stop_s=wall_stop,
        active_duration_s=active_duration,
    )


@dataclass(frozen=True)
class MagTravelCalibration:
    """Portable fitted curve state, excluding the target-specific travel anchor."""

    pipeline: PipelineKind
    feature_key: str
    training_log: str
    training_time_basis: TimeBasis
    training_start_s: float | None
    training_stop_s: float | None
    training_sample_start: int
    training_sample_stop: int
    training_chunk_count: int
    method: CalibrationMethod = "self_supervised_power"
    coefficients: tuple[float, float, float] | None = None
    pred_soft_mg: float | None = None
    mag_knots: tuple[float, ...] = ()
    travel_knots: tuple[float, ...] = ()
    travel_offset_mm: float = 0.0
    model_config: dict[str, Any] = field(default_factory=dict)
    training_diagnostics: dict[str, Any] = field(default_factory=dict)
    source_fingerprint: str | None = None
    format_version: int = 2

    def __post_init__(self):
        if self.pipeline not in ("front", "rear"):
            raise ValueError(f"Unknown pipeline {self.pipeline!r}")
        if self.method in ("self_supervised_power", "oracle_power"):
            if self.coefficients is None or len(self.coefficients) != 3:
                raise ValueError("Power calibration requires exactly three coefficients")
            if self.pred_soft_mg is None:
                raise ValueError("Power calibration requires pred_soft_mg")
        elif self.method in ("oracle_isotonic", "oracle_binned_median"):
            if len(self.mag_knots) < 2 or len(self.mag_knots) != len(self.travel_knots):
                raise ValueError("Oracle calibration requires matching mag/travel knots")
            if np.any(np.diff(np.asarray(self.mag_knots, dtype=float)) <= 0):
                raise ValueError("Oracle magnetic knots must be strictly increasing")
        else:
            raise ValueError(f"Unknown calibration method {self.method!r}")

    def make_model(self) -> MagToTravelModel:
        if self.method not in ("self_supervised_power", "oracle_power"):
            raise ValueError(f"Calibration method {self.method!r} is not a power-law model")
        model = MagToTravelModel(pred_soft_mg=float(self.pred_soft_mg))
        model.set_coeffs(np.asarray(self.coefficients, dtype=float))
        return model

    def predict(self, mag: np.ndarray | float) -> np.ndarray:
        if self.method in ("self_supervised_power", "oracle_power"):
            return self.make_model().pred_x(mag) + float(self.travel_offset_mm)
        return np.interp(
            np.asarray(mag, dtype=float),
            np.asarray(self.mag_knots, dtype=float),
            np.asarray(self.travel_knots, dtype=float),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MagTravelCalibration":
        payload = dict(value)
        if payload.get("coefficients") is not None:
            payload["coefficients"] = tuple(float(item) for item in payload["coefficients"])
        payload["mag_knots"] = tuple(float(item) for item in payload.get("mag_knots", ()))
        payload["travel_knots"] = tuple(float(item) for item in payload.get("travel_knots", ()))
        return cls(**payload)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MagTravelCalibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
