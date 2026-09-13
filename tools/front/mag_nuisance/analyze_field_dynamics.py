#!/usr/bin/env python3
"""Summarize body/world nuisance states from the four-iteration correction.

The solver stores both fields in the gyro/body frame.  A world-fixed field is
supposed to rotate in that frame as the bike rotates, so this script reports:

* body-field direction change in the body frame;
* world-field direction change after rotating it back into the initial/world
  frame (the useful stationarity statistic); and
* world-field direction change as seen in the body frame (mostly bike motion).

Direction rate is the angle between vectors one second apart divided by the
actual elapsed time.  A finite lag is much easier to interpret than a noisy
single-sample derivative at the 10 Hz state rate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.front.mag_nuisance.experiment_mag_nuisance_observability import (  # noqa: E402
    load_signals,
)
from tools.front.mag_nuisance.experiment_unsupervised_mag_xyz import (  # noqa: E402
    DEFAULT_LOGS,
    FORK,
)
from backend.mag_nuisance_core import (  # noqa: E402
    MagSolverWeights,
    integrate_gyro,
    solve_iterative_correction,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def vector_magnitude(vectors: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(vectors, dtype=float), axis=1)


def direction_rate_dps(
    vectors: np.ndarray,
    time_s: np.ndarray,
    *,
    lag_s: float,
    min_magnitude_mg: float,
) -> np.ndarray:
    """Return net angular displacement per second over a finite time lag."""

    vectors = np.asarray(vectors, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    magnitude = vector_magnitude(vectors)
    unit = vectors / np.maximum(magnitude[:, np.newaxis], 1e-12)
    lag_samples = max(1, round(lag_s / np.median(np.diff(time_s))))

    rate = np.full(len(time_s), np.nan)
    dot = np.einsum("ij,ij->i", unit[lag_samples:], unit[:-lag_samples])
    angle_deg = np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))
    elapsed = time_s[lag_samples:] - time_s[:-lag_samples]
    valid = (
        (magnitude[lag_samples:] >= min_magnitude_mg)
        & (magnitude[:-lag_samples] >= min_magnitude_mg)
        & (elapsed > 0.0)
    )
    destination = np.flatnonzero(valid) + lag_samples
    rate[destination] = angle_deg[valid] / elapsed[valid]
    return rate


def vector_angle_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_magnitude = vector_magnitude(left)
    right_magnitude = vector_magnitude(right)
    denominator = np.maximum(left_magnitude * right_magnitude, 1e-12)
    cosine = np.einsum("ij,ij->i", left, right) / denominator
    return np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))


def finite_percentile(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanpercentile(values[np.isfinite(values)], percentile))


def summarize_series(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_p10": finite_percentile(values, 10.0),
        f"{prefix}_median": finite_percentile(values, 50.0),
        f"{prefix}_p90": finite_percentile(values, 90.0),
    }


def downsample_frame(frame: pd.DataFrame, plot_hz: float) -> pd.DataFrame:
    """Keep a regular, bounded-size view without changing full-rate stats."""

    source_hz = 1.0 / np.median(np.diff(frame["time_s"]))
    stride = max(1, round(source_hz / plot_hz))
    sampled = frame.iloc[::stride].copy()
    if sampled.index[-1] != frame.index[-1]:
        sampled = pd.concat([sampled, frame.iloc[[-1]]], ignore_index=True)
    return sampled


def analyze_log(
    name: str,
    args: argparse.Namespace,
    weights: MagSolverWeights,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    signals = load_signals(
        name,
        args.cache_root,
        args.state_hz,
        args.degree,
        args.travel_max_mm,
    )
    result = solve_iterative_correction(
        signals.time_s,
        signals.gyro_dps,
        signals.mag_xyz,
        signals.initial_travel,
        signals.xyz_model,
        weights,
        iterations=4,
    )

    rotations = integrate_gyro(signals.time_s, signals.gyro_dps)
    world_initial = np.einsum("nij,nj->ni", rotations, result.world_field)

    body_magnitude = vector_magnitude(result.body_field)
    world_magnitude = vector_magnitude(result.world_field)
    total_magnitude = vector_magnitude(result.correction)
    body_rate = direction_rate_dps(
        result.body_field,
        signals.time_s,
        lag_s=args.direction_lag_s,
        min_magnitude_mg=args.direction_min_magnitude_mg,
    )
    world_initial_rate = direction_rate_dps(
        world_initial,
        signals.time_s,
        lag_s=args.direction_lag_s,
        min_magnitude_mg=args.direction_min_magnitude_mg,
    )
    world_body_rate = direction_rate_dps(
        result.world_field,
        signals.time_s,
        lag_s=args.direction_lag_s,
        min_magnitude_mg=args.direction_min_magnitude_mg,
    )
    component_angle = vector_angle_deg(result.body_field, result.world_field)

    summary: dict[str, float | str] = {
        "log": name,
        "fork": FORK.get(name, "unknown"),
        "duration_s": float(signals.time_s[-1] - signals.time_s[0]),
        "update_fraction": float(np.mean(result.update_mask)),
        "direction_lag_s": args.direction_lag_s,
        "direction_min_magnitude_mg": args.direction_min_magnitude_mg,
        "body_world_angle_median_deg": finite_percentile(component_angle, 50.0),
    }
    summary.update(summarize_series("body_magnitude_mg", body_magnitude))
    summary.update(summarize_series("world_magnitude_mg", world_magnitude))
    summary.update(summarize_series("total_magnitude_mg", total_magnitude))
    summary.update(summarize_series("body_direction_rate_dps", body_rate))
    summary.update(
        summarize_series("world_initial_direction_rate_dps", world_initial_rate)
    )
    summary.update(
        summarize_series("world_body_direction_rate_dps", world_body_rate)
    )

    time_s = signals.time_s - signals.time_s[0]
    frame = pd.DataFrame(
        {
            "log": name,
            "fork": FORK.get(name, "unknown"),
            "time_s": time_s,
            "body_magnitude_mg": body_magnitude,
            "world_magnitude_mg": world_magnitude,
            "total_magnitude_mg": total_magnitude,
            "body_direction_rate_dps": body_rate,
            "world_initial_direction_rate_dps": world_initial_rate,
            "world_body_direction_rate_dps": world_body_rate,
            "body_world_angle_deg": component_angle,
            "update_active": result.update_mask.astype(int),
        }
    )
    return summary, downsample_frame(frame, args.plot_hz)


def cohort_summary(per_log: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in per_log.columns
        if column not in {"log", "fork"}
        and np.issubdtype(per_log[column].dtype, np.number)
    ]
    rows = []
    for fork, frame in [("all", per_log), *per_log.groupby("fork")]:
        row: dict[str, float | str] = {"fork": fork, "logs": len(frame)}
        row.update({column: float(frame[column].median()) for column in numeric})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_per_log_summary(per_log: pd.DataFrame, output: Path) -> None:
    display = per_log.sort_values(["fork", "log"]).reset_index(drop=True)
    y = np.arange(len(display))
    colors = display["fork"].map({"fox36": "#3274a1", "boxxer": "#e1812c"})

    figure, axes = plt.subplots(1, 2, figsize=(13, 7.5), sharey=True)
    magnitude_columns = [
        ("body_magnitude_mg_median", "body"),
        ("world_magnitude_mg_median", "world"),
        ("total_magnitude_mg_median", "sum"),
    ]
    offsets = [-0.20, 0.0, 0.20]
    markers = ["o", "s", "D"]
    for (column, label), offset, marker in zip(
        magnitude_columns, offsets, markers, strict=True
    ):
        axes[0].scatter(
            display[column], y + offset, s=31, marker=marker, label=label,
            facecolors=colors, edgecolors="white", linewidths=0.5,
        )

    rate_columns = [
        ("body_direction_rate_dps_p90", "body in body frame"),
        ("world_initial_direction_rate_dps_p90", "world in initial frame"),
    ]
    for (column, label), offset, marker in zip(
        rate_columns, [-0.11, 0.11], markers[:2], strict=True
    ):
        axes[1].scatter(
            display[column], y + offset, s=31, marker=marker, label=label,
            facecolors=colors, edgecolors="white", linewidths=0.5,
        )

    axes[0].set_yticks(y, display["log"])
    axes[0].set_xlabel("Median field magnitude (mG)")
    axes[1].set_xlabel(f"90th-percentile {display['direction_lag_s'].iloc[0]:g} s direction rate (deg/s)")
    axes[0].set_title("Estimated nuisance magnitude")
    axes[1].set_title("Directional change")
    for axis in axes:
        axis.grid(axis="x", alpha=0.25)
        axis.legend(loc="lower right", fontsize=8)
    figure.suptitle("Four-iteration body/world nuisance states", fontsize=14)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", default=list(DEFAULT_LOGS))
    parser.add_argument(
        "--cache-root", type=Path, default=REPO_ROOT / "backend/run_artifacts"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports/front_mag_nuisance/field_dynamics",
    )
    parser.add_argument("--state-hz", type=float, default=10.0)
    parser.add_argument("--plot-hz", type=float, default=1.0)
    parser.add_argument("--direction-lag-s", type=float, default=1.0)
    parser.add_argument("--direction-min-magnitude-mg", type=float, default=40.0)
    parser.add_argument("--degree", type=int, choices=(1, 2), default=2)
    parser.add_argument("--travel-max-mm", type=float, default=210.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_hz <= 0.0 or args.plot_hz > args.state_hz:
        raise ValueError("plot-hz must be positive and no larger than state-hz")
    if args.direction_lag_s <= 0.0:
        raise ValueError("direction-lag-s must be positive")
    if args.direction_min_magnitude_mg <= 0.0:
        raise ValueError("direction-min-magnitude-mg must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = MagSolverWeights()
    summaries = []
    timeseries = []
    for name in args.logs:
        print(f"Analyzing {name}...", flush=True)
        summary, frame = analyze_log(name, args, weights)
        summaries.append(summary)
        timeseries.append(frame)

    per_log = pd.DataFrame(summaries)
    cohort = cohort_summary(per_log)
    per_log.to_csv(args.output_dir / "per_log_summary.csv", index=False)
    cohort.to_csv(args.output_dir / "cohort_summary.csv", index=False)
    pd.concat(timeseries, ignore_index=True).to_csv(
        args.output_dir / "timeseries_1hz.csv", index=False, float_format="%.5f"
    )
    plot_per_log_summary(per_log, args.output_dir / "per_log_summary.png")

    metadata = {
        "description": "Final states from the four-iteration isotropic correction",
        "weights": asdict(weights),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "frame_semantics": {
            "body_field": "body/sensor frame",
            "world_field": "body/sensor frame in solver output",
            "world_initial": "world_field rotated by integrated gyro into the initial frame",
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
