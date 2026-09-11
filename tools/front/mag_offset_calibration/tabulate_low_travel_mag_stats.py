#!/usr/bin/env python3
"""Tabulate raw and nuisance-corrected mag magnitude at low GT travel."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.log_registry import DEFAULT_REGISTRY_PATH  # noqa: E402
from backend.mag_nuisance_core import (  # noqa: E402
    PRIMARY_MAG_TO_GYRO,
    integrate_gyro,
    interpolate_nuisance_fields,
)
from tools.front.mag_offset_calibration.sweep_post_mag_correction_fusion import (  # noqa: E402
    DEFAULT_HELD_OUT_SET,
    DEFAULT_TUNING_SETS,
    resolve_cohorts,
    write_csv,
)


def flatten(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def reconstruct_corrected_magnitude(cache: np.lib.npyio.NpzFile) -> np.ndarray:
    cached_key = "mag/nuisance/corrected/norm__x"
    if cached_key in cache:
        return flatten(cache[cached_key])

    full_time = flatten(cache["mag/lpf__t"])
    mag_recorded = np.asarray(cache["mag/lpf__x"], dtype=float)
    gyro = np.asarray(cache["gyro/lpf/gyro1__x"], dtype=float)
    state_time = flatten(cache["mag/nuisance/body/10hz__t"])
    body_state = np.asarray(cache["mag/nuisance/body/10hz__x"], dtype=float)
    world_state = np.asarray(cache["mag/nuisance/world/10hz__x"], dtype=float)

    if mag_recorded.shape != (len(full_time), 3):
        raise ValueError("mag/lpf must have shape (n, 3)")
    if gyro.shape != (len(full_time), 3):
        raise ValueError("gyro/lpf/gyro1 must align with mag/lpf")
    full_rotations = integrate_gyro(full_time, gyro)
    state_index = np.searchsorted(full_time, state_time)
    state_index = np.clip(state_index, 0, len(full_time) - 1)
    if not np.allclose(full_time[state_index], state_time, rtol=0.0, atol=1e-8):
        raise ValueError("nuisance-state times are not samples of the full timeline")
    body_full, world_full = interpolate_nuisance_fields(
        full_time,
        state_time,
        full_rotations,
        full_rotations[state_index],
        body_state,
        world_state,
    )
    mag_gyro = mag_recorded @ PRIMARY_MAG_TO_GYRO.T
    corrected_xyz = mag_gyro - body_full - world_full
    return np.linalg.norm(corrected_xyz, axis=1)


def range_stats(
    values: np.ndarray, truth: np.ndarray, lower_mm: float, upper_mm: float
) -> tuple[int, float, float]:
    mask = (
        np.isfinite(values)
        & np.isfinite(truth)
        & (truth > lower_mm)
        & (truth < upper_mm)
    )
    selected = np.asarray(values, dtype=float)[mask]
    if not len(selected):
        return 0, float("nan"), float("nan")
    return len(selected), float(np.mean(selected)), float(np.std(selected))


def render_markdown(rows: Sequence[dict[str, object]]) -> str:
    lines = [
        "# Low-travel magnetic magnitude statistics",
        "",
        "`mag_norm` is the exact scalar consumed by `GetMagTravelRefPoint` (`mag/norm/corr/lpf`). `corrected_mag` is reconstructed as the norm of full-rate filtered XYZ after subtracting the interpolated nuisance body and world fields, matching `MagNuisanceFullRateCorrection`.",
        "",
        "All ranges use strict ground-truth bounds and all finite samples; standard deviations use NumPy's population convention (`ddof=0`).",
        "",
        "## Cohort macro averages",
        "",
        "Each cell is the mean of the per-log mean and the mean of the per-log within-range standard deviation.",
        "",
        "| cohort | logs | mag_norm 0-5 mean / std | corrected 0-5 mean / std | mag_norm 0-25 mean / std | corrected 0-25 mean / std |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cohort in dict.fromkeys(str(row["cohort"]) for row in rows):
        selected = [row for row in rows if row["cohort"] == cohort]
        average = lambda key: float(np.mean([float(row[key]) for row in selected]))
        lines.append(
            f"| {cohort} | {len(selected)} | {average('mag_norm_0_5_mean_mg'):.1f} / {average('mag_norm_0_5_std_mg'):.1f} | {average('corrected_mag_0_5_mean_mg'):.1f} / {average('corrected_mag_0_5_std_mg'):.1f} | {average('mag_norm_0_25_mean_mg'):.1f} / {average('mag_norm_0_25_std_mg'):.1f} | {average('corrected_mag_0_25_mean_mg'):.1f} / {average('corrected_mag_0_25_std_mg'):.1f} |"
        )
    lines.extend([
        "",
        "## Per-log table",
        "",
        "| cohort | log | N 0-5 | mag_norm 0-5 mean | mag_norm 0-5 std | corrected 0-5 mean | corrected 0-5 std | N 0-25 | mag_norm 0-25 mean | mag_norm 0-25 std | corrected 0-25 mean | corrected 0-25 std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in rows:
        lines.append(
            "| {cohort} | {log} | {n_0_5} | {mag_norm_0_5_mean_mg:.1f} | {mag_norm_0_5_std_mg:.1f} | {corrected_mag_0_5_mean_mg:.1f} | {corrected_mag_0_5_std_mg:.1f} | {n_0_25} | {mag_norm_0_25_mean_mg:.1f} | {mag_norm_0_25_std_mg:.1f} | {corrected_mag_0_25_mean_mg:.1f} | {corrected_mag_0_25_std_mg:.1f} |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--cache-root", type=Path, default=REPO_ROOT / "backend" / "run_artifacts"
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohorts = resolve_cohorts(
        args.registry, (*DEFAULT_TUNING_SETS, DEFAULT_HELD_OUT_SET)
    )
    rows: list[dict[str, object]] = []
    required = {
        "travel__x",
        "mag/norm/corr/lpf__x",
        "mag/lpf__t",
        "mag/lpf__x",
        "gyro/lpf/gyro1__x",
        "mag/nuisance/body/10hz__t",
        "mag/nuisance/body/10hz__x",
        "mag/nuisance/world/10hz__x",
    }
    for cohort, log_names in cohorts.items():
        for log_name in log_names:
            cache_path = args.cache_root / log_name / "cache" / "all.npz"
            with np.load(cache_path, allow_pickle=False) as cache:
                missing = required - set(cache.files)
                if missing:
                    raise KeyError(f"{log_name}: missing cache fields {sorted(missing)}")
                truth = flatten(cache["travel__x"])
                mag_norm = flatten(cache["mag/norm/corr/lpf__x"])
                corrected_mag = reconstruct_corrected_magnitude(cache)
            if not (len(truth) == len(mag_norm) == len(corrected_mag)):
                raise ValueError(f"{log_name}: signal lengths do not match")
            n_0_5, raw_0_5_mean, raw_0_5_std = range_stats(
                mag_norm, truth, 0.0, 5.0
            )
            n_corr_0_5, corrected_0_5_mean, corrected_0_5_std = range_stats(
                corrected_mag, truth, 0.0, 5.0
            )
            n_0_25, raw_0_25_mean, raw_0_25_std = range_stats(
                mag_norm, truth, 0.0, 25.0
            )
            n_corr_0_25, corrected_0_25_mean, corrected_0_25_std = range_stats(
                corrected_mag, truth, 0.0, 25.0
            )
            if n_0_5 != n_corr_0_5 or n_0_25 != n_corr_0_25:
                raise ValueError(f"{log_name}: finite masks differ between scalars")
            rows.append(
                {
                    "cohort": cohort,
                    "log": log_name,
                    "n_0_5": n_0_5,
                    "mag_norm_0_5_mean_mg": raw_0_5_mean,
                    "mag_norm_0_5_std_mg": raw_0_5_std,
                    "corrected_mag_0_5_mean_mg": corrected_0_5_mean,
                    "corrected_mag_0_5_std_mg": corrected_0_5_std,
                    "n_0_25": n_0_25,
                    "mag_norm_0_25_mean_mg": raw_0_25_mean,
                    "mag_norm_0_25_std_mg": raw_0_25_std,
                    "corrected_mag_0_25_mean_mg": corrected_0_25_mean,
                    "corrected_mag_0_25_std_mg": corrected_0_25_std,
                }
            )
            print(f"{cohort}/{log_name}", flush=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_csv, rows)
    args.output_markdown.write_text(render_markdown(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
