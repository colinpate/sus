#!/usr/bin/env python3
"""Compare anti-aliased nuisance observations and full-rate gyro integration.

The validated correction samples 100 Hz pipeline signals every tenth sample.
This experiment separates two possible improvements:

* integrate gyro1 at its full filtered rate before sampling 10 Hz rotations;
* form the nuisance observation from a low-pass or window-aggregated full-rate
  magnetic residual instead of selecting one potentially aliased sample.

Encoder travel is read only after every prediction has been generated.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, sosfiltfilt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.mag_nuisance_core import (  # noqa: E402
    PRIMARY_MAG_TO_GYRO,
    MagSolverWeights,
    fit_scalar_parameterized_xyz,
    integrate_gyro,
    interpolate_nuisance_fields as interpolate_state_fields,
    smooth_body_world_fields,
    solve_iterative_correction,
)
from tools.front.mag_nuisance.experiment_unsupervised_mag_xyz import (  # noqa: E402
    DEFAULT_LOGS,
    FORK,
    aligned,
    aligned_first,
    flatten,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class MultiRateResult:
    travel: np.ndarray
    state_body: np.ndarray
    state_world: np.ndarray
    state_update_mask: np.ndarray
    iteration_change_mm: list[float]


def centered_window_samples(source_hz: float, window_s: float) -> int:
    samples = max(1, round(source_hz * window_s))
    if samples % 2 == 0:
        samples += 1
    return samples


def aggregate_residual(
    residual: np.ndarray,
    state_index: np.ndarray,
    source_hz: float,
    method: str,
    *,
    cutoff_hz: float = 4.0,
    window_s: float = 0.1,
) -> np.ndarray:
    """Anti-alias or aggregate a full-rate XYZ residual onto state samples."""

    residual = np.asarray(residual, dtype=float)
    if method == "point":
        return residual[state_index]
    if method == "lowpass":
        if not 0.0 < cutoff_hz < 0.5 * source_hz:
            raise ValueError("cutoff_hz must lie below the source Nyquist rate")
        sos = butter(4, cutoff_hz, btype="low", fs=source_hz, output="sos")
        aggregated = sosfiltfilt(sos, residual, axis=0)
    elif method == "mean":
        window = centered_window_samples(source_hz, window_s)
        aggregated = uniform_filter1d(
            residual, size=window, axis=0, mode="nearest"
        )
    elif method == "median":
        window = centered_window_samples(source_hz, window_s)
        aggregated = median_filter(residual, size=(window, 1), mode="nearest")
    else:
        raise ValueError(f"Unknown residual aggregation method: {method}")
    return aggregated[state_index]


def solve_multirate_correction(
    time_s: np.ndarray,
    gyro_dps: np.ndarray,
    mag_xyz: np.ndarray,
    initial_travel: np.ndarray,
    xyz_model: object,
    state_index: np.ndarray,
    weights: MagSolverWeights,
    *,
    method: str,
    iterations: int = 4,
    cutoff_hz: float = 4.0,
    window_s: float = 0.1,
) -> MultiRateResult:
    """Alternate full-rate residual formation and a low-rate field smoother."""

    time_s = np.asarray(time_s, dtype=float)
    gyro_dps = np.asarray(gyro_dps, dtype=float)
    mag_xyz = np.asarray(mag_xyz, dtype=float)
    initial_travel = np.asarray(initial_travel, dtype=float).reshape(-1)
    source_hz = 1.0 / float(np.median(np.diff(time_s)))
    state_time = time_s[state_index]
    full_rotations = integrate_gyro(time_s, gyro_dps)
    state_rotations = full_rotations[state_index]

    travel = initial_travel.copy()
    changes: list[float] = []
    state_body = np.zeros((len(state_index), 3))
    state_world = np.zeros((len(state_index), 3))
    state_update_mask = np.zeros(len(state_index), dtype=bool)

    for _ in range(iterations):
        expected_full = xyz_model.predict(travel)
        residual_full = mag_xyz - expected_full
        state_observation = aggregate_residual(
            residual_full,
            state_index,
            source_hz,
            method,
            cutoff_hz=cutoff_hz,
            window_s=window_s,
        )
        state_update_mask = xyz_model.weak(
            travel[state_index], weights.mag_update_threshold
        )
        state_body, state_world = smooth_body_world_fields(
            state_time,
            np.zeros((len(state_index), 3)),
            state_observation,
            np.zeros_like(state_observation),
            state_update_mask,
            weights,
            body_to_reference_rotations=state_rotations,
        )
        body_full, world_full = interpolate_state_fields(
            time_s,
            state_time,
            full_rotations,
            state_rotations,
            state_body,
            state_world,
        )
        inferred = xyz_model.infer(mag_xyz - body_full - world_full)
        apply_mask = xyz_model.weak(
            travel, weights.mag_update_threshold
        ) & (np.linalg.norm(mag_xyz, axis=1) <= weights.mag_update_threshold)
        next_travel = initial_travel.copy()
        next_travel[apply_mask] = inferred[apply_mask]
        changes.append(float(np.sqrt(np.mean((next_travel - travel) ** 2))))
        travel = next_travel

    state_update_mask = xyz_model.weak(
        travel[state_index], weights.mag_update_threshold
    ) & (
        np.linalg.norm(mag_xyz[state_index], axis=1)
        <= weights.mag_update_threshold
    )
    return MultiRateResult(
        travel=travel,
        state_body=state_body,
        state_world=state_world,
        state_update_mask=state_update_mask,
        iteration_change_mm=changes,
    )


def score(
    log_name: str,
    method: str,
    region: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | str]:
    error = prediction[mask] - truth[mask]
    return {
        "log": log_name,
        "fork": FORK.get(log_name, "unknown"),
        "method": method,
        "region": region,
        "samples": int(np.sum(mask)),
        "rmse_mm": float(np.sqrt(np.mean(error**2))),
        "mae_mm": float(np.mean(np.abs(error))),
        "bias_mm": float(np.mean(error)),
        "centered_rmse_mm": float(np.std(error)),
    }


def evaluate_log(
    name: str,
    cache_root: Path,
    state_hz: float,
    alpha: float,
    iterations: int,
    cutoff_hz: list[float],
    window_s: float,
    weights: MagSolverWeights,
    include_filtered: bool,
) -> tuple[list[dict], dict]:
    cache = np.load(cache_root / name / "cache" / "all.npz")
    time_s = flatten(cache["mag/lpf__t"])
    source_hz = 1.0 / float(np.median(np.diff(time_s)))
    stride = max(1, round(source_hz / state_hz))
    state_index = np.arange(0, len(time_s), stride)

    mag_xyz = np.asarray(cache["mag/lpf__x"], dtype=float) @ PRIMARY_MAG_TO_GYRO.T
    gyro_dps = np.asarray(cache["gyro/lpf/gyro1__x"], dtype=float)
    scalar_mag = flatten(
        aligned_first(
            cache, state_index, "mag/norm/corr/lpf", "mag/proj/corr/lpf"
        )
    )
    initial_travel = flatten(cache["travel/solved__x"])
    raw_scalar_travel = flatten(cache["travel/mag_model__x"])
    adjusted_scalar_travel = flatten(cache["travel/mag_model/adj__x"])
    scalar_offset = float(np.median(adjusted_scalar_travel - raw_scalar_travel))
    xyz_model = fit_scalar_parameterized_xyz(
        scalar_mag,
        mag_xyz[state_index],
        np.asarray(cache["mag_model_coeffs"], dtype=float),
        scalar_offset,
        degree=2,
        travel_max_mm=210.0,
    )

    state_time = time_s[state_index]
    state_gyro = gyro_dps[state_index]
    state_mag = mag_xyz[state_index]
    state_initial = initial_travel[state_index]
    full_rotations = integrate_gyro(time_s, gyro_dps)
    current = solve_iterative_correction(
        state_time,
        state_gyro,
        state_mag,
        state_initial,
        xyz_model,
        weights,
        iterations=iterations,
    )
    full_gyro_point = solve_iterative_correction(
        state_time,
        state_gyro,
        state_mag,
        state_initial,
        xyz_model,
        weights,
        iterations=iterations,
        body_to_reference_rotations=full_rotations[state_index],
    )

    proposals: dict[str, np.ndarray] = {
        "pipeline": state_initial,
        "point_decimated_gyro": current.travel,
        "point_full_gyro": full_gyro_point.travel,
    }
    details: dict[str, object] = {
        "log": name,
        "source_hz": source_hz,
        "effective_state_hz": source_hz / stride,
        "state_samples": len(state_index),
        "xyz_bin_count": xyz_model.bin_count,
        "methods": {
            "point_decimated_gyro": {
                "iteration_change_mm": current.iteration_change_mm,
                "update_fraction": float(np.mean(current.update_mask)),
            },
            "point_full_gyro": {
                "iteration_change_mm": full_gyro_point.iteration_change_mm,
                "update_fraction": float(np.mean(full_gyro_point.update_mask)),
            },
        },
    }
    point_multirate = solve_multirate_correction(
        time_s,
        gyro_dps,
        mag_xyz,
        initial_travel,
        xyz_model,
        state_index,
        weights,
        method="point",
        iterations=iterations,
        window_s=window_s,
    )
    proposals["point_multirate_full_gyro"] = point_multirate.travel[state_index]
    details["methods"]["point_multirate_full_gyro"] = {
        "iteration_change_mm": point_multirate.iteration_change_mm,
        "update_fraction": float(np.mean(point_multirate.state_update_mask)),
    }
    if include_filtered:
        for cutoff in cutoff_hz:
            method_name = f"lowpass{cutoff:g}_full_gyro"
            result = solve_multirate_correction(
                time_s,
                gyro_dps,
                mag_xyz,
                initial_travel,
                xyz_model,
                state_index,
                weights,
                method="lowpass",
                iterations=iterations,
                cutoff_hz=cutoff,
                window_s=window_s,
            )
            proposals[method_name] = result.travel[state_index]
            details["methods"][method_name] = {
                "iteration_change_mm": result.iteration_change_mm,
                "update_fraction": float(np.mean(result.state_update_mask)),
            }
        for aggregation in ("mean", "median"):
            method_name = f"{aggregation}{window_s * 1000:g}ms_full_gyro"
            result = solve_multirate_correction(
                time_s,
                gyro_dps,
                mag_xyz,
                initial_travel,
                xyz_model,
                state_index,
                weights,
                method=aggregation,
                iterations=iterations,
                cutoff_hz=cutoff_hz[-1],
                window_s=window_s,
            )
            proposals[method_name] = result.travel[state_index]
            details["methods"][method_name] = {
                "iteration_change_mm": result.iteration_change_mm,
                "update_fraction": float(np.mean(result.state_update_mask)),
            }

    # Encoder truth first appears here.
    truth = flatten(aligned(cache, "travel", state_index))
    if "boring_mask" in cache and len(cache["boring_mask"]) == len(time_s):
        active = np.asarray(cache["boring_mask"], dtype=bool).reshape(-1)[state_index]
    else:
        active = np.ones(len(state_index), dtype=bool)
    active &= np.isfinite(truth) & (truth >= 0.0)
    measured_weak = np.linalg.norm(state_mag, axis=1) < weights.mag_update_threshold
    regions = {
        "all": active,
        "weak": active & measured_weak,
        "strong": active & ~measured_weak,
    }

    rows: list[dict] = []
    for method_name, proposal in proposals.items():
        if method_name == "pipeline":
            prediction = state_initial
        else:
            prediction = state_initial + alpha * (proposal - state_initial)
        for region, mask in regions.items():
            rows.append(
                score(name, method_name, region, prediction, truth, mask)
            )
    return rows, details


def write_report(frame: pd.DataFrame, output_dir: Path) -> None:
    weak = frame[frame["region"] == "weak"]
    baseline = weak[weak["method"] == "pipeline"].set_index("log")["rmse_mm"]
    log_count = len(baseline)
    rows = []
    for method, group in weak.groupby("method"):
        values = group.set_index("log")["rmse_mm"]
        delta = values - baseline
        rows.append(
            {
                "method": method,
                "mean_rmse_mm": float(values.mean()),
                "median_rmse_mm": float(values.median()),
                "mean_delta_mm": float(delta.mean()),
                "improved_logs": int(np.sum(delta < 0)),
                "worst_delta_mm": float(delta.max()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("mean_rmse_mm")
    summary.to_csv(output_dir / "summary.csv", index=False)

    lines = [
        "# Multirate nuisance-observation experiment",
        "",
        "All results use the encoder-blind quadratic XYZ path, four field/travel",
        "iterations, the 1500 mG update/application gate, and the configured output",
        "blend. Encoder travel is used only for these metrics.",
        "",
        "| Method | Mean weak RMSE | Median | Mean delta | Improved | Worst delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| `{row.method}` | {row.mean_rmse_mm:.3f} | "
            f"{row.median_rmse_mm:.3f} | {row.mean_delta_mm:+.3f} | "
            f"{row.improved_logs}/{log_count} | {row.worst_delta_mm:+.3f} |"
        )
    lines.extend(
        [
            "",
            "Per-log and all-region measurements are in `metrics.csv`; state and",
            "iteration diagnostics are in `details.json`.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", default=list(DEFAULT_LOGS))
    parser.add_argument(
        "--cache-root", type=Path, default=REPO_ROOT / "backend/run_artifacts"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "reports/front_mag_nuisance/observability/multirate_observations"
        ),
    )
    parser.add_argument("--state-hz", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--cutoff-hz", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    parser.add_argument("--window-s", type=float, default=0.1)
    parser.add_argument("--mag-sigma", type=float, default=40.0)
    parser.add_argument(
        "--skip-filtered",
        action="store_true",
        help="Run only point-sampled gyro controls (useful for weight sweeps).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.state_hz <= 0 or args.iterations < 1 or args.window_s <= 0:
        raise ValueError("rates/window must be positive and iterations at least one")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    if any(value <= 0 or value >= 0.5 * args.state_hz for value in args.cutoff_hz):
        raise ValueError("cutoffs must lie between zero and the state Nyquist rate")
    weights = MagSolverWeights(mag_sigma=args.mag_sigma)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    details: list[dict] = []
    for name in args.logs:
        print(f"Evaluating {name}...", flush=True)
        log_rows, log_details = evaluate_log(
            name,
            args.cache_root,
            args.state_hz,
            args.alpha,
            args.iterations,
            args.cutoff_hz,
            args.window_s,
            weights,
            not args.skip_filtered,
        )
        rows.extend(log_rows)
        details.append(log_details)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "metrics.csv", index=False)
    with (args.output_dir / "details.json").open("w") as handle:
        json.dump(
            {
                "config": {
                    "state_hz": args.state_hz,
                    "iterations": args.iterations,
                    "alpha": args.alpha,
                    "cutoff_hz": args.cutoff_hz,
                    "window_s": args.window_s,
                    "weights": asdict(weights),
                },
                "logs": details,
            },
            handle,
            indent=2,
        )
    write_report(frame, args.output_dir)
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
