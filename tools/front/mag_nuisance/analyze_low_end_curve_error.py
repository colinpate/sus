#!/usr/bin/env python3
"""Decompose low-travel scalar-magnet error into curve bias and scatter.

This is deliberately an encoder-supervised diagnostic.  For each encoder
travel bin it compares the learned power curve with the median measured
magnitude, then decomposes sample travel error into:

    total error = travel-bin mean error + within-bin residual

The first term is the operational error of the learned curve relative to the
encoder-binned curve.  The residual contains time-varying nuisance field,
sensor noise, and any state dependence not represented by travel alone.

Both raw and centered results are written.  Centering removes the mean
prediction error over the full active recording before analyzing the low end,
so the primary centered result does not charge a constant zero-reference error
to the learned curve's shape.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOGS = [
    "log-0056",
    "log-0063",
    "log-0046",
    "log-0048",
    "log-0049",
    "log-0054",
    "log-0055",
    "log-0058",
    "log-0071_183",
    "log-0072_184",
    "log-0073_185",
    "log-0078-valid",
    "log-0079",
    "log-0080-valid",
    "log-0081",
]
BOXXER_LOGS = {"log-0078-valid", "log-0079", "log-0080-valid", "log-0081"}


def flatten(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def cache_values(cache: np.lib.npyio.NpzFile, *keys: str) -> np.ndarray:
    """Read the first available series, preferring current cache keys."""

    for key in keys:
        cache_key = f"{key}__x"
        if cache_key in cache:
            return np.asarray(cache[cache_key])
    raise KeyError(f"None of the cache series are present: {', '.join(keys)}")


@dataclass(frozen=True)
class LearnedPowerCurve:
    x0_mg: float
    scale: float
    power: float
    offset_mm: float
    soft_mg: float = 50.0

    def travel_from_mag(self, magnitude_mg: np.ndarray | float) -> np.ndarray:
        delta = np.asarray(magnitude_mg, dtype=float) - self.x0_mg
        softened = (np.abs(delta) + self.soft_mg) ** self.power
        softened -= self.soft_mg**self.power
        return np.sign(delta) * softened * self.scale + self.offset_mm

    def mag_from_travel(self, travel_mm: np.ndarray | float) -> np.ndarray:
        adjusted = np.asarray(travel_mm, dtype=float) - self.offset_mm
        powered = np.abs(adjusted) / self.scale + self.soft_mg**self.power
        delta = np.maximum(powered ** (1.0 / self.power) - self.soft_mg, 0.0)
        return self.x0_mg + np.sign(adjusted) * delta


@dataclass(frozen=True)
class LogData:
    log: str
    fork: str
    travel: np.ndarray
    magnitude: np.ndarray
    mag_prediction: np.ndarray
    solved_prediction: np.ndarray
    active: np.ndarray
    curve: LearnedPowerCurve


def load_log(log: str, cache_root: Path) -> LogData:
    cache = np.load(cache_root / log / "cache" / "all.npz")
    travel = flatten(cache["travel__x"])
    magnitude = flatten(
        cache_values(cache, "mag/norm/corr/lpf", "mag/proj/corr/lpf")
    )
    raw_prediction = flatten(cache["travel/mag_model__x"])
    mag_prediction = flatten(cache["travel/mag_model/adj__x"])
    solved_prediction = flatten(cache["travel/solved__x"])
    boring = np.asarray(cache["boring_mask"], dtype=bool).reshape(-1)
    arrays = (travel, magnitude, raw_prediction, mag_prediction, solved_prediction)
    if len({len(values) for values in arrays}) != 1 or len(boring) != len(travel):
        raise ValueError(f"{log}: cache arrays do not align")
    active = boring & (travel >= 0.0)
    for values in arrays:
        active &= np.isfinite(values)
    coefficients = flatten(cache["mag_model_coeffs"])
    if len(coefficients) != 3:
        raise ValueError(f"{log}: expected three mag_model_coeffs")
    curve = LearnedPowerCurve(
        x0_mg=float(coefficients[0]),
        scale=float(coefficients[1]),
        power=float(coefficients[2]),
        offset_mm=float(np.median(mag_prediction - raw_prediction)),
    )
    return LogData(
        log=log,
        fork="boxxer" if log in BOXXER_LOGS else "fox36",
        travel=travel,
        magnitude=magnitude,
        mag_prediction=mag_prediction,
        solved_prediction=solved_prediction,
        active=active,
        curve=curve,
    )


def populated_bins(
    data: LogData,
    bin_mm: float,
    min_samples: int,
    max_travel_mm: float,
) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    bin_id = np.floor(data.travel / bin_mm).astype(int)
    bins: list[tuple[int, np.ndarray]] = []
    for value in np.unique(bin_id[data.active]):
        start = value * bin_mm
        if start < 0.0 or start >= max_travel_mm:
            continue
        mask = data.active & (bin_id == value)
        if int(np.sum(mask)) >= min_samples:
            bins.append((int(value), mask))
    return bin_id, bins


def build_bin_rows(
    data: LogData,
    bin_mm: float,
    min_samples: int,
    max_travel_mm: float,
) -> list[dict[str, float | int | str]]:
    _, bins = populated_bins(data, bin_mm, min_samples, max_travel_mm)
    rows: list[dict[str, float | int | str]] = []
    global_error_offset = float(
        np.mean((data.mag_prediction - data.travel)[data.active])
    )
    for value, mask in bins:
        travel = data.travel[mask]
        magnitude = data.magnitude[mask]
        mag_error = data.mag_prediction[mask] - travel
        solved_error = data.solved_prediction[mask] - travel
        encoder_travel_median = float(np.median(travel))
        encoder_mag_median = float(np.median(magnitude))
        predicted_at_encoder_mag = float(
            data.curve.travel_from_mag(encoder_mag_median)
        )
        model_mag_at_encoder_travel = float(
            data.curve.mag_from_travel(encoder_travel_median)
        )
        centered_model_mag_at_encoder_travel = float(
            data.curve.mag_from_travel(
                encoder_travel_median + global_error_offset
            )
        )
        rows.append(
            {
                "log": data.log,
                "fork": data.fork,
                "bin_start_mm": value * bin_mm,
                "bin_stop_mm": (value + 1) * bin_mm,
                "samples": int(np.sum(mask)),
                "encoder_travel_median_mm": encoder_travel_median,
                "encoder_mag_median_mg": encoder_mag_median,
                "model_travel_at_encoder_mag_mm": predicted_at_encoder_mag,
                "median_curve_error_mm": (
                    predicted_at_encoder_mag - encoder_travel_median
                ),
                "global_error_offset_mm": global_error_offset,
                "centered_median_curve_error_mm": (
                    predicted_at_encoder_mag
                    - global_error_offset
                    - encoder_travel_median
                ),
                "model_mag_at_encoder_travel_mg": model_mag_at_encoder_travel,
                "median_mag_curve_error_mg": (
                    encoder_mag_median - model_mag_at_encoder_travel
                ),
                "centered_model_mag_at_encoder_travel_mg": (
                    centered_model_mag_at_encoder_travel
                ),
                "centered_median_mag_curve_error_mg": (
                    encoder_mag_median - centered_model_mag_at_encoder_travel
                ),
                "mag_error_mean_mm": float(np.mean(mag_error)),
                "mag_error_std_mm": float(np.std(mag_error)),
                "solved_error_mean_mm": float(np.mean(solved_error)),
                "solved_error_std_mm": float(np.std(solved_error)),
                "mag_mad_mg": float(
                    np.median(np.abs(magnitude - encoder_mag_median))
                ),
            }
        )
    return rows


def decompose_prediction(
    data: LogData,
    prediction: np.ndarray,
    region_max_mm: float,
    bin_mm: float,
    min_samples: int,
) -> dict[str, float | int]:
    error = prediction - data.travel
    bin_id = np.floor(data.travel / bin_mm).astype(int)
    region = data.active & (data.travel < region_max_mm)
    bin_mean = np.full(len(error), np.nan, dtype=float)
    valid = np.zeros(len(error), dtype=bool)
    median_curve_errors: list[float] = []
    for value in np.unique(bin_id[region]):
        mask = region & (bin_id == value)
        if int(np.sum(mask)) < min_samples:
            continue
        bin_mean[mask] = float(np.mean(error[mask]))
        valid |= mask
        median_curve_errors.append(
            float(np.median(prediction[mask]) - np.median(data.travel[mask]))
        )
    if not np.any(valid):
        raise ValueError(f"{data.log}: no populated bins below {region_max_mm} mm")

    selected_error = error[valid]
    selected_curve = bin_mean[valid]
    residual = selected_error - selected_curve
    total_mse = float(np.mean(selected_error**2))
    curve_mse = float(np.mean(selected_curve**2))
    residual_mse = float(np.mean(residual**2))
    low_end_bias = float(np.mean(selected_error))
    curve_shape_rms = float(np.sqrt(np.mean((selected_curve - low_end_bias) ** 2)))
    total_rmse = float(np.sqrt(total_mse))
    residual_rmse = float(np.sqrt(residual_mse))
    return {
        "samples": int(np.sum(valid)),
        "total_rmse_mm": total_rmse,
        "curve_rms_mm": float(np.sqrt(curve_mse)),
        "scatter_rms_mm": residual_rmse,
        "curve_mse_fraction": curve_mse / total_mse,
        "scatter_mse_fraction": residual_mse / total_mse,
        "oracle_curve_fix_gain_mm": total_rmse - residual_rmse,
        "low_end_bias_mm": low_end_bias,
        "curve_shape_rms_mm": curve_shape_rms,
        "max_abs_median_curve_error_mm": float(
            np.max(np.abs(median_curve_errors))
        ),
    }


def build_summary_rows(
    data: LogData,
    region_max_values: list[float],
    bin_mm: float,
    min_samples: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for region_max_mm in region_max_values:
        mag = decompose_prediction(
            data, data.mag_prediction, region_max_mm, bin_mm, min_samples
        )
        mag_global_offset = float(
            np.mean((data.mag_prediction - data.travel)[data.active])
        )
        centered_mag_prediction = data.mag_prediction - mag_global_offset
        centered_mag = decompose_prediction(
            data, centered_mag_prediction, region_max_mm, bin_mm, min_samples
        )
        solved = decompose_prediction(
            data, data.solved_prediction, region_max_mm, bin_mm, min_samples
        )
        solved_global_offset = float(
            np.mean((data.solved_prediction - data.travel)[data.active])
        )
        centered_solved_prediction = data.solved_prediction - solved_global_offset
        centered_solved = decompose_prediction(
            data,
            centered_solved_prediction,
            region_max_mm,
            bin_mm,
            min_samples,
        )
        row: dict[str, float | int | str] = {
            "log": data.log,
            "fork": data.fork,
            "region_max_mm": region_max_mm,
            "bin_mm": bin_mm,
            "model_x0_mg": data.curve.x0_mg,
            "model_scale": data.curve.scale,
            "model_power": data.curve.power,
            "model_offset_mm": data.curve.offset_mm,
            "mag_global_error_offset_mm": mag_global_offset,
            "solved_global_error_offset_mm": solved_global_offset,
        }
        row.update({f"mag_{key}": value for key, value in mag.items()})
        row.update(
            {f"mag_centered_{key}": value for key, value in centered_mag.items()}
        )
        row.update({f"solved_{key}": value for key, value in solved.items()})
        row.update(
            {
                f"solved_centered_{key}": value
                for key, value in centered_solved.items()
            }
        )
        rows.append(row)
    return rows


def temporal_decomposition(
    data: LogData,
    prediction: np.ndarray,
    *,
    centered: bool,
    region_max_mm: float,
    travel_bin_mm: float,
    time_blocks: int,
    min_cell_samples: int,
) -> dict[str, float | int | str]:
    """Split error into travel-bin curve, block drift, and within-cell scatter."""

    error = np.asarray(prediction, dtype=float) - data.travel
    travel_bin = np.floor(data.travel / travel_bin_mm).astype(int)
    time_block = np.minimum(
        np.arange(len(error)) * time_blocks // len(error), time_blocks - 1
    )
    region = data.active & (data.travel < region_max_mm)
    valid = np.zeros(len(error), dtype=bool)
    for travel_value in np.unique(travel_bin[region]):
        for time_value in range(time_blocks):
            cell = region & (travel_bin == travel_value) & (time_block == time_value)
            if int(np.sum(cell)) >= min_cell_samples:
                valid |= cell
    if not np.any(valid):
        raise ValueError(f"{data.log}: no populated travel/time cells")

    curve_mean = np.full(len(error), np.nan, dtype=float)
    cell_mean = np.full(len(error), np.nan, dtype=float)
    for travel_value in np.unique(travel_bin[valid]):
        travel_mask = valid & (travel_bin == travel_value)
        curve_mean[travel_mask] = float(np.mean(error[travel_mask]))
        for time_value in np.unique(time_block[travel_mask]):
            cell = travel_mask & (time_block == time_value)
            cell_mean[cell] = float(np.mean(error[cell]))

    selected_error = error[valid]
    selected_curve = curve_mean[valid]
    selected_cell = cell_mean[valid]
    slow_drift = selected_cell - selected_curve
    within_cell = selected_error - selected_cell
    total_mse = float(np.mean(selected_error**2))
    curve_mse = float(np.mean(selected_curve**2))
    slow_mse = float(np.mean(slow_drift**2))
    within_mse = float(np.mean(within_cell**2))
    return {
        "log": data.log,
        "fork": data.fork,
        "centered": centered,
        "region_max_mm": region_max_mm,
        "travel_bin_mm": travel_bin_mm,
        "time_blocks": time_blocks,
        "samples": int(np.sum(valid)),
        "total_rmse_mm": float(np.sqrt(total_mse)),
        "curve_rms_mm": float(np.sqrt(curve_mse)),
        "slow_drift_rms_mm": float(np.sqrt(slow_mse)),
        "within_cell_rms_mm": float(np.sqrt(within_mse)),
        "curve_mse_fraction": curve_mse / total_mse,
        "slow_drift_mse_fraction": slow_mse / total_mse,
        "within_cell_mse_fraction": within_mse / total_mse,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", default=list(DEFAULT_LOGS))
    parser.add_argument(
        "--cache-root", type=Path, default=REPO_ROOT / "backend/run_artifacts"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports/front_mag_nuisance/low_end_error_decomposition",
    )
    parser.add_argument("--bin-mm", type=float, default=5.0)
    parser.add_argument("--min-bin-samples", type=int, default=50)
    parser.add_argument(
        "--region-max-mm", type=float, nargs="+", default=[10.0, 20.0, 30.0, 50.0]
    )
    parser.add_argument("--max-curve-travel-mm", type=float, default=150.0)
    parser.add_argument(
        "--sensitivity-bin-mm", type=float, nargs="+", default=[2.0, 5.0, 10.0]
    )
    parser.add_argument("--time-blocks", type=int, default=5)
    parser.add_argument("--min-time-cell-samples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bin_mm <= 0.0 or args.min_bin_samples < 1:
        raise ValueError("bin-mm and min-bin-samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    bin_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    temporal_rows: list[dict] = []
    data_by_log = [load_log(log, args.cache_root) for log in args.logs]
    for data in data_by_log:
        summary_rows.extend(
            build_summary_rows(
                data, args.region_max_mm, args.bin_mm, args.min_bin_samples
            )
        )
        bin_rows.extend(
            build_bin_rows(
                data,
                args.bin_mm,
                args.min_bin_samples,
                args.max_curve_travel_mm,
            )
        )
        for bin_mm in args.sensitivity_bin_mm:
            values = decompose_prediction(
                data,
                data.mag_prediction,
                30.0,
                bin_mm,
                max(20, int(round(10 * bin_mm))),
            )
            global_offset = float(
                np.mean((data.mag_prediction - data.travel)[data.active])
            )
            centered_values = decompose_prediction(
                data,
                data.mag_prediction - global_offset,
                30.0,
                bin_mm,
                max(20, int(round(10 * bin_mm))),
            )
            sensitivity_rows.append(
                {
                    "log": data.log,
                    "fork": data.fork,
                    "bin_mm": bin_mm,
                    **values,
                    **{
                        f"centered_{key}": value
                        for key, value in centered_values.items()
                    },
                }
            )
        temporal_rows.append(
            temporal_decomposition(
                data,
                data.mag_prediction,
                region_max_mm=30.0,
                travel_bin_mm=args.bin_mm,
                time_blocks=args.time_blocks,
                min_cell_samples=args.min_time_cell_samples,
                centered=False,
            )
        )
        global_offset = float(
            np.mean((data.mag_prediction - data.travel)[data.active])
        )
        temporal_rows.append(
            temporal_decomposition(
                data,
                data.mag_prediction - global_offset,
                region_max_mm=30.0,
                travel_bin_mm=args.bin_mm,
                time_blocks=args.time_blocks,
                min_cell_samples=args.min_time_cell_samples,
                centered=True,
            )
        )

    summary = pd.DataFrame(summary_rows)
    bins = pd.DataFrame(bin_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    temporal = pd.DataFrame(temporal_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    bins.to_csv(args.output_dir / "encoder_binned_curves.csv", index=False)
    sensitivity.to_csv(args.output_dir / "bin_width_sensitivity.csv", index=False)
    temporal.to_csv(args.output_dir / "temporal_decomposition.csv", index=False)

    low = summary[summary["region_max_mm"] == 30.0].sort_values(
        "mag_centered_total_rmse_mm", ascending=False
    )
    columns = [
        "log",
        "fork",
        "mag_global_error_offset_mm",
        "mag_centered_total_rmse_mm",
        "mag_centered_curve_rms_mm",
        "mag_centered_scatter_rms_mm",
        "mag_centered_curve_mse_fraction",
        "mag_centered_oracle_curve_fix_gain_mm",
        "solved_centered_total_rmse_mm",
    ]
    print("Centered low-travel (<30 mm) decomposition:")
    print(low[columns].to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\nResults: {args.output_dir}")


if __name__ == "__main__":
    main()
