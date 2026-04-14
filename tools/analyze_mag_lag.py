#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class LagMetrics:
    lag_ms: float
    count: int
    pearson: float
    rate_corr: float
    poly_rmse: float
    poly_mae: float


def angle_raw_to_travel(angle_raw: np.ndarray) -> np.ndarray:
    angle = angle_raw.astype(float) * (2.0 * np.pi / 4096.0)
    hypotenuse = 120.0
    top_adjacent = 237.5 / 2.0
    top_angle = np.arccos(top_adjacent / hypotenuse)
    top_zeroangle = np.percentile(angle, 99.9)
    net_angle = -1.0 * (angle - top_zeroangle) + top_angle
    travel = 2.0 * (top_adjacent - (hypotenuse * np.cos(net_angle)))
    return travel


def compute_mag_projection(
    mag_xyz: np.ndarray,
    mag_threshold: float = 3000.0,
) -> tuple[np.ndarray, np.ndarray]:
    mag_norm = np.linalg.norm(mag_xyz, axis=1)
    mag_filtered = mag_xyz[mag_norm > mag_threshold]
    if mag_filtered.shape[0] == 0:
        raise RuntimeError(f"No magnetometer samples above threshold {mag_threshold}")
    mean_vector = np.mean(mag_filtered, axis=0)
    mean_norm = np.linalg.norm(mean_vector)
    if mean_norm < 1e-12:
        raise RuntimeError("Could not determine a stable magnetometer projection vector")
    mag_travel_vector = mean_vector / mean_norm
    mag_proj = mag_xyz @ mag_travel_vector
    return mag_proj, mag_travel_vector


def correct_bad_mag_proj(
    mag_xyz: np.ndarray,
    mag_proj: np.ndarray,
    raw_norm_maxdiff: float = 2000.0,
    min_corr_mg: float = 5000.0,
) -> tuple[np.ndarray, np.ndarray]:
    mag_norm = np.linalg.norm(mag_xyz, axis=1)
    bad_mask = np.abs(mag_norm - mag_proj) > raw_norm_maxdiff
    bad_mask &= mag_norm > min_corr_mg
    corrected = mag_proj.copy()
    corrected[bad_mask] = mag_norm[bad_mask]
    return corrected, bad_mask.astype(float)


def build_csv_raw_series(
    csv_path: Path,
    mag_source: str,
    signal_kind: str,
    mag_threshold: float,
    raw_norm_maxdiff: float,
    min_corr_mg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    df = pd.read_csv(csv_path)
    t = df["t_s"].to_numpy(dtype=float)
    travel = angle_raw_to_travel(df["angle_raw"].to_numpy(dtype=float))

    if mag_source == "mmc":
        cols = ["mmc_mG_x", "mmc_mG_y", "mmc_mG_z"]
    else:
        cols = ["lis3mdl_mG_x", "lis3mdl_mG_y", "lis3mdl_mG_z"]

    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"CSV is missing columns required for mag source '{mag_source}': {missing}")

    mag_xyz = df[cols].to_numpy(dtype=float)
    mag_norm = np.linalg.norm(mag_xyz, axis=1)
    mag_proj, _ = compute_mag_projection(mag_xyz, mag_threshold=mag_threshold)
    mag_proj_corr, bad_mask = correct_bad_mag_proj(
        mag_xyz,
        mag_proj,
        raw_norm_maxdiff=raw_norm_maxdiff,
        min_corr_mg=min_corr_mg,
    )

    if signal_kind == "norm":
        signal = mag_norm
        mask = None
    elif signal_kind == "proj":
        signal = mag_proj
        mask = None
    elif signal_kind == "proj_corr":
        signal = mag_proj_corr
        mask = bad_mask
    else:
        raise ValueError(f"Unsupported csv signal kind '{signal_kind}'")

    return t, signal, travel, mask


def load_series(cache: np.lib.npyio.NpzFile, key: str) -> tuple[np.ndarray, np.ndarray]:
    t_key = f"{key}__t"
    x_key = f"{key}__x"
    if t_key not in cache or x_key not in cache:
        raise KeyError(f"Missing time series key '{key}' in cache")

    t = cache[t_key]
    x = cache[x_key]
    if x.ndim > 1 and x.shape[1] == 1:
        x = x[:, 0]
    elif x.ndim > 1:
        raise ValueError(f"Key '{key}' is not 1D; got shape {x.shape}")
    return t.astype(float), x.astype(float)


def shift_signal_to_target(
    signal_t: np.ndarray,
    signal_x: np.ndarray,
    target_t: np.ndarray,
    lag_s: float,
) -> np.ndarray:
    # Positive lag means "delay the signal", so at time t we sample the source at t - lag.
    sample_t = target_t - lag_s
    return np.interp(sample_t, signal_t, signal_x, left=np.nan, right=np.nan)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std < 1e-12 or y_std < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def polyfit_errors(x: np.ndarray, y: np.ndarray, degree: int) -> tuple[float, float]:
    if x.size < degree + 2:
        return float("nan"), float("nan")

    unique_x = np.unique(x)
    fit_degree = min(degree, unique_x.size - 1)
    if fit_degree < 1:
        return float("nan"), float("nan")

    coeffs = np.polyfit(x, y, deg=fit_degree)
    y_hat = np.polyval(coeffs, x)
    err = y_hat - y
    return float(np.sqrt(np.mean(err**2))), float(np.mean(np.abs(err)))


def build_mask(
    target_t: np.ndarray,
    target_x: np.ndarray,
    signal_x: np.ndarray,
    mask_t: np.ndarray | None,
    mask_x: np.ndarray | None,
    motion_threshold: float,
) -> np.ndarray:
    mask = np.isfinite(target_x) & np.isfinite(signal_x)

    if mask_t is not None and mask_x is not None:
        interp_mask = np.interp(target_t, mask_t, mask_x, left=1.0, right=1.0)
        mask &= interp_mask < 0.5

    if motion_threshold > 0:
        travel_v = np.gradient(target_x, target_t, edge_order=2)
        mask &= np.abs(travel_v) >= motion_threshold

    return mask


def score_lag(
    signal_t: np.ndarray,
    signal_x: np.ndarray,
    target_t: np.ndarray,
    target_x: np.ndarray,
    lag_s: float,
    poly_degree: int,
    mask_t: np.ndarray | None,
    mask_x: np.ndarray | None,
    motion_threshold: float,
) -> LagMetrics:
    shifted = shift_signal_to_target(signal_t, signal_x, target_t, lag_s)
    mask = build_mask(target_t, target_x, shifted, mask_t, mask_x, motion_threshold)

    x = shifted[mask]
    y = target_x[mask]
    count = int(np.sum(mask))

    pearson = safe_corr(x, y)

    if count >= 5:
        dx = np.gradient(x, target_t[mask], edge_order=2)
        dy = np.gradient(y, target_t[mask], edge_order=2)
        rate_corr = safe_corr(dx, dy)
    else:
        rate_corr = float("nan")

    poly_rmse, poly_mae = polyfit_errors(x, y, poly_degree)
    return LagMetrics(
        lag_ms=lag_s * 1000.0,
        count=count,
        pearson=pearson,
        rate_corr=rate_corr,
        poly_rmse=poly_rmse,
        poly_mae=poly_mae,
    )


def choose_best(
    results: list[LagMetrics],
    metric: str,
    maximize_abs: bool = False,
) -> LagMetrics:
    valid = [r for r in results if np.isfinite(getattr(r, metric))]
    if not valid:
        raise RuntimeError(f"No valid lag results for metric '{metric}'")

    if maximize_abs:
        return max(valid, key=lambda r: abs(getattr(r, metric)))
    return min(valid, key=lambda r: getattr(r, metric))


def summarize_log(
    log_name: str,
    cache_root: Path,
    logs_dir: Path,
    mode: str,
    signal_key: str,
    target_key: str,
    mask_key: str | None,
    max_lag_ms: float,
    step_ms: float | None,
    poly_degree: int,
    motion_threshold: float,
    top_k: int,
    mag_source: str,
    csv_signal_kind: str,
    mag_threshold: float,
    raw_norm_maxdiff: float,
    min_corr_mg: float,
) -> None:
    if mode == "cache":
        cache_path = cache_root / log_name / "cache" / "all.npz"
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)

        cache = np.load(cache_path)
        signal_t, signal_x = load_series(cache, signal_key)
        target_t, target_x = load_series(cache, target_key)

        if mask_key:
            mask_t, mask_x = load_series(cache, mask_key)
        else:
            mask_t, mask_x = None, None
    else:
        csv_path = logs_dir / f"{log_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        target_t, signal_x, target_x, mask_x = build_csv_raw_series(
            csv_path=csv_path,
            mag_source=mag_source,
            signal_kind=csv_signal_kind,
            mag_threshold=mag_threshold,
            raw_norm_maxdiff=raw_norm_maxdiff,
            min_corr_mg=min_corr_mg,
        )
        signal_t = target_t
        mask_t = target_t if mask_x is not None else None

    target_dt_ms = float(np.median(np.diff(target_t)) * 1000.0)
    signal_dt_ms = float(np.median(np.diff(signal_t)) * 1000.0)
    if step_ms is None:
        step_ms = min(target_dt_ms, signal_dt_ms)

    lag_grid_ms = np.arange(-max_lag_ms, max_lag_ms + 0.5 * step_ms, step_ms)
    results = [
        score_lag(
            signal_t=signal_t,
            signal_x=signal_x,
            target_t=target_t,
            target_x=target_x,
            lag_s=lag_ms / 1000.0,
            poly_degree=poly_degree,
            mask_t=mask_t,
            mask_x=mask_x,
            motion_threshold=motion_threshold,
        )
        for lag_ms in lag_grid_ms
    ]

    best_rate = choose_best(results, "rate_corr", maximize_abs=True)
    best_pearson = choose_best(results, "pearson", maximize_abs=True)
    best_poly = choose_best(results, "poly_rmse", maximize_abs=False)

    poly_sorted = sorted(
        [r for r in results if np.isfinite(r.poly_rmse)],
        key=lambda r: r.poly_rmse,
    )[:top_k]

    print(f"\n{log_name}")
    print(
        "  series:"
        f" mode={mode}"
        f" signal={(signal_key if mode == 'cache' else f'{mag_source}/{csv_signal_kind}')}"
        f" ({signal_x.shape[0]} pts @ {signal_dt_ms:.1f} ms)"
        f" target={(target_key if mode == 'cache' else 'travel_from_angle_raw')}"
        f" ({target_x.shape[0]} pts @ {target_dt_ms:.1f} ms)"
    )
    if mode == "cache" and mask_key:
        print(f"  mask: {mask_key}")
    elif mode == "csv_raw" and mask_x is not None:
        print("  mask: derived raw mag bad-data mask")
    print(
        "  sweep:"
        f" lag range=[{-max_lag_ms:.1f}, {max_lag_ms:.1f}] ms"
        f" step={step_ms:.1f} ms"
        f" motion_threshold={motion_threshold:.3f}"
    )
    print(
        "  best |rate_corr|:"
        f" lag={best_rate.lag_ms:.1f} ms"
        f" rate_corr={best_rate.rate_corr:.4f}"
        f" pearson={best_rate.pearson:.4f}"
        f" poly_rmse={best_rate.poly_rmse:.2f}"
        f" n={best_rate.count}"
    )
    print(
        "  best |pearson|:"
        f" lag={best_pearson.lag_ms:.1f} ms"
        f" pearson={best_pearson.pearson:.4f}"
        f" rate_corr={best_pearson.rate_corr:.4f}"
        f" poly_rmse={best_pearson.poly_rmse:.2f}"
        f" n={best_pearson.count}"
    )
    print(
        "  best poly_rmse:"
        f" lag={best_poly.lag_ms:.1f} ms"
        f" poly_rmse={best_poly.poly_rmse:.2f}"
        f" poly_mae={best_poly.poly_mae:.2f}"
        f" pearson={best_poly.pearson:.4f}"
        f" rate_corr={best_poly.rate_corr:.4f}"
        f" n={best_poly.count}"
    )
    print("  top poly_rmse lags:")
    for r in poly_sorted:
        print(
            f"    lag={r.lag_ms:6.1f} ms"
            f" rmse={r.poly_rmse:8.2f}"
            f" mae={r.poly_mae:7.2f}"
            f" pearson={r.pearson:7.4f}"
            f" rate_corr={r.rate_corr:7.4f}"
            f" n={r.count}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep time shifts to find the best lag between magnetometer-derived data and travel"
    )
    parser.add_argument(
        "logs",
        nargs="*",
        default=["log079"],
        help="Log names without extension",
    )
    parser.add_argument(
        "--mode",
        choices=("cache", "csv_raw"),
        default="cache",
        help="Use cached workspace signals or reconstruct raw signals from the CSV",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("backend/run_artifacts"),
        help="Root containing pipeline cache folders",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing raw CSV logs",
    )
    parser.add_argument(
        "--signal-key",
        default="mag/proj/corr/lpf",
        help="Workspace key for the signal to shift",
    )
    parser.add_argument(
        "--target-key",
        default="travel",
        help="Workspace key for the target signal",
    )
    parser.add_argument(
        "--mask-key",
        default="mag/proj/bad_mask",
        help="Optional workspace key for a 0/1 bad-data mask; pass '' to disable",
    )
    parser.add_argument(
        "--max-lag-ms",
        type=float,
        default=100.0,
        help="Maximum absolute lag to test, in milliseconds",
    )
    parser.add_argument(
        "--step-ms",
        type=float,
        default=None,
        help="Lag step size in milliseconds; defaults to the faster series dt",
    )
    parser.add_argument(
        "--poly-degree",
        type=int,
        default=3,
        help="Polynomial degree for fitting target from shifted signal",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.0,
        help="Optional |d(target)/dt| threshold for keeping only moving samples",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many of the best poly-rmse lags to print",
    )
    parser.add_argument(
        "--mag-source",
        choices=("mmc", "lis3mdl"),
        default="mmc",
        help="Magnetometer source to use in csv_raw mode",
    )
    parser.add_argument(
        "--csv-signal-kind",
        choices=("norm", "proj", "proj_corr"),
        default="proj_corr",
        help="Which raw magnetometer signal to compare in csv_raw mode",
    )
    parser.add_argument(
        "--mag-threshold",
        type=float,
        default=3000.0,
        help="Projection-vector threshold for csv_raw mode",
    )
    parser.add_argument(
        "--raw-norm-maxdiff",
        type=float,
        default=2000.0,
        help="Bad projected-mag threshold for csv_raw mode",
    )
    parser.add_argument(
        "--min-corr-mg",
        type=float,
        default=5000.0,
        help="Minimum raw norm for projected-mag correction in csv_raw mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_key = args.mask_key or None
    for log_name in args.logs:
        summarize_log(
            log_name=log_name,
            cache_root=args.cache_root,
            logs_dir=args.logs_dir,
            mode=args.mode,
            signal_key=args.signal_key,
            target_key=args.target_key,
            mask_key=mask_key,
            max_lag_ms=args.max_lag_ms,
            step_ms=args.step_ms,
            poly_degree=args.poly_degree,
            motion_threshold=args.motion_threshold,
            top_k=args.top_k,
            mag_source=args.mag_source,
            csv_signal_kind=args.csv_signal_kind,
            mag_threshold=args.mag_threshold,
            raw_norm_maxdiff=args.raw_norm_maxdiff,
            min_corr_mg=args.min_corr_mg,
        )


if __name__ == "__main__":
    main()
