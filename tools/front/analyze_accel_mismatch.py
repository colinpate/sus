#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


MS2_PER_MG = 9.81 / 1000.0


@dataclass
class LagStats:
    lag_samples: int
    rmse: float
    corr: float


def derive_gt(cache: np.lib.npyio.NpzFile, use_gradient: bool, use_raw: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    travel_i = cache["travel__x"][:, 0]
    if use_raw:
        t_filt = cache["travel__t"]
        t = cache["accel/proj__t"]
        a_meas = cache["accel/proj__x"][:, 0]
        travel = np.interp(t, t_filt, travel_i)
    else:
        t = cache["accel/lpfhp/proj__t"]
        a_meas = cache["accel/lpfhp/proj__x"][:, 0]
        travel = travel_i
    dt_s = np.diff(t, prepend=t[0] - 0.01)
    if use_gradient:
        v = np.gradient(travel, t, edge_order = 2)
        a_gt = np.gradient(v, t, edge_order = 2) / 1000.0
    else:
        v = np.diff(travel, prepend=travel[0]) / dt_s
        a_gt = np.diff(v, prepend=v[0]) / dt_s / 1000.0
    return t, a_meas, a_gt, v


def sweep_lag(a_meas: np.ndarray, a_gt: np.ndarray, max_lag: int = 6) -> LagStats:
    best: LagStats | None = None
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa = a_meas[-lag:]
            gg = a_gt[: len(aa)]
        elif lag > 0:
            aa = a_meas[:-lag]
            gg = a_gt[lag:]
        else:
            aa = a_meas
            gg = a_gt

        mask = np.abs(gg) > 0.5
        if np.sum(mask) < 100:
            continue

        rmse = float(np.sqrt(np.mean((aa[mask] - gg[mask]) ** 2)))
        corr = float(np.corrcoef(aa[mask], gg[mask])[0, 1])
        stats = LagStats(lag, rmse, corr)
        if best is None or stats.rmse < best.rmse:
            best = stats

    if best is None:
        raise RuntimeError("Could not compute lag stats")
    return best


def align_to_gt_time(cache: np.lib.npyio.NpzFile, values: np.ndarray) -> np.ndarray:
    t_src = cache["accel/proj__t"]
    t_dst = cache["accel/lpfhp/proj__t"]
    if len(t_src) == len(t_dst):
        return values
    return np.interp(t_dst, t_src, values)


def apply_lag(a_meas: np.ndarray, a_gt: np.ndarray, lag_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if lag_samples < 0:
        return a_meas[-lag_samples:], a_gt[: len(a_meas) + lag_samples]
    if lag_samples > 0:
        return a_meas[:-lag_samples], a_gt[lag_samples:]
    return a_meas, a_gt


def error_stats(a_meas: np.ndarray, a_gt: np.ndarray) -> tuple[float, float, float]:
    mask = np.abs(a_gt) > 0.5
    err = a_meas[mask] - a_gt[mask]
    return (
        float(np.sqrt(np.mean(err**2))),
        float(np.mean(np.abs(err))),
        float(np.mean(err)),
    )


def hi_bin_error(a_meas: np.ndarray, a_gt: np.ndarray, pct: float) -> float:
    thresh = np.percentile(a_gt, pct)
    mask = a_gt > thresh
    return float(np.mean(a_meas[mask] - a_gt[mask]))


def signed_lis2_x_ms2(df: pd.DataFrame) -> np.ndarray:
    lis2 = df[["lis2_x", "lis2_y", "lis2_z"]].to_numpy(dtype=float) * MS2_PER_MG
    if np.mean(lis2[:, 0]) > 0:
        lis2 = lis2 @ np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    return lis2[:, 0]


def summarize_log(
    log_name: str,
    logs_dir: Path,
    cache_root: Path,
    cap_mg: float,
    near_cap_ms2: float,
    use_gradient: bool,
    use_raw: bool,
) -> None:
    csv_path = logs_dir / f"{log_name}.csv"
    cache_path = cache_root / log_name / "cache" / "all.npz"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    df = pd.read_csv(csv_path)
    cache = np.load(cache_path)

    t_raw = df["t_s"].to_numpy(dtype=float)
    dt_raw = np.diff(t_raw)
    dseq = np.diff(df["seq"].to_numpy(dtype=int))

    t, a_meas, a_gt, v_gt = derive_gt(cache, use_gradient, use_raw)
    lag0_rmse, lag0_mae, lag0_bias = error_stats(a_meas, a_gt)
    best = sweep_lag(a_meas, a_gt)
    a_best, gt_best = apply_lag(a_meas, a_gt, best.lag_samples)
    best_rmse, best_mae, best_bias = error_stats(a_best, gt_best)

    travel_vec = cache["accel_trav_vec"]
    lis2_proj = cache["accel/lis2_in_lis1__x"] @ travel_vec
    if not use_raw:
        lis2_proj = align_to_gt_time(cache, lis2_proj)

    lis2x = signed_lis2_x_ms2(df)
    lis2x_at_gt = np.interp(t, t_raw, lis2x)
    cap_ms2 = cap_mg * MS2_PER_MG
    raw_cap_mask = lis2x_at_gt <= -cap_ms2
    proj_cap_mask = lis2_proj <= -(cap_ms2 - near_cap_ms2)

    hi95 = a_gt > np.percentile(a_gt, 95)
    hi99 = a_gt > np.percentile(a_gt, 99)

    print(f"\n{log_name}")
    print(
        "  sample rate:"
        f" raw={1.0 / np.median(dt_raw):.1f} Hz"
        f" dt_ms[min/med/max]={dt_raw.min() * 1000:.1f}/{np.median(dt_raw) * 1000:.1f}/{dt_raw.max() * 1000:.1f}"
        f" seq_gaps={np.sum(dseq != 1)}"
    )
    print(
        "  accel-vs-gt:"
        f" lag0 rmse/mae/bias={lag0_rmse:.2f}/{lag0_mae:.2f}/{lag0_bias:.2f}"
        f" best_lag={best.lag_samples} samples"
        f" best rmse/mae/bias={best_rmse:.2f}/{best_mae:.2f}/{best_bias:.2f}"
        f" corr={best.corr:.3f}"
    )
    print(
        "  positive peak error:"
        f" hi95={hi_bin_error(a_meas, a_gt, 95):.2f}"
        f" hi99={hi_bin_error(a_meas, a_gt, 99):.2f}"
        f" hi99@bestlag={hi_bin_error(a_best, gt_best, 99):.2f}"
    )
    print(
        "  lis2 range:"
        f" raw signed-x min/max={lis2x.min() / MS2_PER_MG:.0f}/{lis2x.max() / MS2_PER_MG:.0f} mg"
        f" raw_cap@gt={100.0 * np.mean(raw_cap_mask):.2f}%"
        f" raw_cap@gt_hi99={100.0 * np.mean(raw_cap_mask[hi99]):.2f}%"
    )
    print(
        "  lis2 projected contribution:"
        f" min/max={lis2_proj.min():.1f}/{lis2_proj.max():.1f} m/s^2"
        f" near_cap@gt={100.0 * np.mean(proj_cap_mask):.2f}%"
        f" near_cap@gt_hi99={100.0 * np.mean(proj_cap_mask[hi99]):.2f}%"
    )
    print(
        "  hi99 mean error split:"
        f" near_cap={np.mean((a_meas[hi99 & proj_cap_mask] - a_gt[hi99 & proj_cap_mask])) if np.any(hi99 & proj_cap_mask) else float('nan'):.2f}"
        f" free={np.mean((a_meas[hi99 & ~proj_cap_mask] - a_gt[hi99 & ~proj_cap_mask])):.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose accel/proj vs travel-derived acceleration mismatch")
    parser.add_argument(
        "logs",
        nargs="*",
        default=["log022", "log029", "log030", "log031", "log038", "log056_ccdh", "log078", "log079", "log080", "log085", "log088", "log091"],
        help="Log names without extension",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing log CSV files",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("backend/run_artifacts"),
        help="Root containing pipeline cache folders",
    )
    parser.add_argument(
        "--cap-mg",
        type=float,
        default=15980.0,
        help="Signed lis2 x threshold, in mg, to treat as hitting the accel range limit",
    )
    parser.add_argument(
        "--near-cap-ms2",
        type=float,
        default=10.0,
        help="Projected lis2 margin from the cap, in m/s^2, to count as near saturation",
    )
    parser.add_argument(
        "--use-gradient",
        action="store_true",
        help="Use np.gradient instead of np.diff for getting GT velocity and accel"
    )
    parser.add_argument(
        "--use-raw",
        action="store_true",
        help="Use un-filtered un-decimated data"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for log_name in args.logs:
        summarize_log(
            log_name=log_name,
            logs_dir=args.logs_dir,
            cache_root=args.cache_root,
            cap_mg=args.cap_mg,
            near_cap_ms2=args.near_cap_ms2,
            use_gradient=args.use_gradient,
            use_raw=args.use_raw
        )


if __name__ == "__main__":
    main()
