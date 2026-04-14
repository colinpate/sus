import argparse

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from backend.angle_corruption import (
    ANGLE_ERROR_HALO_S,
    find_corrupt_angle_samples,
    interpolate_masked_signal,
    project_mask_to_timeline,
)
from stats_aggregator import DEFAULT_LOGS

HYPOTENUSE_MM = 125.0
DEC_FREQ_HZ = 100.0


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep zero-angle percentiles and top-link dimensions using a cleaned angle "
            "trace so AS5600 rail glitches do not dominate the travel-vs-accel error."
        )
    )
    parser.add_argument(
        "logs",
        nargs="*",
        default=DEFAULT_LOGS,
        help="Log names to analyze. Defaults to the logs used by refine_mag_proj.py.",
    )
    parser.add_argument(
        "--max-travel",
        type=float,
        default=170.0,
        help="Physical travel limit in mm.",
    )
    parser.add_argument(
        "--baseline-top-adjacent-total",
        type=float,
        default=242.0,
        help="Current measured distance between the upper linkage points, in mm.",
    )
    parser.add_argument(
        "--baseline-zero-percentile",
        type=float,
        default=99.9,
        help="Current zero-angle percentile used by the pipeline.",
    )
    parser.add_argument(
        "--candidate-top-adjacent-totals",
        type=parse_csv_floats,
        default=parse_csv_floats("240,241,242,243,244"),
        help="Comma-separated candidate total upper-link distances in mm.",
    )
    parser.add_argument(
        "--candidate-zero-percentiles",
        type=parse_csv_floats,
        default=parse_csv_floats("99.0,99.2,99.5,99.7,99.9"),
        help="Comma-separated candidate zero-angle percentiles.",
    )
    parser.add_argument(
        "--accel-threshold",
        type=float,
        default=0.5,
        help="Only score accel error where |angle-derived accel| exceeds this threshold in m/s^2.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top-ranked candidates to print per log.",
    )
    return parser.parse_args()


def get_travel(angle: np.ndarray, top_adjacent_mm: float, top_zeroangle: float) -> np.ndarray:
    top_angle = np.arccos(top_adjacent_mm / HYPOTENUSE_MM)
    net_angle = -(angle - top_zeroangle) + top_angle
    return 2.0 * (top_adjacent_mm - (HYPOTENUSE_MM * np.cos(net_angle)))


def get_travel_accel(travel_mm: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    vel_mm_s = np.gradient(travel_mm, t_s, edge_order=2)
    return np.gradient(vel_mm_s, t_s, edge_order=2) / 1000.0


def load_clean_angle(log: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    df = pd.read_csv(f"logs/{log}.csv", usecols=["t_s", "angle_raw"])
    t_raw = df["t_s"].to_numpy(dtype=float)
    angle_raw = df["angle_raw"].to_numpy()

    bad_raw_mask = find_corrupt_angle_samples(angle_raw)
    angle_raw_rad = angle_raw * np.pi * 2 / 4096
    angle_clean = interpolate_masked_signal(angle_raw_rad, bad_raw_mask, sample_pos=t_raw)

    fs_hz = 1 / np.median(np.diff(t_raw))
    sos = butter(N=4, Wn=20.0, btype="low", fs=fs_hz, output="sos")
    angle_lpf = sosfiltfilt(sos, angle_clean)

    dec_factor = max(1, round(fs_hz / DEC_FREQ_HZ))
    angle_dec = angle_lpf[::dec_factor]
    t_dec = t_raw[::dec_factor]
    bad_eval_mask = project_mask_to_timeline(t_raw, bad_raw_mask, t_dec, halo_s=ANGLE_ERROR_HALO_S)

    return angle_dec, t_dec, bad_eval_mask, float(np.mean(bad_raw_mask) * 100.0)


def load_log_inputs(log: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    ws = np.load(f"backend/run_artifacts/{log}/cache/all.npz")
    angle, t_s, bad_eval_mask, bad_raw_pct = load_clean_angle(log)
    accel_proj_ms2 = ws["accel/lpf/proj__x"][:, 0]

    if angle.shape != accel_proj_ms2.shape or angle.shape != t_s.shape:
        raise ValueError(
            f"{log}: cleaned angle, travel time, and accel projection must align, got "
            f"{angle.shape}, {t_s.shape}, {accel_proj_ms2.shape}"
        )

    return angle, accel_proj_ms2, t_s, bad_eval_mask, bad_raw_pct


def evaluate_candidate(
    *,
    angle: np.ndarray,
    accel_proj_ms2: np.ndarray,
    t_s: np.ndarray,
    valid_mask: np.ndarray,
    top_adjacent_total_mm: float,
    zero_percentile: float,
    max_travel_mm: float,
    accel_threshold: float,
) -> dict[str, float]:
    top_adjacent_mm = top_adjacent_total_mm / 2.0
    if top_adjacent_mm >= HYPOTENUSE_MM:
        raise ValueError(
            f"Top-adjacent half-length must stay below {HYPOTENUSE_MM:.1f} mm, "
            f"got {top_adjacent_mm:.3f} mm."
        )

    zero_angle = float(np.percentile(angle[valid_mask], zero_percentile))
    travel = get_travel(angle, top_adjacent_mm, zero_angle)
    travel_accel = get_travel_accel(travel, t_s)

    accel_mask = valid_mask & np.isfinite(travel_accel) & np.isfinite(accel_proj_ms2) & (np.abs(travel_accel) > accel_threshold)
    if not np.any(accel_mask):
        raise ValueError("No valid accel samples remain after masking")

    accel_err = travel_accel[accel_mask] - accel_proj_ms2[accel_mask]
    oob = np.maximum(travel[valid_mask] - max_travel_mm, 0.0)

    return {
        "top_adjacent_total_mm": top_adjacent_total_mm,
        "top_adjacent_mm": top_adjacent_mm,
        "zero_percentile": zero_percentile,
        "zero_angle_rad": zero_angle,
        "zero_angle_deg": float(np.degrees(zero_angle)),
        "travel_min": float(np.min(travel[valid_mask])),
        "travel_max": float(np.max(travel[valid_mask])),
        "travel_p999": float(np.percentile(travel[valid_mask], 99.9)),
        "travel_mean": float(np.mean(travel[valid_mask])),
        "n_over": int(np.sum(travel[valid_mask] > max_travel_mm)),
        "oob_rms": float(np.sqrt(np.mean(oob**2))),
        "accel_rmse": float(np.sqrt(np.mean(accel_err**2))),
        "accel_mean_err": float(np.mean(accel_err)),
        "n_eval": int(np.sum(accel_mask)),
    }


def candidate_rank_key(candidate: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        float(candidate["n_over"]),
        candidate["oob_rms"],
        candidate["accel_rmse"],
        abs(candidate["top_adjacent_total_mm"] - 242.0),
    )


def format_candidate(candidate: dict[str, float], max_travel_mm: float) -> str:
    return (
        f"total={candidate['top_adjacent_total_mm']:.1f} mm, "
        f"zero=p{candidate['zero_percentile']:.1f} ({candidate['zero_angle_deg']:.3f} deg), "
        f"max={candidate['travel_max']:.2f} mm, "
        f"p99.9={candidate['travel_p999']:.2f} mm, "
        f"n>{max_travel_mm:.0f}={candidate['n_over']}, "
        f"acc_rmse={candidate['accel_rmse']:.4f}, "
        f"acc_mean_err={candidate['accel_mean_err']:.4f}, "
        f"n_eval={candidate['n_eval']}"
    )


def main() -> None:
    args = parse_args()

    baseline_rows: list[dict[str, float]] = []
    best_rows: list[dict[str, float]] = []

    for log in args.logs:
        angle, accel_proj_ms2, t_s, bad_eval_mask, bad_raw_pct = load_log_inputs(log)
        valid_mask = ~bad_eval_mask

        baseline = evaluate_candidate(
            angle=angle,
            accel_proj_ms2=accel_proj_ms2,
            t_s=t_s,
            valid_mask=valid_mask,
            top_adjacent_total_mm=args.baseline_top_adjacent_total,
            zero_percentile=args.baseline_zero_percentile,
            max_travel_mm=args.max_travel,
            accel_threshold=args.accel_threshold,
        )

        candidates = []
        for top_adjacent_total_mm in args.candidate_top_adjacent_totals:
            for zero_percentile in args.candidate_zero_percentiles:
                candidates.append(
                    evaluate_candidate(
                        angle=angle,
                        accel_proj_ms2=accel_proj_ms2,
                        t_s=t_s,
                        valid_mask=valid_mask,
                        top_adjacent_total_mm=top_adjacent_total_mm,
                        zero_percentile=zero_percentile,
                        max_travel_mm=args.max_travel,
                        accel_threshold=args.accel_threshold,
                    )
                )

        ranked = sorted(candidates, key=candidate_rank_key)
        best = ranked[0]

        baseline_rows.append(baseline)
        best_rows.append(best)

        print(log)
        print(
            f"  raw angle corruption: {bad_raw_pct:.2f}% after raw padding, "
            f"{np.mean(bad_eval_mask) * 100.0:.2f}% excluded on the 100 Hz travel timeline"
        )
        print("  baseline:", format_candidate(baseline, args.max_travel))
        print("  best candidate:", format_candidate(best, args.max_travel))
        print("  top candidates:")
        for candidate in ranked[: args.top_k]:
            print("   ", format_candidate(candidate, args.max_travel))
        print()

    baseline_over = int(sum(row["n_over"] for row in baseline_rows))
    best_over = int(sum(row["n_over"] for row in best_rows))
    baseline_rmse = float(np.mean([row["accel_rmse"] for row in baseline_rows]))
    best_rmse = float(np.mean([row["accel_rmse"] for row in best_rows]))
    baseline_p999 = float(np.mean([row["travel_p999"] for row in baseline_rows]))
    best_p999 = float(np.mean([row["travel_p999"] for row in best_rows]))

    print("Aggregate summary")
    print(
        f"  baseline: total_over={baseline_over}, mean_acc_rmse={baseline_rmse:.4f}, "
        f"mean_p99.9={baseline_p999:.2f} mm"
    )
    print(
        f"  best-per-log: total_over={best_over}, mean_acc_rmse={best_rmse:.4f}, "
        f"mean_p99.9={best_p999:.2f} mm"
    )


if __name__ == "__main__":
    main()
