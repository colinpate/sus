import argparse

import numpy as np

from stats_aggregator import DEFAULT_LOGS

HYPOTENUSE_MM = 120.0


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep zero-angle percentiles and top-link dimensions to find settings that "
            "keep angle-derived travel physically plausible while staying consistent with "
            "projected acceleration."
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
        default=237.5,
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
        default=parse_csv_floats("237.5,238.0,238.5,239.0,239.5"),
        help="Comma-separated candidate total upper-link distances in mm.",
    )
    parser.add_argument(
        "--candidate-zero-percentiles",
        type=parse_csv_floats,
        default=parse_csv_floats("99.0,99.2,99.5,99.7,99.9"),
        help="Comma-separated candidate zero-angle percentiles.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top-ranked candidates to print per log.",
    )
    return parser.parse_args()


def get_accel_mask(ws, raw_accel_max: float = 150.0, proj_accel_min: float = 1.0) -> np.ndarray:
    accel_raw = ws["accel/lis2__x"][:, 0]
    raw_mask = np.abs(accel_raw) < raw_accel_max
    accel_proj = ws["accel/proj__x"][:, 0]
    proj_mask = np.abs(accel_proj) > proj_accel_min
    return raw_mask & proj_mask


def get_travel(angle: np.ndarray, top_adjacent_mm: float, top_zeroangle: float) -> np.ndarray:
    top_angle = np.arccos(top_adjacent_mm / HYPOTENUSE_MM)
    net_angle = -(angle - top_zeroangle) + top_angle
    return 2.0 * (top_adjacent_mm - (HYPOTENUSE_MM * np.cos(net_angle)))


def get_travel_accel(travel_mm: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    vel_mm_s = np.gradient(travel_mm, t_s, edge_order=2)
    return np.gradient(vel_mm_s, t_s, edge_order=2) / 1000.0


def evaluate_candidate(
    *,
    angle: np.ndarray,
    accel_proj_ms2: np.ndarray,
    t_s: np.ndarray,
    accel_mask: np.ndarray,
    top_adjacent_total_mm: float,
    zero_percentile: float,
    max_travel_mm: float,
) -> dict[str, float]:
    top_adjacent_mm = top_adjacent_total_mm / 2.0
    if top_adjacent_mm >= HYPOTENUSE_MM:
        raise ValueError(
            f"Top-adjacent half-length must stay below {HYPOTENUSE_MM:.1f} mm, "
            f"got {top_adjacent_mm:.3f} mm."
        )

    zero_angle = float(np.percentile(angle, zero_percentile))
    travel = get_travel(angle, top_adjacent_mm, zero_angle)
    travel_accel = get_travel_accel(travel, t_s)

    accel_err = travel_accel[accel_mask] - accel_proj_ms2[accel_mask]
    oob = np.maximum(travel - max_travel_mm, 0.0)

    return {
        "top_adjacent_total_mm": top_adjacent_total_mm,
        "top_adjacent_mm": top_adjacent_mm,
        "zero_percentile": zero_percentile,
        "zero_angle_rad": zero_angle,
        "zero_angle_deg": float(np.degrees(zero_angle)),
        "travel_min": float(np.min(travel)),
        "travel_max": float(np.max(travel)),
        "travel_p999": float(np.percentile(travel, 99.9)),
        "travel_mean": float(np.mean(travel)),
        "n_over": int(np.sum(travel > max_travel_mm)),
        "oob_rms": float(np.sqrt(np.mean(oob**2))),
        "accel_rmse": float(np.sqrt(np.mean(accel_err**2))),
        "accel_mean_err": float(np.mean(accel_err)),
    }


def candidate_rank_key(candidate: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        float(candidate["n_over"]),
        candidate["oob_rms"],
        candidate["accel_rmse"],
        abs(candidate["top_adjacent_total_mm"] - 237.5),
    )


def format_candidate(candidate: dict[str, float], max_travel_mm: float) -> str:
    return (
        f"total={candidate['top_adjacent_total_mm']:.1f} mm, "
        f"zero=p{candidate['zero_percentile']:.1f} ({candidate['zero_angle_deg']:.3f} deg), "
        f"max={candidate['travel_max']:.2f} mm, "
        f"p99.9={candidate['travel_p999']:.2f} mm, "
        f"n>{max_travel_mm:.0f}={candidate['n_over']}, "
        f"acc_rmse={candidate['accel_rmse']:.4f}, "
        f"acc_mean_err={candidate['accel_mean_err']:.4f}"
    )


def main() -> None:
    args = parse_args()

    baseline_rows: list[dict[str, float]] = []
    best_rows: list[dict[str, float]] = []

    for log in args.logs:
        ws = np.load(f"backend/run_artifacts/{log}/cache/all.npz")
        angle = ws["angle__x"][:, 0]
        t_s = ws["angle__t"]
        accel_proj_ms2 = ws["accel/proj__x"][:, 0]
        accel_mask = get_accel_mask(ws)

        baseline = evaluate_candidate(
            angle=angle,
            accel_proj_ms2=accel_proj_ms2,
            t_s=t_s,
            accel_mask=accel_mask,
            top_adjacent_total_mm=args.baseline_top_adjacent_total,
            zero_percentile=args.baseline_zero_percentile,
            max_travel_mm=args.max_travel,
        )

        candidates = []
        for top_adjacent_total_mm in args.candidate_top_adjacent_totals:
            for zero_percentile in args.candidate_zero_percentiles:
                candidates.append(
                    evaluate_candidate(
                        angle=angle,
                        accel_proj_ms2=accel_proj_ms2,
                        t_s=t_s,
                        accel_mask=accel_mask,
                        top_adjacent_total_mm=top_adjacent_total_mm,
                        zero_percentile=zero_percentile,
                        max_travel_mm=args.max_travel,
                    )
                )

        ranked = sorted(candidates, key=candidate_rank_key)
        best = ranked[0]

        baseline_rows.append(baseline)
        best_rows.append(best)

        print(log)
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
