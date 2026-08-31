#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.classes.log_config import (
    EMPTY_LOG_CONFIG_HASH,
    attach_log_config,
    get_log_config_hash,
    get_step_config,
    load_log_config,
)
from tools.stats_aggregator import DEFAULT_LOGS, project_mask_to_timeline


PIPELINE_DEFAULT_HYPOTENUSE_MM = 120.0
PIPELINE_DEFAULT_TOP_ADJACENT_MM = 239.0 / 2.0
PIPELINE_DEFAULT_ZERO_PERCENTILE = 99.5
PIPELINE_DEFAULT_ANGLE_SIGN = 1.0


@dataclass(frozen=True)
class AngleGeometry:
    hypotenuse_mm: float
    top_adjacent_total_mm: float
    zero_percentile: float
    angle_sign: float
    top_zeroangle_rad: float | None = None


@dataclass(frozen=True)
class LogInputs:
    angle_lpf: np.ndarray
    cached_travel_mm: np.ndarray
    accel_proj_ms2: np.ndarray
    t_s: np.ndarray
    valid_mask: np.ndarray
    bad_raw_pct: float
    geometry: AngleGeometry
    cache_matches_config: bool


def parse_csv_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated number")
    return values


def percentile(value: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 100.0:
        raise argparse.ArgumentTypeError("percentile must be between 0 and 100")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep angle-to-travel geometry and zero-angle settings against the same "
            "filtered angle and projected acceleration used by the front pipeline."
        )
    )
    parser.add_argument(
        "logs",
        nargs="*",
        default=DEFAULT_LOGS,
        help="Log names to analyze. Defaults to the logs used by refine_mag_proj.py.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing log CSV and .meta.json files.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("backend/run_artifacts"),
        help="Root containing pipeline cache folders.",
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
        default=None,
        help="Baseline total upper-link distance in mm. Defaults to each log's pipeline config.",
    )
    zero_group = parser.add_mutually_exclusive_group()
    zero_group.add_argument(
        "--baseline-zero-percentile",
        type=percentile,
        default=None,
        help="Baseline zero-angle percentile. Defaults to each log's pipeline config.",
    )
    zero_group.add_argument(
        "--baseline-zero-angle-rad",
        type=float,
        default=None,
        help="Use this fixed zero angle in radians for the baseline instead of a percentile.",
    )
    parser.add_argument(
        "--hypotenuse",
        type=float,
        default=None,
        help="Override the pipeline-configured linkage hypotenuse for all evaluated candidates.",
    )
    parser.add_argument(
        "--angle-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=None,
        help="Override the pipeline-configured angle sign for all evaluated candidates.",
    )
    parser.add_argument(
        "--candidate-top-adjacent-totals",
        type=parse_csv_floats,
        default=None,
        help=(
            "Comma-separated candidate total upper-link distances in mm. By default, sweep "
            "the effective baseline at offsets -2, -1, 0, +1, and +2 mm."
        ),
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
        help="Only score samples where |angle-derived accel| exceeds this threshold in m/s^2.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top-ranked candidates to print per log and in the shared summary.",
    )
    parser.add_argument(
        "--allow-stale-cache",
        action="store_true",
        help="Analyze even when a cache's recorded log-config hash differs from the current config.",
    )
    args = parser.parse_args()

    if args.max_travel <= 0:
        parser.error("--max-travel must be positive")
    if args.hypotenuse is not None and args.hypotenuse <= 0:
        parser.error("--hypotenuse must be positive")
    if args.accel_threshold < 0:
        parser.error("--accel-threshold cannot be negative")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    for value in args.candidate_zero_percentiles:
        if not 0.0 <= value <= 100.0:
            parser.error("--candidate-zero-percentiles values must be between 0 and 100")
    return args


def geometry_from_config(config: dict[str, Any]) -> AngleGeometry:
    ws: dict[str, Any] = {}
    attach_log_config(ws, config)
    step_config = get_step_config(ws, "angle_to_travel", "AngleToTravel")
    return AngleGeometry(
        hypotenuse_mm=float(step_config.get("hypotenuse", PIPELINE_DEFAULT_HYPOTENUSE_MM)),
        top_adjacent_total_mm=2.0
        * float(step_config.get("top_adjacent", PIPELINE_DEFAULT_TOP_ADJACENT_MM)),
        zero_percentile=float(
            step_config.get("top_zeroangle_percentile", PIPELINE_DEFAULT_ZERO_PERCENTILE)
        ),
        angle_sign=float(step_config.get("angle_sign", PIPELINE_DEFAULT_ANGLE_SIGN)),
        top_zeroangle_rad=(
            float(step_config["top_zeroangle"])
            if step_config.get("top_zeroangle") is not None
            else None
        ),
    )


def load_log_inputs(
    log: str,
    *,
    logs_dir: Path,
    cache_root: Path,
    allow_stale_cache: bool,
) -> LogInputs:
    log_path = logs_dir / f"{log}.csv"
    cache_path = cache_root / log / "cache" / "all.npz"
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} does not exist; run backend/pipeline.py {log} first"
        )

    config = load_log_config(log_path)
    geometry = geometry_from_config(config)
    current_config_hash = get_log_config_hash(config)

    with np.load(cache_path, allow_pickle=False) as ws:
        required_keys = (
            "angle/lpf__x",
            "angle/lpf__t",
            "angle/bad_mask__x",
            "angle/bad_mask__t",
            "accel/lpf/proj__x",
            "accel/lpf/proj__t",
            "travel__x",
            "travel__t",
        )
        missing = [key for key in required_keys if key not in ws]
        if missing:
            raise KeyError(f"{cache_path} is missing required arrays: {', '.join(missing)}")

        cached_hash = str(ws["__log_config_hash"].item()) if "__log_config_hash" in ws else None
        cache_matches_config = cached_hash == current_config_hash or (
            cached_hash is None and current_config_hash == EMPTY_LOG_CONFIG_HASH
        )
        if not cache_matches_config and not allow_stale_cache:
            raise ValueError(
                f"{log}: pipeline cache was produced with a different log config; rerun "
                f"backend/pipeline.py {log}, or pass --allow-stale-cache to inspect it anyway"
            )

        angle = np.asarray(ws["angle/lpf__x"][:, 0], dtype=float)
        angle_t = np.asarray(ws["angle/lpf__t"], dtype=float)
        accel = np.asarray(ws["accel/lpf/proj__x"][:, 0], dtype=float)
        accel_t = np.asarray(ws["accel/lpf/proj__t"], dtype=float)
        cached_travel = np.asarray(ws["travel__x"][:, 0], dtype=float)
        travel_t = np.asarray(ws["travel__t"], dtype=float)
        bad_raw_mask = np.asarray(ws["angle/bad_mask__x"][:, 0], dtype=bool)
        bad_raw_t = np.asarray(ws["angle/bad_mask__t"], dtype=float)

    if angle.shape != accel.shape or angle.shape != cached_travel.shape or angle.shape != angle_t.shape:
        raise ValueError(
            f"{log}: cached angle, travel, time, and accel projection must align, got "
            f"{angle.shape}, {cached_travel.shape}, {angle_t.shape}, {accel.shape}"
        )
    if not (
        np.allclose(angle_t, accel_t, rtol=0.0, atol=1e-9)
        and np.allclose(angle_t, travel_t, rtol=0.0, atol=1e-9)
    ):
        raise ValueError(f"{log}: cached angle, travel, and accel timelines do not align")

    bad_eval_mask = project_mask_to_timeline(bad_raw_t, bad_raw_mask, angle_t)
    return LogInputs(
        angle_lpf=angle,
        cached_travel_mm=cached_travel,
        accel_proj_ms2=accel,
        t_s=angle_t,
        valid_mask=~bad_eval_mask,
        bad_raw_pct=float(np.mean(bad_raw_mask) * 100.0),
        geometry=geometry,
        cache_matches_config=cache_matches_config,
    )


def get_travel(
    angle: np.ndarray,
    *,
    hypotenuse_mm: float,
    top_adjacent_mm: float,
    top_zeroangle: float,
) -> np.ndarray:
    top_angle = np.arccos(top_adjacent_mm / hypotenuse_mm)
    net_angle = -(angle - top_zeroangle) + top_angle
    return 2.0 * (top_adjacent_mm - (hypotenuse_mm * np.cos(net_angle)))


def get_travel_accel(travel_mm: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    vel_mm_s = np.gradient(travel_mm, t_s, edge_order=2)
    return np.gradient(vel_mm_s, t_s, edge_order=2) / 1000.0


def evaluate_candidate(
    *,
    angle_lpf: np.ndarray,
    accel_proj_ms2: np.ndarray,
    t_s: np.ndarray,
    valid_mask: np.ndarray,
    hypotenuse_mm: float,
    angle_sign: float,
    top_adjacent_total_mm: float,
    zero_percentile: float,
    max_travel_mm: float,
    accel_threshold: float,
    zero_angle_rad: float | None = None,
) -> dict[str, float]:
    if angle_sign not in {-1.0, 1.0}:
        raise ValueError(f"angle_sign must be -1 or +1, got {angle_sign}")
    top_adjacent_mm = top_adjacent_total_mm / 2.0
    if top_adjacent_mm >= hypotenuse_mm:
        raise ValueError(
            f"Top-adjacent half-length must stay below the {hypotenuse_mm:.3f} mm "
            f"hypotenuse, got {top_adjacent_mm:.3f} mm."
        )
    if top_adjacent_mm <= -hypotenuse_mm:
        raise ValueError(
            f"Top-adjacent half-length must exceed {-hypotenuse_mm:.3f} mm, "
            f"got {top_adjacent_mm:.3f} mm."
        )

    angle = angle_lpf * angle_sign
    # AngleToTravel computes its reference from the full interpolated/filtered signal;
    # the corruption mask only controls scoring in GetAccelError.
    zero_angle = (
        float(zero_angle_rad)
        if zero_angle_rad is not None
        else float(np.percentile(angle, zero_percentile))
    )
    travel = get_travel(
        angle,
        hypotenuse_mm=hypotenuse_mm,
        top_adjacent_mm=top_adjacent_mm,
        top_zeroangle=zero_angle,
    )
    travel_accel = get_travel_accel(travel, t_s)

    accel_mask = (
        valid_mask
        & np.isfinite(travel_accel)
        & np.isfinite(accel_proj_ms2)
        & (np.abs(travel_accel) > accel_threshold)
    )
    if not np.any(accel_mask):
        raise ValueError("No valid accel samples remain after masking")

    # Match GetAccelError in the pipeline so the reported bias has the same sign.
    accel_err = accel_proj_ms2[accel_mask] - travel_accel[accel_mask]
    accel_eval = accel_proj_ms2[accel_mask]
    travel_accel_eval = travel_accel[accel_mask]
    valid_travel = travel[valid_mask]
    oob = np.maximum(valid_travel - max_travel_mm, 0.0)
    accel_corr = float(np.corrcoef(accel_eval, travel_accel_eval)[0, 1])

    return {
        "hypotenuse_mm": hypotenuse_mm,
        "angle_sign": angle_sign,
        "top_adjacent_total_mm": top_adjacent_total_mm,
        "top_adjacent_mm": top_adjacent_mm,
        "zero_percentile": zero_percentile if zero_angle_rad is None else float("nan"),
        "zero_angle_rad": zero_angle,
        "zero_angle_deg": float(np.degrees(zero_angle)),
        "travel_min": float(np.min(valid_travel)),
        "travel_max": float(np.max(valid_travel)),
        "travel_p999": float(np.percentile(valid_travel, 99.9)),
        "travel_mean": float(np.mean(valid_travel)),
        "n_over": int(np.sum(valid_travel > max_travel_mm)),
        "oob_rms": float(np.sqrt(np.mean(oob**2))),
        "oob_sse": float(np.sum(oob**2)),
        "n_valid": int(np.sum(valid_mask)),
        "accel_rmse": float(np.sqrt(np.mean(accel_err**2))),
        "accel_mae": float(np.mean(np.abs(accel_err))),
        "accel_mean_err": float(np.mean(accel_err)),
        "accel_corr": accel_corr,
        "accel_sse": float(np.sum(accel_err**2)),
        "accel_abs_sum": float(np.sum(np.abs(accel_err))),
        "accel_err_sum": float(np.sum(accel_err)),
        "accel_x_sum": float(np.sum(accel_eval)),
        "accel_y_sum": float(np.sum(travel_accel_eval)),
        "accel_x2_sum": float(np.sum(accel_eval**2)),
        "accel_y2_sum": float(np.sum(travel_accel_eval**2)),
        "accel_xy_sum": float(np.sum(accel_eval * travel_accel_eval)),
        "n_eval": int(np.sum(accel_mask)),
    }


def candidate_rank_key(
    candidate: dict[str, float],
    preferred_total_mm: float,
) -> tuple[float, float, float, float]:
    return (
        float(candidate["n_over"]),
        candidate["oob_rms"],
        candidate["accel_rmse"],
        abs(candidate["top_adjacent_total_mm"] - preferred_total_mm),
    )


def format_geometry(geometry: AngleGeometry) -> str:
    zero = (
        f"fixed zero={geometry.top_zeroangle_rad:.6f} rad"
        if geometry.top_zeroangle_rad is not None
        else f"zero=p{geometry.zero_percentile:g}"
    )
    return (
        f"hyp={geometry.hypotenuse_mm:.1f} mm, "
        f"total={geometry.top_adjacent_total_mm:.1f} mm, "
        f"sign={geometry.angle_sign:+.0f}, {zero}"
    )


def format_candidate(candidate: dict[str, float], max_travel_mm: float) -> str:
    zero = (
        f"p{candidate['zero_percentile']:.1f}"
        if np.isfinite(candidate["zero_percentile"])
        else "fixed"
    )
    return (
        f"total={candidate['top_adjacent_total_mm']:.1f} mm, "
        f"zero={zero} ({candidate['zero_angle_deg']:.3f} deg), "
        f"range={candidate['travel_min']:.2f}..{candidate['travel_max']:.2f} mm, "
        f"p99.9={candidate['travel_p999']:.2f} mm, "
        f"n>{max_travel_mm:.0f}={candidate['n_over']}, "
        f"acc_rmse/mae/bias={candidate['accel_rmse']:.4f}/"
        f"{candidate['accel_mae']:.4f}/{candidate['accel_mean_err']:.4f}, "
        f"corr={candidate['accel_corr']:.4f}, n_eval={candidate['n_eval']}"
    )


def pool_candidates(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot pool an empty candidate list")
    n_eval = int(sum(row["n_eval"] for row in rows))
    n_valid = int(sum(row["n_valid"] for row in rows))
    x_sum = float(sum(row["accel_x_sum"] for row in rows))
    y_sum = float(sum(row["accel_y_sum"] for row in rows))
    x2_sum = float(sum(row["accel_x2_sum"] for row in rows))
    y2_sum = float(sum(row["accel_y2_sum"] for row in rows))
    xy_sum = float(sum(row["accel_xy_sum"] for row in rows))
    corr_num = (n_eval * xy_sum) - (x_sum * y_sum)
    corr_den = np.sqrt(
        max((n_eval * x2_sum) - (x_sum**2), 0.0)
        * max((n_eval * y2_sum) - (y_sum**2), 0.0)
    )
    return {
        **rows[0],
        "n_logs": len(rows),
        "n_over": int(sum(row["n_over"] for row in rows)),
        "oob_rms": float(np.sqrt(sum(row["oob_sse"] for row in rows) / n_valid)),
        "accel_rmse": float(np.sqrt(sum(row["accel_sse"] for row in rows) / n_eval)),
        "accel_mae": float(sum(row["accel_abs_sum"] for row in rows) / n_eval),
        "accel_mean_err": float(sum(row["accel_err_sum"] for row in rows) / n_eval),
        "accel_corr": float(corr_num / corr_den) if corr_den > 0 else float("nan"),
        "n_eval": n_eval,
        "n_valid": n_valid,
        "travel_min": float(min(row["travel_min"] for row in rows)),
        "travel_max": float(max(row["travel_max"] for row in rows)),
        "travel_p999": float(np.mean([row["travel_p999"] for row in rows])),
        "zero_angle_deg": float(np.mean([row["zero_angle_deg"] for row in rows])),
    }


def shared_candidate_rank_key(candidate: dict[str, float], preferred_total_mm: float) -> tuple[float, ...]:
    return (
        float(candidate["n_over"]),
        candidate["oob_rms"],
        candidate["accel_rmse"],
        abs(candidate["top_adjacent_total_mm"] - preferred_total_mm),
    )


def main() -> None:
    args = parse_args()

    baseline_rows: list[dict[str, float]] = []
    best_rows: list[dict[str, float]] = []
    shared_rows: dict[tuple[float, float, float, float], list[dict[str, float]]] = defaultdict(list)
    preferred_totals: list[float] = []

    for log in args.logs:
        inputs = load_log_inputs(
            log,
            logs_dir=args.logs_dir,
            cache_root=args.cache_root,
            allow_stale_cache=args.allow_stale_cache,
        )
        pipeline_geometry = inputs.geometry
        effective_hypotenuse = (
            args.hypotenuse if args.hypotenuse is not None else pipeline_geometry.hypotenuse_mm
        )
        effective_angle_sign = (
            args.angle_sign if args.angle_sign is not None else pipeline_geometry.angle_sign
        )
        baseline_total = (
            args.baseline_top_adjacent_total
            if args.baseline_top_adjacent_total is not None
            else pipeline_geometry.top_adjacent_total_mm
        )
        baseline_zero = (
            args.baseline_zero_percentile
            if args.baseline_zero_percentile is not None
            else pipeline_geometry.zero_percentile
        )
        baseline_zero_angle = (
            args.baseline_zero_angle_rad
            if args.baseline_zero_angle_rad is not None
            else (
                pipeline_geometry.top_zeroangle_rad
                if args.baseline_zero_percentile is None
                else None
            )
        )
        candidate_totals = args.candidate_top_adjacent_totals
        if candidate_totals is None:
            candidate_totals = [baseline_total + offset for offset in (-2.0, -1.0, 0.0, 1.0, 2.0)]
        skipped_candidate_totals = [
            total
            for total in candidate_totals
            if not -effective_hypotenuse < (total / 2.0) < effective_hypotenuse
        ]
        candidate_totals = [
            total
            for total in candidate_totals
            if -effective_hypotenuse < (total / 2.0) < effective_hypotenuse
        ]
        if not candidate_totals:
            raise ValueError(
                f"{log}: no candidate top-adjacent totals are valid for the "
                f"{effective_hypotenuse:.3f} mm hypotenuse"
            )

        baseline = evaluate_candidate(
            angle_lpf=inputs.angle_lpf,
            accel_proj_ms2=inputs.accel_proj_ms2,
            t_s=inputs.t_s,
            valid_mask=inputs.valid_mask,
            hypotenuse_mm=effective_hypotenuse,
            angle_sign=effective_angle_sign,
            top_adjacent_total_mm=baseline_total,
            zero_percentile=baseline_zero,
            max_travel_mm=args.max_travel,
            accel_threshold=args.accel_threshold,
            zero_angle_rad=baseline_zero_angle,
        )

        candidates = []
        for top_adjacent_total_mm in candidate_totals:
            for zero_percentile_value in args.candidate_zero_percentiles:
                candidate = evaluate_candidate(
                    angle_lpf=inputs.angle_lpf,
                    accel_proj_ms2=inputs.accel_proj_ms2,
                    t_s=inputs.t_s,
                    valid_mask=inputs.valid_mask,
                    hypotenuse_mm=effective_hypotenuse,
                    angle_sign=effective_angle_sign,
                    top_adjacent_total_mm=top_adjacent_total_mm,
                    zero_percentile=zero_percentile_value,
                    max_travel_mm=args.max_travel,
                    accel_threshold=args.accel_threshold,
                )
                candidates.append(candidate)
                shared_key = (
                    effective_hypotenuse,
                    effective_angle_sign,
                    top_adjacent_total_mm,
                    zero_percentile_value,
                )
                shared_rows[shared_key].append(candidate)

        ranked = sorted(candidates, key=lambda row: candidate_rank_key(row, baseline_total))
        best = ranked[0]

        baseline_rows.append(baseline)
        best_rows.append(best)
        preferred_totals.append(baseline_total)

        overrides = []
        if effective_hypotenuse != pipeline_geometry.hypotenuse_mm:
            overrides.append(f"hyp={effective_hypotenuse:.1f}")
        if effective_angle_sign != pipeline_geometry.angle_sign:
            overrides.append(f"sign={effective_angle_sign:+.0f}")
        if baseline_total != pipeline_geometry.top_adjacent_total_mm:
            overrides.append(f"baseline total={baseline_total:.1f}")
        if baseline_zero != pipeline_geometry.zero_percentile:
            overrides.append(f"baseline zero=p{baseline_zero:g}")
        if args.baseline_zero_angle_rad is not None:
            overrides.append(f"baseline fixed zero={args.baseline_zero_angle_rad:.6f} rad")

        pipeline_angle = inputs.angle_lpf * pipeline_geometry.angle_sign
        pipeline_zero = (
            pipeline_geometry.top_zeroangle_rad
            if pipeline_geometry.top_zeroangle_rad is not None
            else float(np.percentile(pipeline_angle, pipeline_geometry.zero_percentile))
        )
        pipeline_travel = get_travel(
            pipeline_angle,
            hypotenuse_mm=pipeline_geometry.hypotenuse_mm,
            top_adjacent_mm=pipeline_geometry.top_adjacent_total_mm / 2.0,
            top_zeroangle=pipeline_zero,
        )
        cache_travel_max_delta = float(
            np.max(np.abs(pipeline_travel - inputs.cached_travel_mm))
        )
        if inputs.cache_matches_config and cache_travel_max_delta > 1e-8:
            raise ValueError(
                f"{log}: reconstructed pipeline travel differs from the cache by up to "
                f"{cache_travel_max_delta:.6g} mm"
            )

        print(log)
        print(f"  pipeline config: {format_geometry(pipeline_geometry)}")
        if overrides:
            print(f"  analysis overrides: {', '.join(overrides)}")
        if skipped_candidate_totals:
            skipped = ", ".join(f"{value:g}" for value in skipped_candidate_totals)
            print(f"  skipped invalid candidate totals for this hypotenuse: {skipped} mm")
        if not inputs.cache_matches_config:
            print("  WARNING: cache config hash is stale (--allow-stale-cache was used)")
        else:
            print(f"  cache parity: travel max |delta|={cache_travel_max_delta:.3g} mm")
        print(
            f"  cached angle corruption: raw={inputs.bad_raw_pct:.2f}%, "
            f"excluded={np.mean(~inputs.valid_mask) * 100.0:.2f}% on the filtered timeline"
        )
        print("  baseline:", format_candidate(baseline, args.max_travel))
        print("  best candidate:", format_candidate(best, args.max_travel))
        print("  top candidates:")
        for candidate in ranked[: args.top_k]:
            print("   ", format_candidate(candidate, args.max_travel))
        print()

    pooled_baseline = pool_candidates(baseline_rows)
    pooled_best_per_log = pool_candidates(best_rows)
    preferred_total = float(np.mean(preferred_totals))

    complete_shared = [
        pool_candidates(rows)
        for rows in shared_rows.values()
        if len(rows) == len(args.logs)
    ]
    ranked_shared = sorted(
        complete_shared,
        key=lambda row: shared_candidate_rank_key(row, preferred_total),
    )

    print("Aggregate summary (pooled samples)")
    print(
        f"  baseline: total_over={pooled_baseline['n_over']}, "
        f"acc_rmse={pooled_baseline['accel_rmse']:.4f}, "
        f"mean_p99.9={pooled_baseline['travel_p999']:.2f} mm"
    )
    print(
        f"  best-per-log: total_over={pooled_best_per_log['n_over']}, "
        f"acc_rmse={pooled_best_per_log['accel_rmse']:.4f}, "
        f"mean_p99.9={pooled_best_per_log['travel_p999']:.2f} mm"
    )
    if ranked_shared:
        print("  best shared:", format_candidate(ranked_shared[0], args.max_travel))
        print("  top shared candidates:")
        for candidate in ranked_shared[: args.top_k]:
            print("   ", format_candidate(candidate, args.max_travel))
    else:
        print(
            "  best shared: unavailable because the logs do not share a complete "
            "hypotenuse/sign/candidate grid"
        )


if __name__ == "__main__":
    main()
