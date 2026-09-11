#!/usr/bin/env python3
"""Compare deployable front magnetic absolute-offset estimators.

Ground truth is used only to score methods and to describe calibration chunks.
Candidate offsets themselves use magnetometer, accelerometer, and learned scalar
curve data only.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from typing import Callable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.log_registry import DEFAULT_REGISTRY_PATH  # noqa: E402
from backend.mag_to_travel_model_core import MagToTravelModel  # noqa: E402
from backend.travel_solver_core import (  # noqa: E402
    solve_travel,
    solver_weights_for_mag_baseline,
)
from tools.front.mag_offset_calibration.sweep_post_mag_correction_fusion import (  # noqa: E402
    DEFAULT_HELD_OUT_SET,
    DEFAULT_TUNING_SETS,
    centered_error,
    load_inputs,
    resolve_cohorts,
    rmse,
    write_csv,
)


VALIDATION_COHORT = "harry"
TRAVEL_MAX_MM = 170.0


@dataclass(frozen=True)
class RefChunk:
    mag: np.ndarray
    rel_x: np.ndarray
    truth: np.ndarray
    start_mag: float
    start_truth: float
    direction: str


def flatten(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def find_ref_chunks(
    mag: np.ndarray,
    accel_m_s2: np.ndarray,
    truth: np.ndarray,
    time_s: np.ndarray,
    mag_baseline: float,
    *,
    fs_hz: float,
    bump_mag_min: float = 1000.0,
    still_a_max_mm_s2: float = 1000.0,
    bump_dx_min_mm: float = 20.0,
    still_len_s: float = 0.1,
    bump_len_s: float = 0.3,
    stride_s: float = 0.05,
    skips: int = 3,
) -> list[RefChunk]:
    """Reproduce ``GetMagTravelRefPoint.find_chunks`` with diagnostics."""
    mag = flatten(mag)
    accel_mm_s2 = flatten(accel_m_s2) * 1000.0
    truth = flatten(truth)
    time_s = flatten(time_s)
    dt_s = np.diff(time_s, prepend=time_s[0] - 0.01)
    still_len = int(still_len_s * fs_hz)
    bump_len = int(bump_len_s * fs_hz)
    stride = int(stride_s * fs_hz)
    chunk_len = still_len + bump_len
    still_slice = slice(0, still_len)
    bump_slice = slice(still_len, chunk_len)

    chunks: list[RefChunk] = []
    skip = 0
    for index in range(0, len(mag) - chunk_len, stride):
        if skip > 0:
            skip -= 1
            continue
        for direction, chunk_slice in (
            ("forward", slice(index, index + chunk_len)),
            ("reverse", slice(index + chunk_len, index, -1)),
        ):
            mag_chunk = mag[chunk_slice]
            accel_chunk = accel_mm_s2[chunk_slice]
            truth_chunk = truth[chunk_slice]
            dt_chunk = dt_s[chunk_slice]
            mag_still = mag_chunk[still_slice]
            accel_still = accel_chunk[still_slice]
            mag_bump = mag_chunk[bump_slice]
            accel_bump = accel_chunk[bump_slice]
            truth_bump = truth_chunk[bump_slice]
            dt_bump = dt_chunk[bump_slice]
            start_mag = float(np.mean(mag_still))
            if start_mag > mag_baseline:
                continue
            if float(np.max(np.abs(accel_still))) > still_a_max_mm_s2:
                continue
            if float(np.max(mag_bump)) < start_mag + bump_mag_min:
                continue
            velocity = np.cumsum(accel_bump * dt_bump)
            rel_x = np.cumsum(velocity * dt_bump)
            if float(np.max(rel_x)) < bump_dx_min_mm:
                continue
            skip = skips
            chunks.append(
                RefChunk(
                    mag=mag_bump,
                    rel_x=rel_x,
                    truth=truth_bump,
                    start_mag=start_mag,
                    start_truth=float(truth_bump[0]),
                    direction=direction,
                )
            )
    return chunks


def ref_point(
    chunks: Sequence[RefChunk],
    mag_baseline: float,
    *,
    start_quantile: float | None = None,
    start_margin_mg: float | None = None,
    direction: str | None = None,
    start_travel_estimator: Callable[[float], float] | None = None,
) -> tuple[float, float, int]:
    selected = list(chunks)
    if direction is not None:
        selected = [chunk for chunk in selected if chunk.direction == direction]
    if selected and start_quantile is not None:
        threshold = float(
            np.percentile([chunk.start_mag for chunk in selected], start_quantile)
        )
        selected = [chunk for chunk in selected if chunk.start_mag <= threshold]
    if selected and start_margin_mg is not None:
        threshold = min(chunk.start_mag for chunk in selected) + start_margin_mg
        selected = [chunk for chunk in selected if chunk.start_mag <= threshold]
    if not selected:
        return 0.0, mag_baseline + 2000.0, 0

    mag_points = np.concatenate([chunk.mag for chunk in selected])
    x_arrays = []
    for chunk in selected:
        start_x = 0.0 if start_travel_estimator is None else float(
            start_travel_estimator(chunk.start_mag)
        )
        x_arrays.append(chunk.rel_x + start_x)
    x_points = np.concatenate(x_arrays)
    mag_center = max(mag_baseline + 2000.0, float(np.median(mag_points)))
    in_band = (mag_points > mag_center - 1000.0) & (mag_points < mag_center + 1000.0)
    if not np.any(in_band):
        return 0.0, mag_baseline + 2000.0, 0
    return (
        float(np.median(x_points[in_band])),
        float(np.median(mag_points[in_band])),
        len(selected),
    )


def reference_diagnostics(
    chunks: Sequence[RefChunk], mag_baseline: float
) -> dict[str, float]:
    if not chunks:
        return {
            "reference_x_mm": 0.0,
            "reference_truth_mm": float("nan"),
            "reference_error_mm": float("nan"),
            "relative_integration_error_mm": float("nan"),
        }
    mag_points = np.concatenate([chunk.mag for chunk in chunks])
    rel_points = np.concatenate([chunk.rel_x for chunk in chunks])
    truth_points = np.concatenate([chunk.truth for chunk in chunks])
    truth_rel_points = np.concatenate(
        [chunk.truth - chunk.start_truth for chunk in chunks]
    )
    mag_center = max(mag_baseline + 2000.0, float(np.median(mag_points)))
    in_band = (mag_points > mag_center - 1000.0) & (mag_points < mag_center + 1000.0)
    if not np.any(in_band):
        return {
            "reference_x_mm": 0.0,
            "reference_truth_mm": float("nan"),
            "reference_error_mm": float("nan"),
            "relative_integration_error_mm": float("nan"),
        }
    reference_x = float(np.median(rel_points[in_band]))
    reference_truth = float(np.median(truth_points[in_band]))
    reference_rel_truth = float(np.median(truth_rel_points[in_band]))
    return {
        "reference_x_mm": reference_x,
        "reference_truth_mm": reference_truth,
        "reference_error_mm": reference_x - reference_truth,
        "relative_integration_error_mm": reference_x - reference_rel_truth,
    }


def still_mask(accel_m_s2: np.ndarray, fs_hz: float) -> np.ndarray:
    accel_mm_s2 = np.abs(flatten(accel_m_s2)) * 1000.0
    window = max(1, int(0.1 * fs_hz))
    mask = np.zeros(len(accel_mm_s2), dtype=bool)
    for start in range(0, len(accel_mm_s2) - window + 1, window):
        stop = start + window
        if float(np.max(accel_mm_s2[start:stop])) < 1000.0:
            mask[start:stop] = True
    return mask


def oob_offset(
    prediction: np.ndarray,
    *,
    lower_percentile: float,
    upper_percentile: float,
    tie_break: str,
) -> float:
    low, high = np.percentile(
        flatten(prediction), [lower_percentile, upper_percentile]
    )
    feasible_low = -float(low)
    feasible_high = TRAVEL_MAX_MM - float(high)
    if tie_break == "center":
        return 0.5 * (feasible_low + feasible_high)
    if tie_break == "nearest-zero":
        if feasible_low <= feasible_high:
            return float(np.clip(0.0, feasible_low, feasible_high))
        return 0.5 * (feasible_low + feasible_high)
    raise ValueError(tie_break)


def candidate_offsets(
    *,
    raw_prediction: np.ndarray,
    current_offset: float,
    mag: np.ndarray,
    good_mask: np.ndarray,
    still: np.ndarray,
    chunks: Sequence[RefChunk],
    mag_baseline: float,
    model: MagToTravelModel,
) -> dict[str, float]:
    good = good_mask & np.isfinite(raw_prediction)
    still_good = good & still
    values: dict[str, float] = {"current": current_offset}

    ref_variants = {
        "ref_all_no_fallback": {},
        "ref_lowest_start_q25": {"start_quantile": 25.0},
        "ref_lowest_start_q50": {"start_quantile": 50.0},
        "ref_start_within_100mg": {"start_margin_mg": 100.0},
        "ref_start_within_250mg": {"start_margin_mg": 250.0},
        "ref_forward_only": {"direction": "forward"},
        "ref_reverse_only": {"direction": "reverse"},
    }
    for name, kwargs in ref_variants.items():
        ref_x, ref_mag, _ = ref_point(chunks, mag_baseline, **kwargs)
        values[name] = ref_x - float(model.pred_x(ref_mag))

    for percentile in (0.0, 1.0, 2.0, 5.0, 8.0, 10.0, 15.0):
        values[f"prediction_p{percentile:g}_to_zero"] = -float(
            np.percentile(raw_prediction[good], percentile)
        )
    values["pipeline_mag_p8_to_zero"] = -float(
        model.pred_x(float(np.percentile(mag, 8.0)))
    )
    if np.any(still_good):
        for percentile in (0.0, 5.0, 10.0, 25.0):
            values[f"still_prediction_p{percentile:g}_to_zero"] = -float(
                np.percentile(raw_prediction[still_good], percentile)
            )

    zero_mag = float(np.percentile(mag[good], 8.0))
    zero_x = float(model.pred_x(zero_mag))
    start_estimator = lambda start_mag: max(0.0, float(model.pred_x(start_mag)) - zero_x)
    ref_x, ref_mag, _ = ref_point(
        chunks,
        mag_baseline,
        start_travel_estimator=start_estimator,
    )
    values["ref_plus_p8_start_estimate"] = ref_x - float(model.pred_x(ref_mag))

    p2_offset = values["prediction_p2_to_zero"]
    for current_weight in tuple(value / 10.0 for value in range(1, 10)):
        values[f"blend_current_p2_w{current_weight:g}"] = (
            current_weight * current_offset + (1.0 - current_weight) * p2_offset
        )
    values["raise_current_to_p2"] = max(current_offset, p2_offset)
    for threshold in (5, 10, 15, 20, 25):
        values[f"chunk_count_{threshold}_then_current"] = (
            current_offset if len(chunks) >= threshold else p2_offset
        )
    for cap in (0.25, 0.5, 0.75, 1.0):
        for scale in (10, 20, 30, 40, 50):
            current_weight = min(len(chunks) / scale, cap)
            values[f"chunk_confidence_n{scale}_cap{cap:g}"] = (
                current_weight * current_offset
                + (1.0 - current_weight) * p2_offset
            )
    confidence_weight = min(len(chunks) / 10.0, 0.5)
    confidence_offset = (
        confidence_weight * current_offset
        + (1.0 - confidence_weight) * p2_offset
    )
    values["current_if_25_chunks_else_confidence"] = (
        current_offset if len(chunks) >= 25 else confidence_offset
    )

    for lower, upper, suffix in ((0.0, 100.0, "full"), (1.0, 99.0, "p1_p99")):
        for tie_break in ("center", "nearest-zero"):
            values[f"oob_{suffix}_{tie_break}"] = oob_offset(
                raw_prediction[good],
                lower_percentile=lower,
                upper_percentile=upper,
                tie_break=tie_break,
            )
    return values


def load_log(cache_path: Path) -> dict[str, np.ndarray | float]:
    with np.load(cache_path, allow_pickle=False) as cache:
        return {
            "time": flatten(cache["travel__t"]),
            "truth": flatten(cache["travel__x"]),
            "mag": flatten(cache["mag/norm/corr/lpf__x"]),
            "accel": flatten(cache["accel/lpfhp/proj__x"]),
            "raw": flatten(cache["travel/mag_model__x"]),
            "adjusted": flatten(cache["travel/mag_model/adj__x"]),
            "corrected": flatten(cache["travel/mag_nuisance/corrected__x"]),
            "bad": np.asarray(cache["mag/norm/bad_mask__x"], dtype=bool).reshape(-1),
            "eval_mask": np.asarray(cache["boring_mask"], dtype=bool).reshape(-1),
            "baseline": float(flatten(cache["mag_baseline"])[0]),
            "offset": float(flatten(cache["mag_model_offset_mm"])[0]),
            "coefficients": flatten(cache["mag_model_coeffs"]),
        }


def mean(rows: Sequence[dict[str, object]], key: str) -> float:
    return float(np.nanmean([float(row[key]) for row in rows]))


def aggregate_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "offset_mm",
        "offset_error_mm",
        "abs_offset_error_mm",
        "scalar_mean_error_mm",
        "scalar_uncentered_rmse_mm",
        "corrected_mean_error_mm",
        "corrected_uncentered_rmse_mm",
        "corrected_low_uncentered_rmse_mm",
    )
    output: list[dict[str, object]] = []
    for split in ("tuning", "validation"):
        split_rows = [row for row in rows if row["split"] == split]
        cohorts = sorted({str(row["cohort"]) for row in split_rows})
        methods = sorted({str(row["method"]) for row in split_rows})
        for method in methods:
            cohort_summaries: list[dict[str, object]] = []
            for cohort in cohorts:
                selected = [
                    row
                    for row in split_rows
                    if row["cohort"] == cohort and row["method"] == method
                ]
                if not selected:
                    continue
                summary = {
                    "split": split,
                    "cohort": cohort,
                    "method": method,
                    "n_logs": len(selected),
                    **{metric: mean(selected, metric) for metric in metrics},
                }
                cohort_summaries.append(summary)
                output.append(summary)
            if cohort_summaries:
                output.append(
                    {
                        "split": split,
                        "cohort": "cohort-balanced",
                        "method": method,
                        "n_logs": sum(int(row["n_logs"]) for row in cohort_summaries),
                        **{
                            metric: mean(cohort_summaries, metric)
                            for metric in metrics
                        },
                    }
                )
    return output


def replay_solver_candidates(
    method_rows: Sequence[dict[str, object]],
    methods: Sequence[str],
    cache_root: Path,
    max_nfev: int,
    existing_rows: Sequence[dict[str, object]] = (),
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    offset_lookup = {
        (str(row["cohort"]), str(row["log"]), str(row["method"])): float(
            row["offset_mm"]
        )
        for row in method_rows
    }
    logs = sorted(
        {
            (str(row["split"]), str(row["cohort"]), str(row["log"]))
            for row in method_rows
        }
    )
    rows: list[dict[str, object]] = [
        dict(row) for row in existing_rows if str(row["method"]) in methods
    ]
    completed = {
        (str(row["cohort"]), str(row["log"]), str(row["method"]))
        for row in rows
    }
    for index, (split, cohort, log_name) in enumerate(logs, 1):
        inputs, truth, mask, cached_solved = load_inputs(
            cache_root / log_name / "cache" / "all.npz"
        )
        low_mask = mask & (truth >= 0.0) & (truth < 30.0)
        current_offset = offset_lookup[(cohort, log_name, "current")]
        for method in methods:
            if (cohort, log_name, method) in completed:
                continue
            offset = offset_lookup[(cohort, log_name, method)]
            reusable = next(
                (
                    row
                    for row in rows
                    if str(row["cohort"]) == cohort
                    and str(row["log"]) == log_name
                    and np.isclose(float(row["offset_mm"]), offset, atol=1e-12, rtol=0.0)
                ),
                None,
            )
            if reusable is not None:
                reused = dict(reusable)
                reused["method"] = method
                reused["offset_mm"] = offset
                rows.append(reused)
                completed.add((cohort, log_name, method))
                continue
            if method == "current":
                prediction = cached_solved
                success = True
                nfev = 0
            else:
                candidate_inputs = replace(
                    inputs,
                    mag_preds_mm=inputs.mag_preds_mm + offset - current_offset,
                )
                result = solve_travel(
                    candidate_inputs,
                    solver_weights_for_mag_baseline(float(inputs.mag_baseline)),
                    max_nfev=max_nfev,
                )
                prediction = result.x
                success = bool(result.scipy_result.success)
                nfev = int(result.scipy_result.nfev)
            error = prediction - truth
            centered = centered_error(prediction, truth, mask)
            rows.append(
                {
                    "split": split,
                    "cohort": cohort,
                    "log": log_name,
                    "method": method,
                    "offset_mm": offset,
                    "mean_error_mm": float(np.mean(error[mask])),
                    "centered_rmse_mm": rmse(centered[mask]),
                    "low_centered_rmse_mm": rmse(centered[low_mask]),
                    "uncentered_rmse_mm": rmse(error[mask]),
                    "low_uncentered_rmse_mm": rmse(error[low_mask]),
                    "success": success,
                    "nfev": nfev,
                }
            )
        print(f"[solver {index}/{len(logs)}] {cohort}/{log_name}", flush=True)

    metrics = (
        "offset_mm",
        "mean_error_mm",
        "centered_rmse_mm",
        "low_centered_rmse_mm",
        "uncentered_rmse_mm",
        "low_uncentered_rmse_mm",
    )
    aggregate: list[dict[str, object]] = []
    for split in ("tuning", "validation"):
        split_rows = [row for row in rows if row["split"] == split]
        cohorts = sorted({str(row["cohort"]) for row in split_rows})
        for method in methods:
            cohort_summaries: list[dict[str, object]] = []
            for cohort in cohorts:
                selected = [
                    row
                    for row in split_rows
                    if row["cohort"] == cohort and row["method"] == method
                ]
                if not selected:
                    continue
                summary = {
                    "split": split,
                    "cohort": cohort,
                    "method": method,
                    "n_logs": len(selected),
                    "success_fraction": float(
                        np.mean([bool(row["success"]) for row in selected])
                    ),
                    **{metric: mean(selected, metric) for metric in metrics},
                }
                cohort_summaries.append(summary)
                aggregate.append(summary)
            aggregate.append(
                {
                    "split": split,
                    "cohort": "cohort-balanced",
                    "method": method,
                    "n_logs": sum(int(row["n_logs"]) for row in cohort_summaries),
                    "success_fraction": mean(cohort_summaries, "success_fraction"),
                    **{metric: mean(cohort_summaries, metric) for metric in metrics},
                }
            )
    return rows, aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--cache-root", type=Path, default=REPO_ROOT / "backend" / "run_artifacts"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--solver-methods", nargs="+", default=[])
    parser.add_argument("--max-nfev", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort_names = (*DEFAULT_TUNING_SETS, DEFAULT_HELD_OUT_SET)
    cohorts = resolve_cohorts(args.registry, cohort_names)
    chunk_rows: list[dict[str, object]] = []
    method_rows: list[dict[str, object]] = []
    for cohort, log_names in cohorts.items():
        split = "validation" if cohort == VALIDATION_COHORT else "tuning"
        for log_name in log_names:
            data = load_log(args.cache_root / log_name / "cache" / "all.npz")
            time_s = data["time"]
            fs_hz = 1.0 / float(np.median(np.diff(time_s)))
            chunks = find_ref_chunks(
                data["mag"],
                data["accel"],
                data["truth"],
                time_s,
                data["baseline"],
                fs_hz=fs_hz,
            )
            diagnostics = reference_diagnostics(chunks, data["baseline"])
            starts = np.asarray([chunk.start_truth for chunk in chunks], dtype=float)
            start_mags = np.asarray([chunk.start_mag for chunk in chunks], dtype=float)
            chunk_rows.append(
                {
                    "split": split,
                    "cohort": cohort,
                    "log": log_name,
                    "n_chunks": len(chunks),
                    "n_chunks_start_le_3mm": int(np.sum(starts <= 3.0)),
                    "fraction_chunks_start_le_3mm": (
                        float(np.mean(starts <= 3.0)) if len(starts) else 0.0
                    ),
                    "median_chunk_start_truth_mm": (
                        float(np.median(starts)) if len(starts) else float("nan")
                    ),
                    "median_chunk_start_mag_mg": (
                        float(np.median(start_mags)) if len(start_mags) else float("nan")
                    ),
                    **diagnostics,
                    "cached_offset_mm": data["offset"],
                }
            )

            model = MagToTravelModel(pred_soft_mg=50.0)
            model.set_coeffs(data["coefficients"])
            candidates = candidate_offsets(
                raw_prediction=data["raw"],
                current_offset=data["offset"],
                mag=data["mag"],
                good_mask=~data["bad"],
                still=still_mask(data["accel"], fs_hz),
                chunks=chunks,
                mag_baseline=data["baseline"],
                model=model,
            )
            eval_mask = data["eval_mask"]
            truth = data["truth"]
            low_mask = eval_mask & (truth >= 0.0) & (truth < 30.0)
            oracle_offset = float(np.mean(truth[eval_mask] - data["raw"][eval_mask]))
            for method, offset in candidates.items():
                scalar = data["raw"] + offset
                corrected = data["corrected"] + offset - data["offset"]
                scalar_error = scalar - truth
                corrected_error = corrected - truth
                method_rows.append(
                    {
                        "split": split,
                        "cohort": cohort,
                        "log": log_name,
                        "method": method,
                        "offset_mm": offset,
                        "oracle_offset_mm": oracle_offset,
                        "offset_error_mm": offset - oracle_offset,
                        "abs_offset_error_mm": abs(offset - oracle_offset),
                        "scalar_mean_error_mm": float(np.mean(scalar_error[eval_mask])),
                        "scalar_uncentered_rmse_mm": rmse(scalar_error[eval_mask]),
                        "corrected_mean_error_mm": float(
                            np.mean(corrected_error[eval_mask])
                        ),
                        "corrected_uncentered_rmse_mm": rmse(
                            corrected_error[eval_mask]
                        ),
                        "corrected_low_uncentered_rmse_mm": rmse(
                            corrected_error[low_mask]
                        ),
                    }
                )

    aggregate = aggregate_rows(method_rows)
    tuning = [
        row
        for row in aggregate
        if row["split"] == "tuning" and row["cohort"] == "cohort-balanced"
    ]
    selected = min(
        tuning,
        key=lambda row: (
            float(row["corrected_uncentered_rmse_mm"]),
            float(row["abs_offset_error_mm"]),
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "chunk_diagnostics.csv", chunk_rows)
    write_csv(args.output_dir / "per_log_method_metrics.csv", method_rows)
    write_csv(args.output_dir / "aggregate_method_metrics.csv", aggregate)
    (args.output_dir / "selection.json").write_text(
        json.dumps(
            {
                "validation_cohort": VALIDATION_COHORT,
                "tuning_cohorts": [name for name in cohort_names if name != VALIDATION_COHORT],
                "selection_metric": "cohort-balanced corrected_uncentered_rmse_mm",
                "selected_method": selected["method"],
                "tuning_summary": selected,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.solver_methods:
        solver_path = args.output_dir / "solver_per_log_metrics.csv"
        existing_solver_rows: list[dict[str, object]] = []
        if solver_path.exists():
            with solver_path.open(newline="", encoding="utf-8") as handle:
                existing_solver_rows = list(csv.DictReader(handle))
        solver_rows, solver_aggregate = replay_solver_candidates(
            method_rows,
            args.solver_methods,
            args.cache_root,
            args.max_nfev,
            existing_solver_rows,
        )
        write_csv(args.output_dir / "solver_per_log_metrics.csv", solver_rows)
        write_csv(args.output_dir / "solver_aggregate_metrics.csv", solver_aggregate)
    print(json.dumps(selected, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
