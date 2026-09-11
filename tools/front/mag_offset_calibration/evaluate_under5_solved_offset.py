#!/usr/bin/env python3
"""Evaluate a perfect under-5 mm detector as a final-solved offset anchor."""

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
from tools.front.mag_offset_calibration.sweep_post_mag_correction_fusion import (  # noqa: E402
    DEFAULT_HELD_OUT_SET,
    DEFAULT_TUNING_SETS,
    resolve_cohorts,
    rmse,
    write_csv,
)


TARGET_MEAN_MM = 2.5


def flatten(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def offset_from_detector(
    solved: np.ndarray,
    detector_mask: np.ndarray,
    *,
    target_mean_mm: float = TARGET_MEAN_MM,
) -> float:
    solved = flatten(solved)
    detector_mask = np.asarray(detector_mask, dtype=bool).reshape(-1)
    if not np.any(detector_mask):
        raise ValueError("detector mask is empty")
    return float(target_mean_mm - np.mean(solved[detector_mask]))


def render_markdown(
    rows: Sequence[dict[str, object]],
    aggregate: Sequence[dict[str, object]],
) -> str:
    lines = [
        "# Perfect under-5 mm detector offset experiment",
        "",
        "For each log, the detector mask is exactly `0 < GT travel < 5 mm`. The applied whole-trajectory offset is `2.5 mm - mean(final_solved[detector_mask])`. The detector uses every finite sample; errors are evaluated on the existing `boring_mask`.",
        "",
        "## Cohort macro averages",
        "",
        "| cohort | logs | detected GT mean | detected solved mean | applied offset | mean error before | mean error after | mean absolute error before | mean absolute error after | RMSE before | RMSE after | logs with lower absolute ME |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            "| {cohort} | {n_logs} | {detected_gt_mean_mm:.3f} | {detected_solved_mean_mm:.3f} | {applied_offset_mm:+.3f} | {mean_error_before_mm:+.3f} | {mean_error_after_mm:+.3f} | {mean_abs_error_before_mm:.3f} | {mean_abs_error_after_mm:.3f} | {rmse_before_mm:.3f} | {rmse_after_mm:.3f} | {abs_me_improved_logs}/{n_logs} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Per-log table",
            "",
            "| cohort | log | detected samples | detected GT mean | detected solved mean | applied offset | mean error before | mean error after | detected-band error after | RMSE before | RMSE after |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {cohort} | {log} | {n_detected} | {detected_gt_mean_mm:.3f} | {detected_solved_mean_mm:.3f} | {applied_offset_mm:+.3f} | {mean_error_before_mm:+.3f} | {mean_error_after_mm:+.3f} | {detected_error_after_mm:+.3f} | {rmse_before_mm:.3f} | {rmse_after_mm:.3f} |".format(
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
    for cohort, log_names in cohorts.items():
        for log_name in log_names:
            cache_path = args.cache_root / log_name / "cache" / "all.npz"
            with np.load(cache_path, allow_pickle=False) as cache:
                truth = flatten(cache["travel__x"])
                solved = flatten(cache["travel/solved__x"])
                eval_mask = np.asarray(cache["boring_mask"], dtype=bool).reshape(-1)
            finite = np.isfinite(truth) & np.isfinite(solved)
            detector_mask = finite & (truth > 0.0) & (truth < 5.0)
            eval_mask &= finite
            applied_offset = offset_from_detector(solved, detector_mask)
            shifted = solved + applied_offset
            error_before = solved - truth
            error_after = shifted - truth
            mean_before = float(np.mean(error_before[eval_mask]))
            mean_after = float(np.mean(error_after[eval_mask]))
            rows.append(
                {
                    "cohort": cohort,
                    "log": log_name,
                    "n_detected": int(np.sum(detector_mask)),
                    "detected_gt_mean_mm": float(np.mean(truth[detector_mask])),
                    "detected_solved_mean_mm": float(np.mean(solved[detector_mask])),
                    "applied_offset_mm": applied_offset,
                    "mean_error_before_mm": mean_before,
                    "mean_error_after_mm": mean_after,
                    "abs_mean_error_before_mm": abs(mean_before),
                    "abs_mean_error_after_mm": abs(mean_after),
                    "detected_error_after_mm": float(
                        np.mean(error_after[detector_mask])
                    ),
                    "rmse_before_mm": rmse(error_before[eval_mask]),
                    "rmse_after_mm": rmse(error_after[eval_mask]),
                }
            )

    aggregate: list[dict[str, object]] = []
    for cohort in cohorts:
        selected = [row for row in rows if row["cohort"] == cohort]
        average_keys = (
            "detected_gt_mean_mm",
            "detected_solved_mean_mm",
            "applied_offset_mm",
            "mean_error_before_mm",
            "mean_error_after_mm",
            "rmse_before_mm",
            "rmse_after_mm",
        )
        aggregate.append(
            {
                "cohort": cohort,
                "n_logs": len(selected),
                **{
                    key: float(np.mean([float(row[key]) for row in selected]))
                    for key in average_keys
                },
                "mean_abs_error_before_mm": float(
                    np.mean([float(row["abs_mean_error_before_mm"]) for row in selected])
                ),
                "mean_abs_error_after_mm": float(
                    np.mean([float(row["abs_mean_error_after_mm"]) for row in selected])
                ),
                "abs_me_improved_logs": sum(
                    float(row["abs_mean_error_after_mm"])
                    < float(row["abs_mean_error_before_mm"])
                    for row in selected
                ),
            }
        )
    aggregate.append(
        {
            "cohort": "cohort-balanced",
            "n_logs": len(rows),
            **{
                key: float(np.mean([float(row[key]) for row in aggregate]))
                for key in (
                    "detected_gt_mean_mm",
                    "detected_solved_mean_mm",
                    "applied_offset_mm",
                    "mean_error_before_mm",
                    "mean_error_after_mm",
                    "mean_abs_error_before_mm",
                    "mean_abs_error_after_mm",
                    "rmse_before_mm",
                    "rmse_after_mm",
                )
            },
            "abs_me_improved_logs": sum(
                int(row["abs_me_improved_logs"]) for row in aggregate
            ),
        }
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_csv, rows)
    write_csv(args.output_csv.with_name("under5_solved_offset_aggregate.csv"), aggregate)
    args.output_markdown.write_text(
        render_markdown(rows, aggregate), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
