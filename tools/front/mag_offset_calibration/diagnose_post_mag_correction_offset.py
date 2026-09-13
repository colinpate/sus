#!/usr/bin/env python3
"""Oracle-offset diagnostic for post-mag-correction fusion gating.

This intentionally uses ground truth to remove one constant offset per log. It is
an explanatory experiment, not a deployable calibration or a tuning procedure.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import replace
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.front.mag_offset_calibration.sweep_post_mag_correction_fusion import (
    CURRENT_FLOOR,
    DEFAULT_HELD_OUT_SET,
    DEFAULT_TUNING_SETS,
    LOW_TRAVEL_MAX_MM,
    centered_error,
    load_inputs,
    resolve_cohorts,
    rmse,
    write_csv,
)
from backend.log_registry import DEFAULT_REGISTRY_PATH
from backend.travel_solver_core import solve_travel, solver_weights_for_mag_baseline


DEFAULT_MODES = ("baseline", "750")


def run_log(
    cohort: str,
    log_name: str,
    modes: Sequence[str],
    cache_root: str,
    max_nfev: int,
) -> list[dict[str, object]]:
    inputs, truth, mask, _ = load_inputs(
        Path(cache_root) / log_name / "cache" / "all.npz"
    )
    offset_mm = -float(np.median(inputs.mag_preds_mm[mask] - truth[mask]))
    shifted_inputs = replace(
        inputs,
        mag_preds_mm=inputs.mag_preds_mm + offset_mm,
    )
    low_mask = mask & (truth >= 0.0) & (truth < LOW_TRAVEL_MAX_MM)
    rows: list[dict[str, object]] = []
    for mode in modes:
        threshold = (
            float(inputs.mag_baseline) if mode == "baseline" else float(mode)
        )
        weights = replace(
            solver_weights_for_mag_baseline(float(inputs.mag_baseline)),
            mag_x_thresh=threshold,
            mag_off_floor=CURRENT_FLOOR,
        )
        result = solve_travel(shifted_inputs, weights, max_nfev=max_nfev)
        prediction = result.x
        centered = centered_error(prediction, truth, mask)
        error = prediction - truth
        rows.append(
            {
                "cohort": cohort,
                "log": log_name,
                "threshold_mode": mode,
                "oracle_mag_offset_mm": offset_mm,
                "mean_error_mm": float(np.mean(error[mask])),
                "centered_rmse_mm": rmse(centered[mask]),
                "low_travel_centered_rmse_mm": rmse(centered[low_mask]),
                "uncentered_rmse_mm": rmse(error[mask]),
                "success": bool(result.scipy_result.success),
                "nfev": int(result.scipy_result.nfev),
            }
        )
    return rows


def aggregate(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "oracle_mag_offset_mm",
        "mean_error_mm",
        "centered_rmse_mm",
        "low_travel_centered_rmse_mm",
        "uncentered_rmse_mm",
    )
    output: list[dict[str, object]] = []
    cohorts = sorted({str(row["cohort"]) for row in rows})
    modes = tuple(dict.fromkeys(str(row["threshold_mode"]) for row in rows))
    for mode in modes:
        cohort_rows: list[dict[str, object]] = []
        for cohort in cohorts:
            selected = [
                row
                for row in rows
                if row["cohort"] == cohort and row["threshold_mode"] == mode
            ]
            summary = {
                "cohort": cohort,
                "threshold_mode": mode,
                "n_logs": len(selected),
                "success_fraction": float(np.mean([row["success"] for row in selected])),
                **{
                    metric: float(np.mean([float(row[metric]) for row in selected]))
                    for metric in metrics
                },
            }
            cohort_rows.append(summary)
            output.append(summary)
        output.append(
            {
                "cohort": "cohort-balanced",
                "threshold_mode": mode,
                "n_logs": len(rows) // len(modes),
                "success_fraction": float(
                    np.mean([float(row["success_fraction"]) for row in cohort_rows])
                ),
                **{
                    metric: float(np.mean([float(row[metric]) for row in cohort_rows]))
                    for metric in metrics
                },
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--cache-root", type=Path, default=REPO_ROOT / "backend" / "run_artifacts"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohorts = resolve_cohorts(
        args.registry, (*DEFAULT_TUNING_SETS, DEFAULT_HELD_OUT_SET)
    )
    tasks = [
        (cohort, log_name)
        for cohort, log_names in cohorts.items()
        for log_name in log_names
    ]
    rows: list[dict[str, object]] = []
    if args.workers <= 1:
        for index, (cohort, log_name) in enumerate(tasks, 1):
            rows.extend(
                run_log(
                    cohort,
                    log_name,
                    args.modes,
                    str(args.cache_root),
                    args.max_nfev,
                )
            )
            print(f"[{index}/{len(tasks)}] {cohort}/{log_name}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_log,
                    cohort,
                    log_name,
                    args.modes,
                    str(args.cache_root),
                    args.max_nfev,
                ): (cohort, log_name)
                for cohort, log_name in tasks
            }
            for index, future in enumerate(as_completed(futures), 1):
                cohort, log_name = futures[future]
                rows.extend(future.result())
                print(f"[{index}/{len(tasks)}] {cohort}/{log_name}", flush=True)

    rows.sort(key=lambda row: (str(row["cohort"]), str(row["log"]), str(row["threshold_mode"])))
    summaries = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "oracle_offset_per_log.csv", rows)
    write_csv(args.output_dir / "oracle_offset_aggregate.csv", summaries)
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=list(summaries[0]),
    )
    writer.writeheader()
    writer.writerows(summaries)


if __name__ == "__main__":
    main()
