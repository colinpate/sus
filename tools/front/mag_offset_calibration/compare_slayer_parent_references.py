#!/usr/bin/env python3
"""Redo Slayer filtered-log parent references in corrected magnetic coordinates.

For every filtered child log, this experiment reruns the first solver and
magnetic nuisance correction with zero scalar offset.  It then compares:

* the historical fixed parent reference stored in the registry;
* a reference estimated independently from the child; and
* a reference estimated from the complete parent log and reused by the child.

All reference detection uses the production Slayer safety policy and the
selected 0.20-second bump window.  Ground truth is used only for evaluation
and reference-error diagnostics.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from log_registry import DEFAULT_REGISTRY_PATH, LogRegistry  # noqa: E402
from tools.front.mag_offset_calibration.compare_ref_point_placement import (  # noqa: E402
    load_raw_inputs,
    max_offset_delta,
    reference_config,
    run_nuisance_path,
    run_solver,
)
from tools.front.mag_offset_calibration.sweep_ref_point_parameters import (  # noqa: E402
    LogData,
    apply_fallback,
    evaluate_config,
    flatten,
    load_log,
    model_for,
    rmse,
    write_csv,
)


CORE_SENSOR_COLUMNS = (
    ("lis1_x", "lis1_y", "lis1_z"),
    ("lis2_x", "lis2_y", "lis2_z"),
    ("gyro1_dps10_x", "gyro1_dps10_y", "gyro1_dps10_z"),
    ("gyro2_dps10_x", "gyro2_dps10_y", "gyro2_dps10_z"),
    ("mmc_mG_x", "mmc_mG_y", "mmc_mG_z"),
)
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "slayer_chunk_pipeline_grown_20pct_post_ref_v1"
HISTORICAL_REFERENCES = {
    "log-0145": (59.70822822060133, 3082.6808146518842),
    "log-0147": (0.0, 1249.5147284632924),
    "log-0151": (78.19572979772903, 3334.5191674375515),
    "log-0152": (0.0, 1311.4907327788778),
    "log-0155": (0.0, 1342.9160200017532),
}


def production_config(name: str, max_offset_delta_mm: float):
    return replace(
        reference_config(name, bump_len_s=0.2, max_offset_delta_mm=max_offset_delta_mm),
        min_ref_points=20,
    )


def corrected_data(data: LogData, nuisance) -> LogData:
    return replace(
        data,
        mag_mg=nuisance.corrected_mag_mg,
        base_corrected_mm=nuisance.corrected_travel_mm,
    )


def fallback_reference(data: LogData, percentile: float = 8.0) -> tuple[float, float]:
    finite_mag = data.mag_mg[np.isfinite(data.mag_mg)]
    if finite_mag.size == 0:
        raise ValueError(f"{data.log}: no finite corrected magnetic samples")
    return 0.0, float(np.percentile(finite_mag, percentile))


def calculate_parent_reference(
    registry_path: str,
    cache_root: str,
    parent_log: str,
    max_nfev: int,
) -> dict[str, Any]:
    registry = LogRegistry.load(Path(registry_path))
    data = load_log(
        registry,
        Path(cache_root),
        parent_log,
        split="parent",
        cohort="slayer",
    )
    raw = load_raw_inputs(registry, Path(cache_root), parent_log)
    nuisance = run_nuisance_path(data, raw, 0.0, max_nfev=max_nfev)
    target = corrected_data(data, nuisance)
    config = production_config(
        "parent_post_corrected_0p2s",
        max_offset_delta(raw.processing_config, "post"),
    )
    result = evaluate_config(target, config, {})
    if bool(result["calibration_found"]):
        reference = (
            float(result["reference_x_mm"]),
            float(result["reference_mag_mg"]),
        )
        source = "detected"
    else:
        reference = fallback_reference(target, config.fallback_percentile)
        source = "p8_fallback"
    return {
        "parent_log": parent_log,
        "reference_x_mm": reference[0],
        "reference_mag_mg": reference[1],
        "source": source,
        "n_chunks": int(result["n_chunks"]),
        "n_ref_points": int(result["n_ref_points"]),
        "reference_truth_mm": float(result["reference_truth_mm"]),
        "reference_error_mm": float(result["reference_error_mm"]),
        "parent_apply_used_fallback": bool(result["used_fallback"]),
        "parent_apply_fallback_reason": str(result["fallback_reason"]),
    }


def fixed_reference_offset(
    target: LogData,
    config,
    reference: tuple[float, float],
) -> tuple[float, bool, str]:
    ref_x, ref_mag = reference
    raw_offset = float(ref_x) - float(model_for(target).pred_x(float(ref_mag)))
    return apply_fallback(target, config, raw_offset)


def core_dropout_mask(log_name: str, expected_length: int) -> np.ndarray:
    path = REPO_ROOT / "logs" / "converted" / f"{log_name}.csv"
    columns = [column for group in CORE_SENSOR_COLUMNS for column in group]
    frame = pd.read_csv(path, usecols=columns)
    raw_bad = np.logical_or.reduce(
        [
            np.all(frame[list(group)].to_numpy(dtype=float) == 0.0, axis=1)
            for group in CORE_SENSOR_COLUMNS
        ]
    )
    mask = raw_bad[::2]
    if len(mask) != expected_length:
        raise ValueError(
            f"{log_name}: dropout mask length {len(mask)} != cache length {expected_length}"
        )
    return mask


def metric_row(
    data: LogData,
    prediction: np.ndarray,
    *,
    parent_log: str,
    variant: str,
    stage: str,
    evaluation: str,
    mask: np.ndarray,
    offset_mm: float,
    used_fallback: bool,
    fallback_reason: str,
    final_success: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    valid = mask & np.isfinite(prediction) & np.isfinite(data.truth_mm)
    error = np.asarray(prediction)[valid] - data.truth_mm[valid]
    return (
        {
            "log": data.log,
            "parent_log": parent_log,
            "variant": variant,
            "stage": stage,
            "evaluation": evaluation,
            "samples": len(error),
            "offset_mm": offset_mm,
            "used_fallback": used_fallback,
            "fallback_reason": fallback_reason,
            "mean_error_mm": float(np.mean(error)),
            "rmse_mm": rmse(error),
            "centered_rmse_mm": rmse(error - float(np.mean(error))),
            "final_solver_success": final_success,
        },
        error,
    )


def run_child(
    registry_path: str,
    cache_root: str,
    child_log: str,
    parent_reference: tuple[float, float],
    max_nfev: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    registry = LogRegistry.load(Path(registry_path))
    data = load_log(
        registry,
        Path(cache_root),
        child_log,
        split="validation",
        cohort="slayer-filtered",
    )
    parent_log = data.parent_log
    raw = load_raw_inputs(registry, Path(cache_root), child_log)
    nuisance = run_nuisance_path(data, raw, 0.0, max_nfev=max_nfev)
    target = corrected_data(data, nuisance)
    config = production_config(
        "child_post_corrected_0p2s",
        max_offset_delta(raw.processing_config, "post"),
    )
    automatic = evaluate_config(target, config, {})
    variants: list[tuple[str, float, bool, str]] = [
        (
            "child_automatic",
            float(automatic["offset_mm"]),
            bool(automatic["used_fallback"]),
            str(automatic["fallback_reason"]),
        )
    ]
    for name, reference in (
        ("registry_old_parent", HISTORICAL_REFERENCES[parent_log]),
        ("recomputed_parent", parent_reference),
    ):
        offset, used_fallback, reason = fixed_reference_offset(target, config, reference)
        variants.append((name, offset, used_fallback, reason))

    dropout = core_dropout_mask(child_log, len(data.time_s))
    eval_masks = {
        "unmasked": data.eval_mask,
        "dropout_masked": data.eval_mask & ~dropout,
    }
    rows: list[dict[str, Any]] = []
    residuals: dict[str, np.ndarray] = {}
    for variant, offset, used_fallback, reason in variants:
        corrected_prediction = nuisance.corrected_travel_mm + offset
        final_prediction, final_success = run_solver(
            raw,
            corrected_prediction,
            nuisance.corrected_mag_mg,
            nuisance.corrected_baseline_mg,
            first_pass=False,
            max_nfev=max_nfev,
        )
        for stage, prediction in (
            ("corrected_observation", corrected_prediction),
            ("final_solver", final_prediction),
        ):
            for evaluation, mask in eval_masks.items():
                row, error = metric_row(
                    data,
                    prediction,
                    parent_log=parent_log,
                    variant=variant,
                    stage=stage,
                    evaluation=evaluation,
                    mask=mask,
                    offset_mm=offset,
                    used_fallback=used_fallback,
                    fallback_reason=reason,
                    final_success=final_success,
                )
                rows.append(row)
                residuals[f"{variant}|{stage}|{evaluation}"] = error
    return rows, residuals


def aggregate(
    rows: Sequence[dict[str, Any]],
    residuals: dict[str, list[np.ndarray]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for variant in sorted({str(row["variant"]) for row in rows}):
        for stage in sorted({str(row["stage"]) for row in rows}):
            for evaluation in sorted({str(row["evaluation"]) for row in rows}):
                selected = [
                    row
                    for row in rows
                    if row["variant"] == variant
                    and row["stage"] == stage
                    and row["evaluation"] == evaluation
                ]
                if not selected:
                    continue
                parents = sorted({str(row["parent_log"]) for row in selected})
                parent_rmse = [
                    float(
                        np.mean(
                            [
                                float(row["rmse_mm"])
                                for row in selected
                                if row["parent_log"] == parent
                            ]
                        )
                    )
                    for parent in parents
                ]
                key = f"{variant}|{stage}|{evaluation}"
                errors = residuals[key]
                pooled = np.concatenate(errors)
                centered = np.concatenate([error - np.mean(error) for error in errors])
                output.append(
                    {
                        "variant": variant,
                        "stage": stage,
                        "evaluation": evaluation,
                        "n_children": len(selected),
                        "n_parents": len(parents),
                        "fallbacks": int(sum(bool(row["used_fallback"]) for row in selected)),
                        "child_mean_rmse_mm": float(
                            np.mean([float(row["rmse_mm"]) for row in selected])
                        ),
                        "parent_balanced_rmse_mm": float(np.mean(parent_rmse)),
                        "pooled_rmse_mm": rmse(pooled),
                        "pooled_child_centered_rmse_mm": rmse(centered),
                        "pooled_mean_error_mm": float(np.mean(pooled)),
                    }
                )
    return output


def fmt(value: object) -> str:
    return "—" if not np.isfinite(float(value)) else f"{float(value):.2f}"


def write_report(
    output_dir: Path,
    parent_rows: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    aggregates: Sequence[dict[str, Any]],
) -> None:
    final = {
        (str(row["variant"]), str(row["evaluation"])): row
        for row in aggregates
        if row["stage"] == "final_solver"
    }
    lines = [
        "# Slayer filtered-log corrected parent-reference experiment",
        "",
        "This rerun preserves the earlier manual strategy—one absolute magnetic reference from each complete parent log reused by its filtered child chunks—but recalculates that reference after nuisance correction. The detector uses corrected magnitude, a 0.20-second bump window, the existing magnetic-band selector, `bump_mag_min=1000 mG`, and the Slayer 20-point/40-mm safety gates. Both solvers and nuisance correction are rerun from the same zero-offset starting point for every policy.",
        "",
        "## Final adjusted-travel results",
        "",
        "| Reference policy | Evaluation | Fallbacks | Child-mean RMSE | Parent-balanced RMSE | Pooled RMSE | Pooled child-centered RMSE | Pooled mean error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "registry_old_parent": "Existing registry parent reference",
        "child_automatic": "Automatic per child",
        "recomputed_parent": "Recomputed corrected parent reference",
    }
    for variant in ("registry_old_parent", "child_automatic", "recomputed_parent"):
        for evaluation in ("unmasked", "dropout_masked"):
            row = final[(variant, evaluation)]
            lines.append(
                f"| {labels[variant]} | {evaluation.replace('_', ' ')} | {row['fallbacks']}/{row['n_children']} | "
                f"{fmt(row['child_mean_rmse_mm'])} | {fmt(row['parent_balanced_rmse_mm'])} | "
                f"{fmt(row['pooled_rmse_mm'])} | {fmt(row['pooled_child_centered_rmse_mm'])} | "
                f"{fmt(row['pooled_mean_error_mm'])} |"
            )

    old = final[("registry_old_parent", "unmasked")]
    automatic = final[("child_automatic", "unmasked")]
    recomputed = final[("recomputed_parent", "unmasked")]
    lines.extend(
        [
            "",
            "## Conclusions",
            "",
            f"- Recomputing parent references in corrected coordinates lowers pooled RMSE from {old['pooled_rmse_mm']:.2f} to {recomputed['pooled_rmse_mm']:.2f} mm, but child-mean RMSE slightly worsens from {old['child_mean_rmse_mm']:.2f} to {recomputed['child_mean_rmse_mm']:.2f} mm. It is not a reliable replacement for the old manual values.",
            f"- Automatic corrected-coordinate calibration per child is substantially better: {automatic['child_mean_rmse_mm']:.2f} mm child-mean, {automatic['parent_balanced_rmse_mm']:.2f} mm parent-balanced, and {automatic['pooled_rmse_mm']:.2f} mm pooled RMSE.",
            "- Only `log-0145` has a parent reference supported by at least 20 selected points. The other four parents fall back to their own corrected-magnitude p8 zero; `log-0151` is the nearest miss with 17 points.",
            "- The two `log-0147` children require very different local offsets. That branch instability is why one shared parent reference cannot solve both chunks.",
            "- Production action: remove the seven stale fixed-reference overrides and let each filtered child use the guarded post-correction policy. Keep the fixed raw baselines and shared accelerometer rotation.",
        ]
    )

    lines.extend(
        [
            "",
            "## Recomputed parent references",
            "",
            "| Parent | Source | Reference x | Corrected mag | Chunks | Selected points | Reference error vs GT |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(parent_rows, key=lambda item: str(item["parent_log"])):
        lines.append(
            f"| {row['parent_log']} | {row['source']} | {fmt(row['reference_x_mm'])} mm | "
            f"{fmt(row['reference_mag_mg'])} mG | {row['n_chunks']} | {row['n_ref_points']} | "
            f"{fmt(row['reference_error_mm'])} mm |"
        )

    lines.extend(
        [
            "",
            "## Per-child final solver (unmasked)",
            "",
            "| Child | Parent | Existing RMSE / mean error | Automatic RMSE / mean error | Recomputed-parent RMSE / mean error |",
            "|---|---|---:|---:|---:|",
        ]
    )
    child_rows = [
        row
        for row in rows
        if row["stage"] == "final_solver" and row["evaluation"] == "unmasked"
    ]
    for child in sorted({str(row["log"]) for row in child_rows}):
        by_variant = {
            str(row["variant"]): row for row in child_rows if row["log"] == child
        }
        old = by_variant["registry_old_parent"]
        automatic = by_variant["child_automatic"]
        parent = by_variant["recomputed_parent"]
        lines.append(
            f"| {child} | {old['parent_log']} | {fmt(old['rmse_mm'])} / {float(old['mean_error_mm']):+.2f} | "
            f"{fmt(automatic['rmse_mm'])} / {float(automatic['mean_error_mm']):+.2f} | "
            f"{fmt(parent['rmse_mm'])} / {float(parent['mean_error_mm']):+.2f} |"
        )
    lines.extend(
        [
            "",
            "The prior grown-chunk artifact reported 23.55 mm pooled absolute RMSE and 5.70 mm pooled chunk-centered RMSE. Its magnetic model/reference application occurred before nuisance correction, so the current-policy reruns above are the meaningful absolute-error comparison; centered results should remain close because all three reference policies primarily change a scalar offset.",
            "",
            "Detailed results are in `per_child_metrics.csv`, `aggregate_metrics.csv`, and `parent_references.csv`.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--cache-root", type=Path, default=REPO_ROOT / "backend" / "run_artifacts"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = LogRegistry.load(args.registry)
    children = [
        record.log_id
        for record in registry.select(set_name="slayer-filtered", usable_only=True)
    ]
    parent_by_child = {
        child: str(registry.resolve(child).metadata["parent_log"]) for child in children
    }
    parents = sorted(set(parent_by_child.values()))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parent_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                calculate_parent_reference,
                str(args.registry),
                str(args.cache_root),
                parent,
                args.max_nfev,
            ): parent
            for parent in parents
        }
        for future in as_completed(futures):
            parent = futures[future]
            parent_rows.append(future.result())
            print(f"parent complete: {parent}", flush=True)
    references = {
        str(row["parent_log"]): (
            float(row["reference_x_mm"]),
            float(row["reference_mag_mg"]),
        )
        for row in parent_rows
    }

    rows: list[dict[str, Any]] = []
    residuals: dict[str, list[np.ndarray]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_child,
                str(args.registry),
                str(args.cache_root),
                child,
                references[parent_by_child[child]],
                args.max_nfev,
            ): child
            for child in children
        }
        for future in as_completed(futures):
            child = futures[future]
            child_rows, child_residuals = future.result()
            rows.extend(child_rows)
            for key, error in child_residuals.items():
                residuals.setdefault(key, []).append(error)
            print(f"child complete: {child}", flush=True)

    parent_rows.sort(key=lambda row: str(row["parent_log"]))
    rows.sort(
        key=lambda row: (
            str(row["log"]),
            str(row["variant"]),
            str(row["stage"]),
            str(row["evaluation"]),
        )
    )
    aggregates = aggregate(rows, residuals)
    write_csv(args.output_dir / "parent_references.csv", parent_rows)
    write_csv(args.output_dir / "per_child_metrics.csv", rows)
    write_csv(args.output_dir / "aggregate_metrics.csv", aggregates)
    write_report(args.output_dir, parent_rows, rows, aggregates)
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
