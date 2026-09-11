#!/usr/bin/env python3
"""Validate front magnetic offset estimators on legacy cached logs only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.log_registry import DEFAULT_REGISTRY_PATH, LogRegistry  # noqa: E402
from tools.front.mag_offset_calibration.explore_mag_offset_calibration import (  # noqa: E402
    find_ref_chunks,
    oob_offset,
    reference_diagnostics,
)
from tools.front.mag_offset_calibration.sweep_post_mag_correction_fusion import (  # noqa: E402
    rmse,
    write_csv,
)


METHODS = (
    "current_cached",
    "pipeline_p8_to_zero",
    "prediction_p2_to_zero",
    "oob_p1_p99_nearest_zero",
    "fixed_current_p2_blend",
    "chunk_confidence_n10_cap0.5",
    "current_if_25_chunks_else_confidence",
)


def select_mag_key(keys: set[str]) -> str | None:
    for key in (
        "mag/norm/corr/lpf__x",
        "mag/proj/corr/lpf__x",
        "mag/proj/lpf__x",
    ):
        if key in keys:
            return key
    return None


def flatten(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def discover_legacy_logs(
    registry_path: Path, cache_root: Path
) -> list[tuple[str, Path]]:
    registry = LogRegistry.load(registry_path)
    required = {
        "travel__t",
        "travel__x",
        "accel/lpfhp/proj__x",
        "travel/mag_model__x",
        "travel/mag_model/adj__x",
        "mag_baseline",
        "boring_mask",
    }
    selected: list[tuple[str, Path]] = []
    for log in registry.select(filters={"pipeline": "front"}, usable_only=True):
        # The modern registry cohort uses dashed IDs; the legacy front archive
        # uses IDs such as log022 and log056_ccdh.
        if log.log_id.startswith("log-") or log.sets:
            continue
        cache_path = cache_root / log.log_id / "cache" / "all.npz"
        if not cache_path.exists():
            continue
        with np.load(cache_path, allow_pickle=False) as cache:
            keys = set(cache.files)
            mag_key = select_mag_key(keys)
            if required <= keys and mag_key is not None:
                selected.append((log.log_id, cache_path))
    return sorted(selected)


def candidate_offsets(
    raw_prediction: np.ndarray,
    current_offset: float,
    good_mask: np.ndarray,
    chunk_count: int,
) -> dict[str, float]:
    good_prediction = raw_prediction[good_mask & np.isfinite(raw_prediction)]
    p2_offset = -float(np.percentile(good_prediction, 2.0))
    p8_offset = -float(np.percentile(good_prediction, 8.0))
    oob = oob_offset(
        good_prediction,
        lower_percentile=1.0,
        upper_percentile=99.0,
        tie_break="nearest-zero",
    )
    confidence = min(chunk_count / 10.0, 0.5)
    confidence_offset = (
        confidence * current_offset + (1.0 - confidence) * p2_offset
    )
    return {
        "current_cached": current_offset,
        "pipeline_p8_to_zero": p8_offset,
        "prediction_p2_to_zero": p2_offset,
        "oob_p1_p99_nearest_zero": oob,
        "fixed_current_p2_blend": 0.5 * current_offset + 0.5 * p2_offset,
        "chunk_confidence_n10_cap0.5": confidence_offset,
        "current_if_25_chunks_else_confidence": (
            current_offset if chunk_count >= 25 else confidence_offset
        ),
    }


def aggregate(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "offset_mm",
        "offset_error_mm",
        "abs_offset_error_mm",
        "mean_error_mm",
        "uncentered_rmse_mm",
        "low_uncentered_rmse_mm",
    )
    output = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        output.append(
            {
                "method": method,
                "n_logs": len(selected),
                **{
                    metric: float(np.mean([float(row[metric]) for row in selected]))
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs = discover_legacy_logs(args.registry, args.cache_root)
    if not logs:
        raise RuntimeError("no eligible legacy front caches found")

    rows: list[dict[str, object]] = []
    chunk_rows: list[dict[str, object]] = []
    for log_name, cache_path in logs:
        with np.load(cache_path, allow_pickle=False) as cache:
            keys = set(cache.files)
            mag_key = select_mag_key(keys)
            if mag_key is None:
                raise KeyError(f"{log_name}: no scalar magnetic model input")
            time_s = flatten(cache["travel__t"])
            truth = flatten(cache["travel__x"])
            mag = flatten(cache[mag_key])
            accel = flatten(cache["accel/lpfhp/proj__x"])
            raw = flatten(cache["travel/mag_model__x"])
            adjusted = flatten(cache["travel/mag_model/adj__x"])
            eval_mask = np.asarray(cache["boring_mask"], dtype=bool).reshape(-1)
            baseline = float(flatten(cache["mag_baseline"])[0])
            if "mag/norm/bad_mask__x" in keys:
                good_mask = ~np.asarray(
                    cache["mag/norm/bad_mask__x"], dtype=bool
                ).reshape(-1)
            elif "mag/proj/bad_mask__x" in keys:
                good_mask = ~np.asarray(
                    cache["mag/proj/bad_mask__x"], dtype=bool
                ).reshape(-1)
            else:
                good_mask = np.isfinite(mag)

        lengths = {len(time_s), len(truth), len(mag), len(accel), len(raw), len(adjusted), len(eval_mask)}
        if len(lengths) != 1:
            raise ValueError(f"{log_name}: unaligned legacy cache arrays {sorted(lengths)}")
        eval_mask &= np.isfinite(truth) & np.isfinite(raw)
        if not np.any(eval_mask):
            raise ValueError(f"{log_name}: empty evaluation mask")

        applied_delta = adjusted - raw
        current_offset = float(np.median(applied_delta[np.isfinite(applied_delta)]))
        offset_nonconstant_max_mm = float(
            np.max(np.abs(applied_delta - current_offset))
        )
        fs_hz = 1.0 / float(np.median(np.diff(time_s)))
        chunks = find_ref_chunks(
            mag,
            accel,
            truth,
            time_s,
            baseline,
            fs_hz=fs_hz,
        )
        starts = np.asarray([chunk.start_truth for chunk in chunks], dtype=float)
        chunk_rows.append(
            {
                "log": log_name,
                "mag_key": mag_key,
                "n_chunks": len(chunks),
                "fraction_chunks_start_le_3mm": (
                    float(np.mean(starts <= 3.0)) if len(starts) else 0.0
                ),
                "median_chunk_start_truth_mm": (
                    float(np.median(starts)) if len(starts) else float("nan")
                ),
                **reference_diagnostics(chunks, baseline),
                "cached_offset_nonconstant_max_mm": offset_nonconstant_max_mm,
            }
        )

        oracle_offset = float(np.mean(truth[eval_mask] - raw[eval_mask]))
        low_mask = eval_mask & (truth >= 0.0) & (truth < 30.0)
        offsets = candidate_offsets(raw, current_offset, good_mask, len(chunks))
        for method, offset in offsets.items():
            prediction = raw + offset
            error = prediction - truth
            rows.append(
                {
                    "log": log_name,
                    "method": method,
                    "n_chunks": len(chunks),
                    "offset_mm": offset,
                    "oracle_offset_mm": oracle_offset,
                    "offset_error_mm": offset - oracle_offset,
                    "abs_offset_error_mm": abs(offset - oracle_offset),
                    "mean_error_mm": float(np.mean(error[eval_mask])),
                    "uncentered_rmse_mm": rmse(error[eval_mask]),
                    "low_uncentered_rmse_mm": (
                        rmse(error[low_mask]) if np.any(low_mask) else float("nan")
                    ),
                }
            )

    summaries = aggregate(rows)
    current = {
        str(row["log"]): float(row["uncentered_rmse_mm"])
        for row in rows
        if row["method"] == "current_cached"
    }
    wins = {
        method: sum(
            float(row["uncentered_rmse_mm"]) < current[str(row["log"])]
            for row in rows
            if row["method"] == method
        )
        for method in METHODS
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "legacy_chunk_diagnostics.csv", chunk_rows)
    write_csv(args.output_dir / "legacy_per_log_metrics.csv", rows)
    write_csv(args.output_dir / "legacy_aggregate_metrics.csv", summaries)
    payload = {
        "logs": [log_name for log_name, _ in logs],
        "n_logs": len(logs),
        "methods": list(METHODS),
        "wins_vs_current": wins,
    }
    (args.output_dir / "legacy_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
