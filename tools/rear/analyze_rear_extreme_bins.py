#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import numpy as np
import scipy.stats

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mag_to_travel_model_core import MagToTravelChunk, MagToTravelModel  # noqa: E402
from rear_mag_model import RearMagModel  # noqa: E402


TRAVEL_BIN_EDGES = np.linspace(0.0, 150.0, 6)
TRAVEL_BIN_LABELS = ["0-30", "30-60", "60-90", "90-120", "120-150"]
ANGLE_ERROR_HALO_S = 0.08


Row = dict[str, object]


@dataclass
class LogData:
    name: str
    t: np.ndarray
    travel: np.ndarray
    mag: np.ndarray
    accel_raw: np.ndarray
    accel_corr: np.ndarray
    mag_pred_adj: np.ndarray
    boring_mask: np.ndarray
    valid_mask: np.ndarray
    zv_points: np.ndarray
    coeffs: np.ndarray


def flatten_1d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr.reshape(-1)


def read_csv(path: Path) -> list[Row]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[Row]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def project_bad_angle_mask(source_t: np.ndarray, source_mask: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    source_t = flatten_1d(source_t)
    source_mask = np.asarray(source_mask).astype(bool).reshape(-1)
    target_t = flatten_1d(target_t)
    projected = np.zeros(len(target_t), dtype=bool)
    if len(source_t) == 0 or not np.any(source_mask):
        return projected

    bad_idx = np.flatnonzero(source_mask)
    split_idx = np.where(np.diff(bad_idx) > 1)[0]
    run_starts = np.r_[bad_idx[0], bad_idx[split_idx + 1]]
    run_ends = np.r_[bad_idx[split_idx], bad_idx[-1]]
    for start_idx, end_idx in zip(run_starts, run_ends):
        start_t = source_t[start_idx] - ANGLE_ERROR_HALO_S
        end_t = source_t[end_idx] + ANGLE_ERROR_HALO_S
        start = np.searchsorted(target_t, start_t, side="left")
        end = np.searchsorted(target_t, end_t, side="right")
        projected[start:end] = True
    return projected


def load_log(log_name: str) -> LogData:
    ws = np.load(REPO_ROOT / "backend" / "run_artifacts" / log_name / "cache" / "all.npz")
    t = flatten_1d(ws["travel__t"])
    travel = flatten_1d(ws["travel__x"])
    mag = flatten_1d(ws["mag/proj/lpf__x"])
    accel_raw = flatten_1d(ws["accel/lphp/proj__x"])
    accel_corr = flatten_1d(ws["accel/lphp/proj/zv__x"])
    mag_pred_adj = flatten_1d(ws["travel/mag_model/adj__x"])
    boring_mask = np.asarray(ws["boring_mask"]).astype(bool).reshape(-1)

    angle_bad = np.zeros(len(t), dtype=bool)
    if "angle/bad_mask__x" in ws and "angle/bad_mask__t" in ws:
        angle_bad = project_bad_angle_mask(ws["angle/bad_mask__t"], ws["angle/bad_mask__x"], t)

    valid_mask = (
        boring_mask
        & np.isfinite(travel)
        & np.isfinite(mag)
        & np.isfinite(accel_corr)
        & np.isfinite(mag_pred_adj)
        & ~angle_bad
    )
    zv_key = "mag_zv_points/accel_corr" if "mag_zv_points/accel_corr" in ws else "mag_zv_points"
    return LogData(
        name=log_name,
        t=t,
        travel=travel,
        mag=mag,
        accel_raw=accel_raw,
        accel_corr=accel_corr,
        mag_pred_adj=mag_pred_adj,
        boring_mask=boring_mask,
        valid_mask=valid_mask,
        zv_points=np.asarray(ws[zv_key], dtype=int).reshape(-1),
        coeffs=np.asarray(ws["mag_model_coeffs"], dtype=float).reshape(3),
    )


def travel_bin_masks(travel: np.ndarray) -> list[np.ndarray]:
    masks = []
    for idx, (lo, hi) in enumerate(zip(TRAVEL_BIN_EDGES[:-1], TRAVEL_BIN_EDGES[1:])):
        if idx == len(TRAVEL_BIN_EDGES) - 2:
            masks.append((travel >= lo) & (travel <= hi))
        else:
            masks.append((travel >= lo) & (travel < hi))
    return masks


def centered_error(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pred_masked = pred[mask]
    gt_masked = gt[mask]
    return (pred_masked - np.mean(pred_masked)) - (gt_masked - np.mean(gt_masked))


def safe_corr(x: np.ndarray, y: np.ndarray, *, spearman: bool = False) -> float:
    x = flatten_1d(x)
    y = flatten_1d(y)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    if spearman:
        return float(scipy.stats.spearmanr(x, y).correlation)
    return float(np.corrcoef(x, y)[0, 1])


def regression_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = flatten_1d(x)
    y = flatten_1d(y)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or np.ptp(x) <= 1e-9:
        return float("nan")
    return float(scipy.stats.linregress(x, y).slope)


def model_slope_dx_dmag(coeffs: np.ndarray, mag: np.ndarray) -> np.ndarray:
    x0, y_scale, power = coeffs
    pred_soft_mg = RearMagModel.pred_soft_mg
    dx = np.asarray(mag, dtype=float) - x0
    return y_scale * power * (np.abs(dx) + pred_soft_mg) ** (power - 1.0)


def dense_index_mask(length: int, indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    idx = np.asarray(indices, dtype=int).reshape(-1)
    idx = idx[(idx >= 0) & (idx < length)]
    mask[idx] = True
    return mask


def reconstruct_chunks(log: LogData) -> tuple[list[MagToTravelChunk], list[MagToTravelChunk]]:
    model = RearMagModel(
        x0_weight=0.0,
        chunking_method="centered_zv",
    )
    chunks = model.create_chunks(log.zv_points, log.mag, log.accel_corr, log.t)
    with contextlib.redirect_stdout(io.StringIO()):
        model.prepare_chunks(chunks)
        model.calc_chunks_errors(
            chunks,
            log.travel,
            np.gradient(log.travel, log.t, edge_order=2),
            np.gradient(np.gradient(log.travel, log.t, edge_order=2), log.t, edge_order=2),
        )
    filtered = model.filter_chunks(chunks, model.get_filter_fns())
    return chunks, filtered


def chunk_center_indices(chunks: list[MagToTravelChunk]) -> np.ndarray:
    return np.asarray([chunk.slice_i.start + chunk.zv_idx for chunk in chunks], dtype=int)


def chunk_bin_counts(chunks: list[MagToTravelChunk], log: LogData) -> list[int]:
    if not chunks:
        return [0] * len(TRAVEL_BIN_LABELS)
    centers = chunk_center_indices(chunks)
    center_travel = log.travel[centers]
    return [int(np.sum(mask)) for mask in travel_bin_masks(center_travel)]


def chunk_metrics_for_mask(chunks: list[MagToTravelChunk], mask: np.ndarray) -> dict[str, float]:
    if not chunks or not np.any(mask):
        return {
            "chunk_count": 0,
            "chunk_dx_median": float("nan"),
            "chunk_dx_p90": float("nan"),
            "chunk_abs_b_x_corr_median": float("nan"),
            "chunk_x_err_median": float("nan"),
            "chunk_x_err_p90": float("nan"),
            "chunk_fit_resid_median": float("nan"),
        }
    subset = [chunk for chunk, keep in zip(chunks, mask) if keep]
    dx = np.asarray([chunk.metrics["dx"] for chunk in subset], dtype=float)
    abs_corr = np.asarray([chunk.metrics["abs_b_x_corr"] for chunk in subset], dtype=float)
    x_err = np.asarray([np.median(np.abs(chunk.errors["x"])) for chunk in subset], dtype=float)
    return {
        "chunk_count": int(len(subset)),
        "chunk_dx_median": float(np.nanmedian(dx)),
        "chunk_dx_p90": float(np.nanpercentile(dx, 90.0)),
        "chunk_abs_b_x_corr_median": float(np.nanmedian(abs_corr)),
        "chunk_x_err_median": float(np.nanmedian(x_err)),
        "chunk_x_err_p90": float(np.nanpercentile(x_err, 90.0)),
    }


def make_bin_rows(log: LogData, all_chunks: list[MagToTravelChunk], filtered_chunks: list[MagToTravelChunk]) -> list[Row]:
    err_all = centered_error(log.mag_pred_adj, log.travel, log.valid_mask)
    gt_roi = log.travel[log.valid_mask]
    pred_roi = log.mag_pred_adj[log.valid_mask]
    mag_roi = log.mag[log.valid_mask]
    raw_bias = (log.accel_raw - log.accel_corr)[log.valid_mask]
    zv_mask = dense_index_mask(len(log.travel), log.zv_points)[log.valid_mask]
    bin_masks = travel_bin_masks(gt_roi)

    all_centers = chunk_center_indices(all_chunks)
    filt_centers = chunk_center_indices(filtered_chunks)
    all_center_travel = log.travel[all_centers] if len(all_centers) else np.asarray([])
    filt_center_travel = log.travel[filt_centers] if len(filt_centers) else np.asarray([])
    all_chunk_bin_masks = travel_bin_masks(all_center_travel)
    filt_chunk_bin_masks = travel_bin_masks(filt_center_travel)

    rows: list[Row] = []
    for bin_idx, label in enumerate(TRAVEL_BIN_LABELS):
        mask = bin_masks[bin_idx]
        count = int(np.sum(mask))
        if count:
            bin_err = err_all[mask]
            bin_mag = mag_roi[mask]
            bin_travel = gt_roi[mask]
            model_slope = model_slope_dx_dmag(log.coeffs, bin_mag)
            empirical_slope = regression_slope(bin_mag, bin_travel)
            pred_slope = regression_slope(bin_mag, pred_roi[mask])
            mag_span = float(np.percentile(bin_mag, 95.0) - np.percentile(bin_mag, 5.0))
            travel_span = float(np.percentile(bin_travel, 95.0) - np.percentile(bin_travel, 5.0))
            row: Row = {
                "log": log.name,
                "bin_idx": bin_idx,
                "bin": label,
                "sample_count": count,
                "sample_pct": 100.0 * count / max(1, len(gt_roi)),
                "rmse": float(np.sqrt(np.mean(bin_err**2))),
                "mae": float(np.mean(np.abs(bin_err))),
                "mean_signed_error": float(np.mean(bin_err)),
                "median_signed_error": float(np.median(bin_err)),
                "mag_p05": float(np.percentile(bin_mag, 5.0)),
                "mag_p50": float(np.percentile(bin_mag, 50.0)),
                "mag_p95": float(np.percentile(bin_mag, 95.0)),
                "mag_span_p05_p95": mag_span,
                "travel_span_p05_p95": travel_span,
                "mag_travel_spearman": safe_corr(bin_mag, bin_travel, spearman=True),
                "empirical_dx_dmag": empirical_slope,
                "pred_dx_dmag": pred_slope,
                "model_dx_dmag_median": float(np.nanmedian(model_slope)),
                "model_to_empirical_slope_ratio": float(np.nanmedian(model_slope) / empirical_slope)
                if np.isfinite(empirical_slope) and abs(empirical_slope) > 1e-12
                else float("nan"),
                "zv_count": int(np.sum(zv_mask[mask])),
                "zv_per_1k_samples": 1000.0 * float(np.mean(zv_mask[mask])),
                "accel_bias_abs_p95": float(np.percentile(np.abs(raw_bias[mask]), 95.0)),
                "accel_bias_mean": float(np.mean(raw_bias[mask])),
            }
        else:
            row = {
                "log": log.name,
                "bin_idx": bin_idx,
                "bin": label,
                "sample_count": 0,
                "sample_pct": 0.0,
                "rmse": float("nan"),
            }

        all_metrics = chunk_metrics_for_mask(all_chunks, all_chunk_bin_masks[bin_idx])
        filt_metrics = chunk_metrics_for_mask(filtered_chunks, filt_chunk_bin_masks[bin_idx])
        row.update({f"all_{key}": value for key, value in all_metrics.items()})
        row.update({f"filtered_{key}": value for key, value in filt_metrics.items()})
        if int(row["all_chunk_count"]):
            row["chunk_survival_pct"] = 100.0 * int(row["filtered_chunk_count"]) / int(row["all_chunk_count"])
        else:
            row["chunk_survival_pct"] = float("nan")
        rows.append(row)
    return rows


def make_log_summary(
    log: LogData,
    bin_rows: list[Row],
    stats_summary: dict[str, Row],
    mag_adj_bins: dict[str, Row],
) -> Row:
    summary = stats_summary[log.name]
    bins = mag_adj_bins[log.name]
    row: Row = {
        "log": log.name,
        "overall_mag_bin_rmse": float(summary["mag_adj_bin_rmse"]),
        "overall_sample_rmse": float(summary.get("mag_adj_rmse", summary["mag_adj_bin_rmse"]))
        if "mag_adj_rmse" in summary
        else float("nan"),
        "coeff_x0": float(log.coeffs[0]),
        "coeff_y_scale": float(log.coeffs[1]),
        "coeff_power": float(log.coeffs[2]),
        "travel_max": float(np.nanmax(log.travel[log.valid_mask])),
        "travel_p99": float(np.nanpercentile(log.travel[log.valid_mask], 99.0)),
        "mag_span_p01_p99": float(
            np.nanpercentile(log.mag[log.valid_mask], 99.0)
            - np.nanpercentile(log.mag[log.valid_mask], 1.0)
        ),
        "zv_per_1k": 1000.0
        * float(np.mean(dense_index_mask(len(log.travel), log.zv_points)[log.valid_mask])),
    }
    for idx, label in enumerate(TRAVEL_BIN_LABELS):
        rmse = float(bins[f"bin{idx}_rmse"])
        row[f"bin{idx}_rmse"] = rmse
        row[f"bin{idx}_excess"] = rmse - float(summary["mag_adj_bin_rmse"])
        row[f"bin{idx}_ratio"] = rmse / float(summary["mag_adj_bin_rmse"])
        row[f"bin{idx}_samples"] = int(bin_rows[idx]["sample_count"])
        row[f"bin{idx}_sample_pct"] = float(bin_rows[idx]["sample_pct"])
        row[f"bin{idx}_filtered_chunks"] = int(bin_rows[idx]["filtered_chunk_count"])
        row[f"bin{idx}_chunk_survival_pct"] = float(bin_rows[idx]["chunk_survival_pct"])
        row[f"bin{idx}_signed_error"] = float(bin_rows[idx].get("mean_signed_error", float("nan")))
        row[f"bin{idx}_model_to_emp_slope"] = float(
            bin_rows[idx].get("model_to_empirical_slope_ratio", float("nan"))
        )
    return row


def dict_by_log(rows: list[Row]) -> dict[str, Row]:
    return {str(row["log"]): row for row in rows}


def rank_lines(rows: list[Row], key: str, *, reverse: bool = True, n: int = 5) -> list[str]:
    valid = [row for row in rows if np.isfinite(float(row.get(key, float("nan"))))]
    valid = sorted(valid, key=lambda row: float(row[key]), reverse=reverse)
    return [
        f"- `{row['log']}`: {key} `{float(row[key]):.2f}`"
        for row in valid[:n]
    ]


def fmt(value: Any, digits: int = 2) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_f):
        return "nan"
    return f"{value_f:.{digits}f}"


def write_report(path: Path, log_rows: list[Row], bin_rows: list[Row]) -> None:
    high_worst = sorted(log_rows, key=lambda row: float(row["bin4_excess"]), reverse=True)
    low_worst = sorted(log_rows, key=lambda row: float(row["bin0_excess"]), reverse=True)
    overall_best = sorted(log_rows, key=lambda row: float(row["overall_mag_bin_rmse"]))
    overall_worst = sorted(log_rows, key=lambda row: float(row["overall_mag_bin_rmse"]), reverse=True)

    lines = [
        "# Rear Mag Model Extreme-Bin Deep Dive",
        "",
        "This analysis compares mag-model errors against raw/cache-level signals and reconstructed training chunks.",
        "",
        "## Headline",
        "",
        (
            "- High-travel failures are mostly endpoint-support failures: very little time above 120 mm, "
            "very few filtered training chunks centered there, and often a learned curve that is too steep "
            "for the empirical high-travel mag slope."
        ),
        (
            "- Low-travel failures are less about missing samples and more about the learned zero-end shape: "
            "bad low bins usually have plenty of samples, but the selected chunks and fitted curve create a "
            "signed offset at the bottom of travel."
        ),
        (
            "- The solver only weakly changes these patterns, so the mag-model learning/coverage problem is "
            "visible before solver fusion."
        ),
        "",
        "## Worst High-Travel Excess",
        "",
        "| Log | Overall bin RMSE | 120-150 RMSE | Excess | Samples | Filtered chunks | Signed err | Slope ratio | Chunk survival |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in high_worst[:6]:
        lines.append(
            f"| `{row['log']}` | {fmt(row['overall_mag_bin_rmse'])} | {fmt(row['bin4_rmse'])} | "
            f"{fmt(row['bin4_excess'])} | {int(row['bin4_samples'])} | "
            f"{int(row['bin4_filtered_chunks'])} | {fmt(row['bin4_signed_error'])} | "
            f"{fmt(row['bin4_model_to_emp_slope'])} | {fmt(row['bin4_chunk_survival_pct'])}% |"
        )

    lines.extend(
        [
            "",
            "## Worst Low-Travel Excess",
            "",
            "| Log | Overall bin RMSE | 0-30 RMSE | Excess | Samples | Filtered chunks | Signed err | Slope ratio | Chunk survival |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in low_worst[:6]:
        lines.append(
            f"| `{row['log']}` | {fmt(row['overall_mag_bin_rmse'])} | {fmt(row['bin0_rmse'])} | "
            f"{fmt(row['bin0_excess'])} | {int(row['bin0_samples'])} | "
            f"{int(row['bin0_filtered_chunks'])} | {fmt(row['bin0_signed_error'])} | "
            f"{fmt(row['bin0_model_to_emp_slope'])} | {fmt(row['bin0_chunk_survival_pct'])}% |"
        )

    lines.extend(
        [
            "",
            "## Best Overall Logs",
            "",
            "| Log | Overall bin RMSE | 0-30 | 120-150 | Low chunks | High chunks | Power | y_scale |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in overall_best[:6]:
        lines.append(
            f"| `{row['log']}` | {fmt(row['overall_mag_bin_rmse'])} | {fmt(row['bin0_rmse'])} | "
            f"{fmt(row['bin4_rmse'])} | {int(row['bin0_filtered_chunks'])} | "
            f"{int(row['bin4_filtered_chunks'])} | {fmt(row['coeff_power'], 3)} | "
            f"{fmt(row['coeff_y_scale'])} |"
        )

    lines.extend(
        [
            "",
            "## Worst Overall Logs",
            "",
            "| Log | Overall bin RMSE | 0-30 | 120-150 | Low chunks | High chunks | Power | y_scale |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in overall_worst[:6]:
        lines.append(
            f"| `{row['log']}` | {fmt(row['overall_mag_bin_rmse'])} | {fmt(row['bin0_rmse'])} | "
            f"{fmt(row['bin4_rmse'])} | {int(row['bin0_filtered_chunks'])} | "
            f"{int(row['bin4_filtered_chunks'])} | {fmt(row['coeff_power'], 3)} | "
            f"{fmt(row['coeff_y_scale'])} |"
        )

    high_bin_rows = [row for row in bin_rows if int(row["bin_idx"]) == 4]
    low_bin_rows = [row for row in bin_rows if int(row["bin_idx"]) == 0]

    def correlation_line(label: str, rows: list[Row], x_key: str, y_key: str = "rmse") -> str:
        x = np.asarray([float(row.get(x_key, float("nan"))) for row in rows], dtype=float)
        y = np.asarray([float(row.get(y_key, float("nan"))) for row in rows], dtype=float)
        corr = safe_corr(x, y, spearman=True)
        return f"- `{label}` Spearman `{x_key}` vs `{y_key}`: `{fmt(corr, 3)}`"

    lines.extend(
        [
            "",
            "## Correlation Clues",
            "",
            correlation_line("high bin", high_bin_rows, "filtered_chunk_count"),
            correlation_line("high bin", high_bin_rows, "sample_count"),
            correlation_line("high bin", high_bin_rows, "model_to_empirical_slope_ratio"),
            correlation_line("high bin", high_bin_rows, "chunk_survival_pct"),
            correlation_line("low bin", low_bin_rows, "filtered_chunk_count"),
            correlation_line("low bin", low_bin_rows, "sample_count"),
            correlation_line("low bin", low_bin_rows, "model_to_empirical_slope_ratio"),
            correlation_line("low bin", low_bin_rows, "chunk_survival_pct"),
            "",
            "## Notes",
            "",
            (
                "- `Slope ratio` is learned model `dx/dmag` divided by empirical local `dx/dmag`. "
                "Values above 1 mean the learned curve is steeper than the observed mag/travel cloud in that bin."
            ),
            (
                "- `Signed err` is centered prediction error in the bin. Positive means predicted travel is high "
                "relative to GT after centering; negative means low."
            ),
            (
                "- `Filtered chunks` counts chunks centered in that travel bin after the same `RearMagModel` dx filter "
                "used by the pipeline."
            ),
            "",
            "Full tables:",
            "",
            "- `log_summary.csv`",
            "- `bin_metrics.csv`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep-dive rear mag-model errors at travel extrema.")
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path("reports/stats_aggregator/rear/rear_all_4hz_accel"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/rear_extreme_bin_deep_dive"),
    )
    parser.add_argument("--logs", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_rows = dict_by_log(read_csv(args.stats_dir / "diagnostics_binned_summary.csv"))
    mag_adj_bins = dict_by_log(read_csv(args.stats_dir / "diagnostics_mag_adj_bins.csv"))
    logs = args.logs or list(summary_rows.keys())

    all_log_rows: list[Row] = []
    all_bin_rows: list[Row] = []
    for log_name in logs:
        print(f"Analyzing {log_name}", flush=True)
        log = load_log(log_name)
        all_chunks, filtered_chunks = reconstruct_chunks(log)
        bin_rows = make_bin_rows(log, all_chunks, filtered_chunks)
        all_bin_rows.extend(bin_rows)
        all_log_rows.append(make_log_summary(log, bin_rows, summary_rows, mag_adj_bins))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "bin_metrics.csv", all_bin_rows)
    write_csv(args.out_dir / "log_summary.csv", all_log_rows)
    write_report(args.out_dir / "report.md", all_log_rows, all_bin_rows)
    print(f"Wrote {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
