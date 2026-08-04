#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAVEL_BIN_EDGES = np.linspace(0.0, 150.0, 6)
TRAVEL_BIN_LABELS = ["0-30", "30-60", "60-90", "90-120", "120-150"]
MIN_BIN_SAMPLES = 100
ANGLE_ERROR_HALO_S = 0.08


Row = dict[str, object]


def flatten_1d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr.reshape(-1)


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


def load_log(log_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ws = np.load(REPO_ROOT / "backend" / "run_artifacts" / log_name / "cache" / "all.npz")
    t = flatten_1d(ws["travel__t"])
    travel = flatten_1d(ws["travel__x"])
    pred = flatten_1d(ws["travel/mag_model/adj__x"])
    boring = np.asarray(ws["boring_mask"]).astype(bool).reshape(-1)
    angle_bad = np.zeros(len(t), dtype=bool)
    if "angle/bad_mask__x" in ws and "angle/bad_mask__t" in ws:
        angle_bad = project_bad_angle_mask(ws["angle/bad_mask__t"], ws["angle/bad_mask__x"], t)
    valid = boring & np.isfinite(travel) & np.isfinite(pred) & ~angle_bad
    return pred[valid], travel[valid], t[valid]


def travel_bin_masks(travel: np.ndarray) -> list[np.ndarray]:
    masks = []
    for idx, (lo, hi) in enumerate(zip(TRAVEL_BIN_EDGES[:-1], TRAVEL_BIN_EDGES[1:])):
        upper = travel <= hi if idx == len(TRAVEL_BIN_EDGES) - 2 else travel < hi
        masks.append((travel >= lo) & upper)
    return masks


def bin_mses(err: np.ndarray, travel: np.ndarray, *, min_count: int) -> tuple[float, list[float], list[int]]:
    mses: list[float] = []
    rmses: list[float] = []
    counts: list[int] = []
    for mask in travel_bin_masks(travel):
        count = int(np.sum(mask))
        counts.append(count)
        if count:
            mse = float(np.mean(err[mask] ** 2))
            rmses.append(float(np.sqrt(mse)))
            if count >= min_count:
                mses.append(mse)
        else:
            rmses.append(float("nan"))
    return (float(np.sqrt(np.mean(mses))) if mses else float("nan")), rmses, counts


def bin_means(values: np.ndarray, travel: np.ndarray, *, min_count: int) -> list[float]:
    means: list[float] = []
    for mask in travel_bin_masks(travel):
        if int(np.sum(mask)) >= min_count:
            means.append(float(np.mean(values[mask])))
    return means


def offset_for_method(pred: np.ndarray, travel: np.ndarray, method: str) -> float:
    residual = travel - pred
    if method == "sample_mean":
        return float(np.mean(residual))
    if method == "sample_median":
        return float(np.median(residual))
    if method == "bin_mean_eligible":
        means = bin_means(residual, travel, min_count=MIN_BIN_SAMPLES)
        return float(np.mean(means)) if means else float("nan")
    if method == "bin_mean_all":
        means = bin_means(residual, travel, min_count=1)
        return float(np.mean(means)) if means else float("nan")
    if method == "tail_mean":
        masks = travel_bin_masks(travel)
        selected = [masks[0], masks[-1]]
        means = [float(np.mean(residual[mask])) for mask in selected if np.any(mask)]
        return float(np.mean(means)) if means else float("nan")
    raise ValueError(f"Unknown centering method {method!r}")


def evaluate_method(log_name: str, pred: np.ndarray, travel: np.ndarray, method: str) -> Row:
    offset = offset_for_method(pred, travel, method)
    err = pred + offset - travel
    bin_rmse_eligible, rmses, counts = bin_mses(err, travel, min_count=MIN_BIN_SAMPLES)
    bin_rmse_all, _, _ = bin_mses(err, travel, min_count=1)
    row: Row = {
        "log": log_name,
        "method": method,
        "offset": offset,
        "sample_rmse": float(np.sqrt(np.mean(err**2))),
        "sample_mae": float(np.mean(np.abs(err))),
        "bin_rmse_eligible": bin_rmse_eligible,
        "bin_rmse_all": bin_rmse_all,
    }
    for idx, (label, rmse, count) in enumerate(zip(TRAVEL_BIN_LABELS, rmses, counts)):
        row[f"bin{idx}_label"] = label
        row[f"bin{idx}_rmse"] = rmse
        row[f"bin{idx}_n"] = count
        row[f"bin{idx}_mean_err"] = float(np.mean(err[travel_bin_masks(travel)[idx]])) if count else float("nan")
    return row


def bias_decomposition(log_name: str, pred: np.ndarray, travel: np.ndarray) -> list[Row]:
    offset = offset_for_method(pred, travel, "sample_mean")
    err = pred + offset - travel
    rows = []
    for idx, (label, mask) in enumerate(zip(TRAVEL_BIN_LABELS, travel_bin_masks(travel))):
        count = int(np.sum(mask))
        if not count:
            continue
        bin_err = err[mask]
        bias = float(np.mean(bin_err))
        centered = bin_err - bias
        rows.append(
            {
                "log": log_name,
                "bin_idx": idx,
                "bin": label,
                "n": count,
                "current_rmse": float(np.sqrt(np.mean(bin_err**2))),
                "bin_bias": bias,
                "within_bin_rmse": float(np.sqrt(np.mean(centered**2))),
                "bias_rmse_fraction": float((bias**2) / np.mean(bin_err**2)) if np.mean(bin_err**2) > 0 else float("nan"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[Row]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean_metric(rows: list[Row], method: str, key: str) -> float:
    vals = [float(row[key]) for row in rows if row["method"] == method and np.isfinite(float(row[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def write_report(path: Path, rows: list[Row], bias_rows: list[Row], methods: list[str]) -> None:
    lines = [
        "# Rear Mag Model Centering Effects",
        "",
        "This compares constant-offset choices for evaluating `travel/mag_model/adj` against GT travel.",
        "",
        "## Offset Methods",
        "",
        "- `sample_mean`: current centered metric; offset minimizes sample-weighted RMSE over the boring mask.",
        "- `sample_median`: robust global offset.",
        "- `bin_mean_eligible`: offset minimizes equal-bin RMSE over bins with at least 100 samples.",
        "- `bin_mean_all`: offset minimizes equal-bin RMSE over every observed bin, including sparse tails.",
        "- `tail_mean`: offset averages the 0-30 and 120-150 bin residual means.",
        "",
        "## Aggregate Metrics",
        "",
        "| Method | Mean sample RMSE | Mean eligible-bin RMSE | Mean all-bin RMSE | Mean 0-30 RMSE | Mean 120-150 RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        lines.append(
            f"| `{method}` | {mean_metric(rows, method, 'sample_rmse'):.2f} | "
            f"{mean_metric(rows, method, 'bin_rmse_eligible'):.2f} | "
            f"{mean_metric(rows, method, 'bin_rmse_all'):.2f} | "
            f"{mean_metric(rows, method, 'bin0_rmse'):.2f} | "
            f"{mean_metric(rows, method, 'bin4_rmse'):.2f} |"
        )

    current = [row for row in rows if row["method"] == "sample_mean"]
    balanced = {row["log"]: row for row in rows if row["method"] == "bin_mean_all"}
    lines.extend(
        [
            "",
            "## Biggest High-Tail Changes Under All-Bin Offset",
            "",
            "| Log | Current 120-150 | All-bin 120-150 | Delta | Current 0-30 | All-bin 0-30 | Offset delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    deltas = []
    for row in current:
        other = balanced[row["log"]]
        deltas.append((float(row["bin4_rmse"]) - float(other["bin4_rmse"]), row, other))
    for delta, row, other in sorted(deltas, key=lambda item: abs(item[0]), reverse=True)[:8]:
        lines.append(
            f"| `{row['log']}` | {float(row['bin4_rmse']):.2f} | {float(other['bin4_rmse']):.2f} | "
            f"{float(other['bin4_rmse']) - float(row['bin4_rmse']):+.2f} | "
            f"{float(row['bin0_rmse']):.2f} | {float(other['bin0_rmse']):.2f} | "
            f"{float(other['offset']) - float(row['offset']):+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Bias Decomposition Under Current Offset",
            "",
            "| Log | Bin | RMSE | Mean bias | Within-bin RMSE | Bias fraction |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    interesting = [row for row in bias_rows if int(row["bin_idx"]) in (0, 4)]
    interesting = sorted(interesting, key=lambda row: float(row["bias_rmse_fraction"]), reverse=True)
    for row in interesting[:14]:
        lines.append(
            f"| `{row['log']}` | `{row['bin']}` | {float(row['current_rmse']):.2f} | "
            f"{float(row['bin_bias']):+.2f} | {float(row['within_bin_rmse']):.2f} | "
            f"{100.0 * float(row['bias_rmse_fraction']):.0f}% |"
        )

    lines.extend(
        [
            "",
            "## Takeaway",
            "",
            (
                "The current centering is the optimal single offset for sample-weighted RMSE, but it is not "
                "aligned with an equal-bin metric. Dense mid-travel samples dominate the offset, so sparse "
                "tails can inherit a large bin bias."
            ),
            (
                "For shape evaluation independent of Y-offset, use an offset fitted with the same weighting "
                "as the metric: `bin_mean_eligible` for the current eligible-bin score, or `bin_mean_all` / "
                "a separate tail metric when sparse endpoints matter."
            ),
            "",
            "Full tables:",
            "",
            "- `centering_metrics.csv`",
            "- `bin_bias_decomposition.csv`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rear mag-model centering choices.")
    parser.add_argument("--logs", nargs="*", default=[f"log{i}_rear" for i in [140, 141, 142, 143, 144, 145, 148, 149, 150, 151, 152, 153, 154]])
    parser.add_argument("--out-dir", type=Path, default=Path("reports/rear_centering_effects"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = ["sample_mean", "sample_median", "bin_mean_eligible", "bin_mean_all", "tail_mean"]
    metric_rows: list[Row] = []
    bias_rows: list[Row] = []
    for log_name in args.logs:
        pred, travel, _ = load_log(log_name)
        for method in methods:
            metric_rows.append(evaluate_method(log_name, pred, travel, method))
        bias_rows.extend(bias_decomposition(log_name, pred, travel))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "centering_metrics.csv", metric_rows)
    write_csv(args.out_dir / "bin_bias_decomposition.csv", bias_rows)
    write_report(args.out_dir / "report.md", metric_rows, bias_rows, methods)
    print(f"Wrote {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
