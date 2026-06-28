#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mag_to_travel_model_core import MagToTravelModel  # noqa: E402


ANGLE_ERROR_HALO_S = 0.08


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_curve_data(log_name: str) -> dict[str, np.ndarray]:
    ws = np.load(REPO_ROOT / "backend" / "run_artifacts" / log_name / "cache" / "all.npz")
    t = flatten_1d(ws["travel__t"])
    travel = flatten_1d(ws["travel__x"])
    mag = flatten_1d(ws["mag/proj/lpf__x"])
    pred_adj = flatten_1d(ws["travel/mag_model/adj__x"])
    boring = np.asarray(ws["boring_mask"]).astype(bool).reshape(-1)

    angle_bad = np.zeros(len(t), dtype=bool)
    if "angle/bad_mask__x" in ws and "angle/bad_mask__t" in ws:
        angle_bad = project_bad_angle_mask(ws["angle/bad_mask__t"], ws["angle/bad_mask__x"], t)

    valid = boring & np.isfinite(travel) & np.isfinite(mag) & np.isfinite(pred_adj) & ~angle_bad
    return {
        "mag": mag,
        "travel": travel,
        "pred_adj": pred_adj,
        "valid": valid,
        "coeffs": np.asarray(ws["mag_model_coeffs"], dtype=float).reshape(3),
    }


def smooth_curve(mag: np.ndarray, pred_adj: np.ndarray, coeffs: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = MagToTravelModel(pred_soft_mg=50.0)
    model.set_coeffs(coeffs)
    finite = np.isfinite(mag) & np.isfinite(pred_adj)
    if not np.any(finite):
        raise ValueError("No finite mag/model samples available for curve plotting")

    valid_mag = mag[valid & np.isfinite(mag)]
    lo = float(np.min(valid_mag))
    hi = float(np.max(valid_mag))
    pad = 0.02 * max(hi - lo, 1.0)
    grid = np.linspace(lo - pad, hi + pad, 700)

    # Match the cached adjusted model exactly instead of recomputing the zero
    # percentile on a plotted subset.
    offset = float(np.median(model.pred_x(mag[finite]) - pred_adj[finite]))
    curve = model.pred_x(grid) - offset
    return grid, curve


def choose_logs(summary_path: Path, count: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = read_csv(summary_path)
    rows = [row for row in rows if np.isfinite(float(row["bin4_rmse"]))]
    worst = sorted(rows, key=lambda row: float(row["bin4_rmse"]), reverse=True)[:count]
    best = sorted(rows, key=lambda row: float(row["bin4_rmse"]))[:count]
    return worst, best


def scatter_indices(mask: np.ndarray, *, max_points: int, seed: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if len(idx) <= max_points:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=max_points, replace=False))


def stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def plot_log(ax: plt.Axes, row: dict[str, str], *, group: str, seed: int) -> None:
    data = load_curve_data(row["log"])
    mag = data["mag"]
    travel = data["travel"]
    valid = data["valid"]
    high = valid & (travel >= 120.0) & (travel <= 150.0)
    mid_low = valid & ~high

    idx = scatter_indices(mid_low, max_points=10000, seed=seed)
    ax.scatter(
        mag[idx],
        travel[idx],
        s=3,
        c="#9ca3af",
        alpha=0.18,
        linewidths=0,
        rasterized=True,
        label="GT travel",
    )
    high_idx = scatter_indices(high, max_points=2500, seed=seed + 1)
    ax.scatter(
        mag[high_idx],
        travel[high_idx],
        s=12,
        c="#dc2626",
        alpha=0.78,
        linewidths=0,
        rasterized=True,
        label="GT >=120 mm",
    )

    grid, curve = smooth_curve(mag, data["pred_adj"], data["coeffs"], valid)
    order = np.argsort(grid)
    ax.plot(grid[order], curve[order], c="#2563eb", lw=2.0, label="learned curve")
    ax.set_xlim(float(np.min(grid)), float(np.max(grid)))

    ax.set_title(
        (
            f"{group}: {row['log']}\n"
            f"120-150 RMSE {float(row['bin4_rmse']):.1f} mm, "
            f"n={int(float(row['bin4_samples']))}, chunks={int(float(row['bin4_filtered_chunks']))}"
        ),
        fontsize=10,
    )
    ax.set_xlabel("mag/proj/lpf (mG)")
    ax.set_ylabel("travel (mm)")
    ax.set_ylim(-5, 150)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def write_plot_note(path: Path, worst: list[dict[str, str]], best: list[dict[str, str]]) -> None:
    lines = [
        "# Rear Mag Curve Plots",
        "",
        "Plots overlay the learned mag-to-travel curve on `mag/proj/lpf` vs GT travel samples.",
        "Gray points are boring-mask GT samples; red points are GT travel >=120 mm.",
        (
            "The curve offset is matched to the cached `travel/mag_model/adj` signal, and the curve range "
            "spans the full boring-mask mag range so it includes the high-travel tail."
        ),
        "",
        "## Selected Logs",
        "",
        "| Group | Log | 120-150 RMSE | Samples | Filtered chunks |",
        "|---|---|---:|---:|---:|",
    ]
    for group, rows in (("Worst", worst), ("Best", best)):
        for row in rows:
            lines.append(
                f"| {group} | `{row['log']}` | {float(row['bin4_rmse']):.2f} | "
                f"{int(float(row['bin4_samples']))} | {int(float(row['bin4_filtered_chunks']))} |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot rear mag-model curves for high-travel best/worst logs.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/rear_extreme_bin_deep_dive/log_summary.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/rear_extreme_bin_deep_dive/curve_plots"),
    )
    parser.add_argument("--count", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worst, best = choose_logs(args.summary, args.count)
    selected = [("Worst", row) for row in worst] + [("Best", row) for row in best]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, args.count, figsize=(5.2 * args.count, 8.8), constrained_layout=True)
    if args.count == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for ax, (group, row) in zip(axes.ravel(), selected):
        plot_log(ax, row, group=group, seed=stable_seed(row["log"]))
    fig.suptitle("Rear Mag-To-Travel Learned Curves: High-Travel Best vs Worst", fontsize=14)
    combined_path = args.out_dir / "high_travel_best_worst_curves.png"
    fig.savefig(combined_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    for group, row in selected:
        fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
        plot_log(ax, row, group=group, seed=stable_seed(row["log"]))
        ax.legend(loc="best", frameon=False)
        fig.savefig(args.out_dir / f"{group.lower()}_{row['log']}_curve.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    write_plot_note(args.out_dir / "README.md", worst, best)
    print(f"Wrote {combined_path}")


if __name__ == "__main__":
    main()
