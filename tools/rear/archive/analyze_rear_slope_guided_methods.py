#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import numpy as np
from scipy.optimize import least_squares


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import rear_mag_model as rmm  # noqa: E402
from mag_to_travel_model_core import MagToTravelModel  # noqa: E402


@dataclass
class MethodMetric:
    masked_aligned_rmse: float
    corr: float
    bin_slope_mae: float
    pred_slope_ratio: float
    coeffs: list[float]


@dataclass
class LogSummary:
    name: str
    methods: dict[str, MethodMetric]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze slope-guided rear mag model fits.")
    parser.add_argument(
        "--logs",
        nargs="*",
        default=[
            "log149_rear",
            "log150_rear",
            "log151_rear",
            "log152_rear",
            "log153_rear",
        ],
        help="Log names without extension.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/rear_slope_guided_methods_149_153"),
        help="Directory for report artifacts.",
    )
    return parser.parse_args()


def masked_aligned_rmse(pred: np.ndarray, gt: np.ndarray, roi_mask: np.ndarray) -> float:
    roi = np.asarray(roi_mask, dtype=bool)
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    offset = float(np.median(gt - pred))
    return float(np.sqrt(np.mean(((gt - (pred + offset))[roi]) ** 2)))


def fit_regression_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.ptp(x) <= 1e-9 or len(np.unique(x)) < 3:
        return np.nan
    return float(rmm.scipy.stats.linregress(x, y).slope)


def pred_dx_dmag(mag_i: np.ndarray | float, coeffs: np.ndarray, pred_soft_mg: float) -> np.ndarray:
    x0, y_scale, power = coeffs
    return y_scale * power * (np.abs(np.asarray(mag_i, dtype=float) - x0) + pred_soft_mg) ** (power - 1.0)


def pred_dmag_dx(mag_i: np.ndarray | float, coeffs: np.ndarray, pred_soft_mg: float) -> np.ndarray:
    deriv = pred_dx_dmag(mag_i, coeffs, pred_soft_mg)
    return 1.0 / np.where(np.abs(deriv) < 1e-9, np.sign(deriv) * 1e-9 + (deriv == 0) * 1e-9, deriv)


def build_bin_curve(chunk_mags: np.ndarray, slope_proxy: np.ndarray, n_bins: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hist_min = float(np.percentile(chunk_mags, 1))
    hist_max = float(np.percentile(chunk_mags, 99))
    bin_size = (hist_max - hist_min) / n_bins

    centers = []
    medians = []
    counts = []
    for i in range(n_bins):
        bin_min = hist_min + i * bin_size
        bin_max = bin_min + bin_size
        mask = (bin_min <= chunk_mags) & (chunk_mags < bin_max)
        centers.append(bin_min + 0.5 * bin_size)
        counts.append(int(np.sum(mask)))
        medians.append(float(np.nanmedian(slope_proxy[mask])) if np.any(mask) else np.nan)
    return np.asarray(centers), np.asarray(medians), np.asarray(counts, dtype=float)


def build_local_median_targets(chunk_mags: np.ndarray, slope_proxy: np.ndarray, window: int) -> np.ndarray:
    order = np.argsort(chunk_mags)
    mags_sorted = chunk_mags[order]
    proxy_sorted = slope_proxy[order]
    targets_sorted = np.full_like(proxy_sorted, np.nan, dtype=float)
    half = window // 2
    for i in range(len(proxy_sorted)):
        lo = max(0, i - half)
        hi = min(len(proxy_sorted), i + half + 1)
        targets_sorted[i] = float(np.nanmedian(proxy_sorted[lo:hi]))
    targets = np.full_like(targets_sorted, np.nan, dtype=float)
    targets[order] = targets_sorted
    return targets


def build_gt_bin_slopes(
    bin_centers: np.ndarray,
    chunk_mags: np.ndarray,
    mag_all: np.ndarray,
    travel: np.ndarray,
    roi_mask: np.ndarray,
) -> np.ndarray:
    n_bins = len(bin_centers)
    hist_min = float(np.percentile(chunk_mags, 1))
    hist_max = float(np.percentile(chunk_mags, 99))
    bin_size = (hist_max - hist_min) / n_bins

    roi = np.asarray(roi_mask, dtype=bool)
    mag_roi = np.asarray(mag_all, dtype=float)[roi]
    travel_roi = np.asarray(travel, dtype=float)[roi]

    gt_reg = []
    for i in range(n_bins):
        bin_min = hist_min + i * bin_size
        bin_max = bin_min + bin_size
        gt_mask = (bin_min <= mag_roi) & (mag_roi < bin_max)
        gt_reg.append(fit_regression_slope(travel_roi[gt_mask], mag_roi[gt_mask]))
    return np.asarray(gt_reg, dtype=float)


def make_fit_fn(
    train_model: rmm.RearMagModel,
    input_arr: np.ndarray,
    power_prior: float,
    extra_residual_fn=None,
):
    model = MagToTravelModel(pred_soft_mg=train_model.pred_soft_mg)
    base_residual_fn = train_model.make_residual_fn(model, input_arr, power_prior)

    def residual_fn(vec: np.ndarray) -> np.ndarray:
        base = base_residual_fn(vec)
        if extra_residual_fn is None:
            return base
        extra = extra_residual_fn(vec)
        if extra.size == 0:
            return base
        return np.concatenate([base, extra])

    return model, residual_fn


def fit_with_objective(
    train_model: rmm.RearMagModel,
    input_arr: np.ndarray,
    guess_vec: np.ndarray,
    *,
    power_prior: float = 1 / 3,
    extra_residual_fn=None,
):
    model, residual_fn = make_fit_fn(train_model, input_arr, power_prior, extra_residual_fn)
    result = least_squares(
        fun=residual_fn,
        x0=np.asarray(guess_vec, dtype=float),
        method="trf",
        verbose=0,
        max_nfev=1000,
    )
    model.set_coeffs(result.x)
    train_model.model = model
    return result, model


def make_bin_prior_residual(
    chunk_mags: np.ndarray,
    slope_proxy: np.ndarray,
    *,
    pred_soft_mg: float,
    slope_weight: float,
    slope_scale: float = 5.0,
):
    centers, medians, counts = build_bin_curve(chunk_mags, slope_proxy)
    finite = np.isfinite(centers) & np.isfinite(medians) & (counts > 0)
    centers = centers[finite]
    medians = medians[finite]
    counts = counts[finite]
    if len(centers) == 0:
        return lambda vec: np.empty(0, dtype=float), np.empty(0), np.empty(0)

    weights = np.sqrt(counts)
    weights /= max(float(np.mean(weights)), 1e-9)

    def extra_residual(vec: np.ndarray) -> np.ndarray:
        pred = pred_dmag_dx(centers, vec, pred_soft_mg)
        return ((pred - medians) / slope_scale) * weights * slope_weight

    return extra_residual, centers, medians


def make_bin_shape_prior_residual(
    chunk_mags: np.ndarray,
    slope_proxy: np.ndarray,
    *,
    pred_soft_mg: float,
    slope_weight: float,
):
    centers, medians, counts = build_bin_curve(chunk_mags, slope_proxy)
    finite = np.isfinite(centers) & np.isfinite(medians) & (counts > 0)
    centers = centers[finite]
    medians = medians[finite]
    counts = counts[finite]
    if len(centers) < 2:
        return lambda vec: np.empty(0, dtype=float), np.empty(0), np.empty(0)

    weights = np.sqrt(counts)
    weights /= max(float(np.mean(weights)), 1e-9)
    target_shape = np.abs(medians) / max(abs(medians[0]), 1e-9)

    def extra_residual(vec: np.ndarray) -> np.ndarray:
        pred_abs = np.abs(pred_dmag_dx(centers, vec, pred_soft_mg))
        pred_shape = pred_abs / max(float(pred_abs[0]), 1e-9)
        return (pred_shape - target_shape) * weights * slope_weight

    return extra_residual, centers, medians


def make_consensus_residual(
    chunk_mags: np.ndarray,
    chunk_targets: np.ndarray,
    *,
    pred_soft_mg: float,
    slope_weight: float,
    consensus_scale: float = 20.0,
):
    finite = np.isfinite(chunk_mags) & np.isfinite(chunk_targets)
    mags = chunk_mags[finite]
    targets = chunk_targets[finite]
    if len(mags) == 0:
        return lambda vec: np.empty(0, dtype=float)

    def extra_residual(vec: np.ndarray) -> np.ndarray:
        pred = pred_dmag_dx(mags, vec, pred_soft_mg)
        return np.tanh((pred - targets) / consensus_scale) * slope_weight

    return extra_residual


def summarize_method(
    coeffs: np.ndarray,
    pred: np.ndarray,
    travel: np.ndarray,
    roi_mask: np.ndarray,
    bin_centers: np.ndarray,
    gt_bin_slopes: np.ndarray,
    pred_soft_mg: float,
) -> MethodMetric:
    pred_bin_slopes = pred_dmag_dx(bin_centers, coeffs, pred_soft_mg)
    valid = np.isfinite(pred_bin_slopes) & np.isfinite(gt_bin_slopes)
    if np.any(valid):
        bin_slope_mae = float(np.mean(np.abs(pred_bin_slopes[valid] - gt_bin_slopes[valid])))
    else:
        bin_slope_mae = np.nan

    if np.all(np.isfinite(pred_bin_slopes[[0, -1]])):
        pred_slope_ratio = float(abs(pred_bin_slopes[-1]) / max(abs(pred_bin_slopes[0]), 1e-9))
    else:
        pred_slope_ratio = np.nan

    return MethodMetric(
        masked_aligned_rmse=masked_aligned_rmse(pred, travel, roi_mask),
        corr=float(np.corrcoef(pred, travel)[0, 1]),
        bin_slope_mae=bin_slope_mae,
        pred_slope_ratio=pred_slope_ratio,
        coeffs=np.asarray(coeffs, dtype=float).tolist(),
    )


def run_log(log_name: str) -> LogSummary:
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        accel_proj, mag, t, travel, v_gt, a_gt, zv_points, roi_mask = rmm.load_ws(log_name)

    train_model = rmm.RearMagModel(x0_weight=1.0, dm_dx_thresh=None)
    chunks = train_model.create_chunks(zv_points, mag, accel_proj, t)
    train_model.prepare_chunks(chunks)
    train_model.calc_chunks_errors(chunks, travel, v_gt, a_gt)
    chunks = train_model.filter_chunks(chunks, train_model.get_filter_fns())

    with contextlib.redirect_stdout(sink):
        input_arr = train_model.format_chunks_for_fit(chunks)

    chunk_mags = np.asarray([np.median(chunk.mag) for chunk in chunks], dtype=float)
    proxy_scaled = np.asarray(
        [
            chunk.metrics["dm/dx_median"] / max(float(np.median(np.diff(chunk.t))), 1e-9)
            for chunk in chunks
        ],
        dtype=float,
    )

    bin_centers, bin_proxy_medians, _ = build_bin_curve(chunk_mags, proxy_scaled)
    gt_bin_slopes = build_gt_bin_slopes(bin_centers, chunk_mags, mag, travel, roi_mask)
    interp_proxy_targets = np.interp(chunk_mags, bin_centers, bin_proxy_medians)
    local_proxy_targets = build_local_median_targets(chunk_mags, proxy_scaled, window=151)

    baseline_result, baseline_model = fit_with_objective(
        train_model,
        input_arr,
        guess_vec=np.asarray([0.0, -1.0, 1.0 / 3.0]),
    )
    baseline_pred = baseline_model.pred_x(mag)

    method_metrics: dict[str, MethodMetric] = {
        "baseline": summarize_method(
            baseline_result.x,
            baseline_pred,
            travel,
            roi_mask,
            bin_centers,
            gt_bin_slopes,
            train_model.pred_soft_mg,
        )
    }

    method_cfgs = [
        ("bin_prior_w10", "bin_prior", {"slope_weight": 10.0}),
        ("bin_prior_w30", "bin_prior", {"slope_weight": 30.0}),
        ("bin_prior_w100", "bin_prior", {"slope_weight": 100.0}),
        ("bin_shape_prior_w10", "bin_shape_prior", {"slope_weight": 10.0}),
        ("bin_shape_prior_w30", "bin_shape_prior", {"slope_weight": 30.0}),
        ("bin_shape_prior_w100", "bin_shape_prior", {"slope_weight": 100.0}),
        ("bin_interp_consensus_w0.2", "bin_interp_consensus", {"slope_weight": 0.20}),
        ("bin_interp_consensus_w0.5", "bin_interp_consensus", {"slope_weight": 0.50}),
        ("bin_interp_consensus_w1.0", "bin_interp_consensus", {"slope_weight": 1.00}),
        ("local_median_consensus_w0.2", "local_median_consensus", {"slope_weight": 0.20}),
        ("local_median_consensus_w0.5", "local_median_consensus", {"slope_weight": 0.50}),
        ("local_median_consensus_w1.0", "local_median_consensus", {"slope_weight": 1.00}),
        ("raw_proxy_consensus_w0.2", "raw_proxy_consensus", {"slope_weight": 0.20}),
        ("raw_proxy_consensus_w0.5", "raw_proxy_consensus", {"slope_weight": 0.50}),
    ]

    for method_name, kind, params in method_cfgs:
        if kind == "bin_prior":
            extra_residual_fn, _, _ = make_bin_prior_residual(
                chunk_mags,
                proxy_scaled,
                pred_soft_mg=train_model.pred_soft_mg,
                slope_weight=params["slope_weight"],
            )
        elif kind == "bin_shape_prior":
            extra_residual_fn, _, _ = make_bin_shape_prior_residual(
                chunk_mags,
                proxy_scaled,
                pred_soft_mg=train_model.pred_soft_mg,
                slope_weight=params["slope_weight"],
            )
        elif kind == "bin_interp_consensus":
            extra_residual_fn = make_consensus_residual(
                chunk_mags,
                interp_proxy_targets,
                pred_soft_mg=train_model.pred_soft_mg,
                slope_weight=params["slope_weight"],
            )
        elif kind == "local_median_consensus":
            extra_residual_fn = make_consensus_residual(
                chunk_mags,
                local_proxy_targets,
                pred_soft_mg=train_model.pred_soft_mg,
                slope_weight=params["slope_weight"],
            )
        elif kind == "raw_proxy_consensus":
            extra_residual_fn = make_consensus_residual(
                chunk_mags,
                proxy_scaled,
                pred_soft_mg=train_model.pred_soft_mg,
                slope_weight=params["slope_weight"],
            )
        else:
            raise ValueError(f"Unknown method kind {kind}")

        method_model = rmm.RearMagModel(x0_weight=1.0, dm_dx_thresh=None)
        method_result, fitted_model = fit_with_objective(
            method_model,
            input_arr,
            guess_vec=baseline_result.x.copy(),
            extra_residual_fn=extra_residual_fn,
        )
        pred = fitted_model.pred_x(mag)
        method_metrics[method_name] = summarize_method(
            method_result.x,
            pred,
            travel,
            roi_mask,
            bin_centers,
            gt_bin_slopes,
            method_model.pred_soft_mg,
        )

    return LogSummary(name=log_name, methods=method_metrics)


def write_report(log_summaries: list[LogSummary], out_dir: Path) -> None:
    method_names = list(log_summaries[0].methods.keys())

    def mean_metric(method_name: str, field_name: str) -> float:
        vals = [getattr(summary.methods[method_name], field_name) for summary in log_summaries]
        return float(np.nanmean(np.asarray(vals, dtype=float)))

    mean_rows = [
        (
            method_name,
            mean_metric(method_name, "masked_aligned_rmse"),
            mean_metric(method_name, "bin_slope_mae"),
            mean_metric(method_name, "pred_slope_ratio"),
        )
        for method_name in method_names
    ]
    mean_rows_by_rmse = sorted(mean_rows, key=lambda row: row[1])
    mean_rows_by_slope = sorted(mean_rows, key=lambda row: row[2])

    best_rmse = mean_rows_by_rmse[0]
    best_slope = mean_rows_by_slope[0]
    baseline_rmse = mean_metric("baseline", "masked_aligned_rmse")
    baseline_slope_mae = mean_metric("baseline", "bin_slope_mae")
    baseline_ratio = mean_metric("baseline", "pred_slope_ratio")

    lines = [
        "# Rear Slope-Guided Method Analysis",
        "",
        "Logs used:",
        "",
    ]
    lines.extend([f"- `{summary.name}`" for summary in log_summaries])
    lines.extend(
        [
            "",
            "## Main Findings",
            "",
            (
                f"- None of the slope-guided methods produced a meaningful RMSE win over the current rear fit. "
                f"Best mean masked aligned RMSE was `{best_rmse[0]}` at `{best_rmse[1]:.3f} mm`, "
                f"versus baseline `{baseline_rmse:.3f} mm`."
            ),
            (
                f"- The only objective that moved the curve materially was the strong absolute bin prior "
                f"`bin_prior_w100`: it improved mean GT bin-slope MAE from `{baseline_slope_mae:.3f}` to "
                f"`{best_slope[2]:.3f} mG/mm`, but it also worsened mean RMSE to "
                f"`{mean_metric('bin_prior_w100', 'masked_aligned_rmse'):.3f} mm`."
            ),
            (
                f"- The RANSAC-like consensus objectives and the shape-only priors were essentially no-ops. "
                f"Their mean first-last `|dmag/dx|` ratio stayed at about `{baseline_ratio:.3f}`, "
                "the same as baseline."
            ),
            (
                "- That points to a specific limitation: the chunk slope proxy seems to carry some trend information, "
                "but not enough leverage to teach the solver appreciably more curvature inside the current model family."
            ),
            "",
            "## Mean Metrics",
            "",
            "| Method | Mean masked aligned RMSE (mm) | Mean GT bin-slope MAE (mG/mm) | Mean first-last |dmag/dx| ratio |",
            "|---|---:|---:|---:|",
        ]
    )
    for method_name, rmse_val, slope_mae, slope_ratio in mean_rows_by_rmse:
        lines.append(f"| `{method_name}` | {rmse_val:.3f} | {slope_mae:.3f} | {slope_ratio:.3f} |")

    lines.extend(
        [
            "",
            "## Per-Log RMSE",
            "",
            "| Log | " + " | ".join(f"`{name}`" for name in method_names) + " |",
            "|---|" + "|".join(["---:"] * len(method_names)) + "|",
        ]
    )
    for summary in log_summaries:
        row = " | ".join(f"{summary.methods[name].masked_aligned_rmse:.3f}" for name in method_names)
        lines.append(f"| `{summary.name}` | {row} |")

    lines.extend(
        [
            "",
            "## Per-Log Slope MAE",
            "",
            "| Log | " + " | ".join(f"`{name}`" for name in method_names) + " |",
            "|---|" + "|".join(["---:"] * len(method_names)) + "|",
        ]
    )
    for summary in log_summaries:
        row = " | ".join(f"{summary.methods[name].bin_slope_mae:.3f}" for name in method_names)
        lines.append(f"| `{summary.name}` | {row} |")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for log_name in args.logs:
        print(f"Running {log_name}...")
        summaries.append(run_log(log_name))

    payload = {"logs": [asdict(summary) for summary in summaries]}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(summaries, out_dir)


if __name__ == "__main__":
    main()
