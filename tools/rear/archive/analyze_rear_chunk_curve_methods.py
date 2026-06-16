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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from sklearn.isotonic import IsotonicRegression


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import rear_mag_model as rmm  # noqa: E402
from mag_to_travel_model_core import MagToTravelChunk, MagToTravelModel  # noqa: E402


CHUNK_SUPPORT_THRESHOLDS = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)


@dataclass
class MethodEval:
    masked_aligned_rmse: float
    aligned_rmse: float
    corr: float
    slope_ratio_q10_q90: float


@dataclass
class LogSummary:
    name: str
    chunk_count: int
    chunk_zero_mag_median: float
    current: MethodEval
    x0_penalty: MethodEval
    endpoint_power: MethodEval
    endpoint_power_bounded: MethodEval
    binned_secant: MethodEval
    monotone_knots: MethodEval
    oracle_fixed_power_scan: MethodEval
    oracle_isotonic: MethodEval
    current_vs_oracle_chunk_support: dict[str, list[int] | int | float]


class PiecewiseCurve:
    def __init__(self, knots: np.ndarray, values: np.ndarray):
        self.knots = np.asarray(knots, dtype=float)
        self.values = np.asarray(values, dtype=float)

    def pred(self, mag: np.ndarray) -> np.ndarray:
        return np.interp(
            np.asarray(mag, dtype=float),
            self.knots,
            self.values,
            left=self.values[0],
            right=self.values[-1],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze chunk-based rear mag-to-travel curve methods.")
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
        default=Path("reports/rear_chunk_curve_methods_149_153"),
        help="Directory for report artifacts.",
    )
    return parser.parse_args()


def rmse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x**2)))


def aligned_metrics(pred: np.ndarray, gt: np.ndarray, roi_mask: np.ndarray) -> MethodEval:
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    roi = np.asarray(roi_mask, dtype=bool)

    offset = float(np.median(gt - pred))
    pred_aligned = pred + offset
    aligned_rmse = rmse(gt - pred_aligned)
    masked_aligned_rmse = rmse((gt - pred_aligned)[roi])
    corr = float(np.corrcoef(pred, gt)[0, 1])

    q10, q90 = np.percentile(pred[roi] * 0 + np.arange(np.sum(roi)), [10, 90])
    del q10, q90  # unused placeholder to keep signature symmetry in helpers

    return MethodEval(
        masked_aligned_rmse=masked_aligned_rmse,
        aligned_rmse=aligned_rmse,
        corr=corr,
        slope_ratio_q10_q90=np.nan,
    )


def set_slope_ratio(metrics: MethodEval, pred: np.ndarray, mag: np.ndarray, roi_mask: np.ndarray) -> MethodEval:
    roi = np.asarray(roi_mask, dtype=bool)
    mag_roi = np.asarray(mag, dtype=float)[roi]
    pred_roi = np.asarray(pred, dtype=float)[roi]
    order = np.argsort(mag_roi)
    mag_sorted = mag_roi[order]
    pred_sorted = pred_roi[order]
    slope = np.abs(np.gradient(pred_sorted, mag_sorted))
    slope = slope[np.isfinite(slope)]
    if len(slope) == 0:
        ratio = np.nan
    else:
        q10, q90 = np.percentile(slope, [10, 90])
        ratio = float(q90 / max(q10, 1e-9))
    return MethodEval(
        masked_aligned_rmse=metrics.masked_aligned_rmse,
        aligned_rmse=metrics.aligned_rmse,
        corr=metrics.corr,
        slope_ratio_q10_q90=ratio,
    )


def evaluate_curve(pred: np.ndarray, mag: np.ndarray, travel: np.ndarray, roi_mask: np.ndarray) -> MethodEval:
    return set_slope_ratio(aligned_metrics(pred, travel, roi_mask), pred, mag, roi_mask)


def load_case(log_name: str):
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        accel_proj, mag, t, travel, v_gt, a_gt, zv_points, roi_mask = rmm.load_ws(log_name)
        current_model = rmm.RearMagModel(x0_weight=0.0)
        chunks = current_model.create_chunks(zv_points, mag, accel_proj, t)
        current_model.prepare_chunks(chunks)
        current_model.calc_chunks_errors(chunks, travel, v_gt, a_gt)
        chunks = current_model.filter_chunks(chunks, current_model.get_filter_fns())
    return accel_proj, mag, t, travel, roi_mask.astype(bool), chunks


def fit_current_powerlaw(
    chunks: list[MagToTravelChunk],
    pred_soft_mg: float,
    x0_weight: float,
) -> np.ndarray:
    model = rmm.RearMagModel(x0_weight=x0_weight)
    input_arr = model.format_chunks_for_fit(chunks)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        result = model.fit_model(input_arr, guess_vec=[0, -1, 1 / 3])
    return result.x.copy()


def fit_oracle_fixed_power_scan(
    mag_roi: np.ndarray,
    travel_roi: np.ndarray,
    pred_soft_mg: float,
    power: float = 1 / 3,
    num_x0: int = 2500,
) -> tuple[np.ndarray, float]:
    model = MagToTravelModel(pred_soft_mg=pred_soft_mg)
    mag_min = float(np.min(mag_roi))
    mag_max = float(np.max(mag_roi))
    span = mag_max - mag_min
    x0_grid = np.linspace(mag_min - 0.5 * span, mag_max + 0.5 * span, num_x0)

    best_rmse = np.inf
    best_vec: np.ndarray | None = None
    best_offset = 0.0
    for x0 in x0_grid:
        feat = model.pred_x(mag_roi, np.array([x0, 1.0, power]))
        A = np.column_stack([feat, np.ones_like(feat)])
        coeffs, *_ = np.linalg.lstsq(A, travel_roi, rcond=None)
        pred = A @ coeffs
        cur_rmse = rmse(pred - travel_roi)
        if cur_rmse < best_rmse:
            best_rmse = cur_rmse
            best_vec = np.array([x0, coeffs[0], power], dtype=float)
            best_offset = float(coeffs[1])
    assert best_vec is not None
    return best_vec, best_offset


def fit_oracle_isotonic(mag_roi: np.ndarray, travel_roi: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(mag_roi, travel_roi)
    return iso


def fit_endpoint_powerlaw(
    chunks: list[MagToTravelChunk],
    pred_soft_mg: float,
    power_weight: float = 1000.0,
    guess: np.ndarray | None = None,
) -> np.ndarray:
    if guess is None:
        guess = np.array([0.0, -1.0, 1 / 3], dtype=float)
    model = MagToTravelModel(pred_soft_mg=pred_soft_mg)
    m0 = np.array([chunk.mag[0] for chunk in chunks], dtype=float)
    m1 = np.array([chunk.mag[-1] for chunk in chunks], dtype=float)
    dx = np.array([chunk.x[-1] - chunk.x[0] for chunk in chunks], dtype=float)

    def residuals(vec: np.ndarray) -> np.ndarray:
        pred = model.pred_x(m1, vec) - model.pred_x(m0, vec)
        return np.concatenate([pred - dx, np.array([(vec[2] - 1 / 3) * power_weight])])

    result = scipy.optimize.least_squares(
        residuals,
        x0=np.asarray(guess, dtype=float),
        method="trf",
        max_nfev=4000,
    )
    return result.x.copy()


def fit_endpoint_powerlaw_bounded(
    chunks: list[MagToTravelChunk],
    pred_soft_mg: float,
    power_weight: float = 1000.0,
) -> np.ndarray:
    model = MagToTravelModel(pred_soft_mg=pred_soft_mg)
    m0 = np.array([chunk.mag[0] for chunk in chunks], dtype=float)
    m1 = np.array([chunk.mag[-1] for chunk in chunks], dtype=float)
    dx = np.array([chunk.x[-1] - chunk.x[0] for chunk in chunks], dtype=float)
    mags = np.concatenate([m0, m1])
    mag_min = float(np.min(mags))
    mag_max = float(np.max(mags))
    span = mag_max - mag_min

    def residuals(vec: np.ndarray) -> np.ndarray:
        pred = model.pred_x(m1, vec) - model.pred_x(m0, vec)
        parts = [pred - dx]
        if power_weight > 0:
            parts.append(np.array([(vec[2] - 1 / 3) * power_weight]))
        return np.concatenate(parts)

    guesses = []
    for x0 in (
        mag_min - 0.5 * span,
        mag_min - 0.1 * span,
        mag_min + 0.1 * span,
        float(np.median(mags)),
        mag_max,
    ):
        for y_scale in (-0.002, -0.005, -0.01, -0.02, -0.05):
            for power in (0.15, 0.25, 1 / 3, 0.5, 0.8):
                guesses.append(np.array([x0, y_scale, power], dtype=float))

    bounds = (
        np.array([mag_min - 0.75 * span, -1000.0, 0.05], dtype=float),
        np.array([mag_max + 0.75 * span, 0.0, 2.0], dtype=float),
    )

    best_cost = np.inf
    best_vec: np.ndarray | None = None
    for guess in guesses:
        try:
            result = scipy.optimize.least_squares(
                residuals,
                x0=guess,
                bounds=bounds,
                method="trf",
                max_nfev=1500,
            )
        except ValueError:
            continue
        if result.cost < best_cost:
            best_cost = float(result.cost)
            best_vec = result.x.copy()
    assert best_vec is not None
    return best_vec


def fit_binned_secant_curve(
    chunks: list[MagToTravelChunk],
    bin_count: int = 18,
    min_abs_dm: float = 250.0,
) -> PiecewiseCurve | None:
    mids: list[float] = []
    slopes: list[float] = []
    for chunk in chunks:
        mag0 = float(chunk.mag[chunk.zv_idx])
        x0 = float(chunk.x[chunk.zv_idx])
        for idx in range(chunk.chunk_len):
            if idx == chunk.zv_idx:
                continue
            dm = float(chunk.mag[idx] - mag0)
            dx = float(chunk.x[idx] - x0)
            if abs(dm) < min_abs_dm:
                continue
            mids.append(0.5 * (float(chunk.mag[idx]) + mag0))
            slopes.append(dx / dm)

    if len(mids) < bin_count * 10:
        return None

    mids_arr = np.asarray(mids, dtype=float)
    slopes_arr = np.asarray(slopes, dtype=float)
    edges = np.quantile(mids_arr, np.linspace(0.0, 1.0, bin_count + 1))
    edges = np.unique(edges)

    centers: list[float] = []
    med_slopes: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (mids_arr >= lo) & (mids_arr <= hi)
        if int(np.sum(mask)) < 20:
            continue
        centers.append(float(np.median(mids_arr[mask])))
        med_slopes.append(float(np.median(slopes_arr[mask])))

    if len(centers) < 4:
        return None

    centers_arr = np.asarray(centers, dtype=float)
    abs_slope = np.maximum(1e-6, -np.asarray(med_slopes, dtype=float))
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    abs_slope_fit = iso.fit_transform(centers_arr, abs_slope)

    values = np.zeros(len(centers_arr), dtype=float)
    for idx in range(len(centers_arr) - 2, -1, -1):
        dm = centers_arr[idx + 1] - centers_arr[idx]
        values[idx] = values[idx + 1] + abs_slope_fit[idx] * dm
    return PiecewiseCurve(centers_arr, values)


def interp_weights(knots: np.ndarray, mags: np.ndarray) -> np.ndarray:
    knots = np.asarray(knots, dtype=float)
    mags = np.asarray(mags, dtype=float)
    idx = np.searchsorted(knots, mags, side="right") - 1
    idx = np.clip(idx, 0, len(knots) - 2)
    left = knots[idx]
    right = knots[idx + 1]
    frac = (mags - left) / np.maximum(right - left, 1e-9)
    weights = np.zeros((len(mags), len(knots)), dtype=float)
    rows = np.arange(len(mags))
    weights[rows, idx] = 1.0 - frac
    weights[rows, idx + 1] = frac

    below = mags <= knots[0]
    above = mags >= knots[-1]
    if np.any(below):
        weights[below, :] = 0.0
        weights[below, 0] = 1.0
    if np.any(above):
        weights[above, :] = 0.0
        weights[above, -1] = 1.0
    return weights


def fit_monotone_knots_curve(
    chunks: list[MagToTravelChunk],
    knot_count: int = 14,
    smooth_weight: float = 30.0,
) -> PiecewiseCurve:
    zero_mags = np.array([chunk.mag[chunk.zv_idx] for chunk in chunks], dtype=float)
    knots = np.quantile(zero_mags, np.linspace(0.0, 1.0, knot_count))
    knots = np.unique(knots)
    if len(knots) < 4:
        knots = np.linspace(float(np.min(zero_mags)), float(np.max(zero_mags)), knot_count)
    knot_count = len(knots)

    transform = np.zeros((knot_count, knot_count - 1), dtype=float)
    for idx in range(1, knot_count):
        transform[idx, :idx] = -1.0

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for chunk in chunks:
        w0 = interp_weights(knots, np.array([chunk.mag[chunk.zv_idx]], dtype=float))[0]
        for idx in range(chunk.chunk_len):
            if idx == chunk.zv_idx:
                continue
            wi = interp_weights(knots, np.array([chunk.mag[idx]], dtype=float))[0]
            rows.append((wi - w0) @ transform)
            rhs.append(float(chunk.x[idx]))

    A = np.vstack(rows)
    b = np.asarray(rhs, dtype=float)

    if smooth_weight > 0 and knot_count >= 3:
        second_diff = np.zeros((knot_count - 2, knot_count), dtype=float)
        for idx in range(knot_count - 2):
            second_diff[idx, idx : idx + 3] = [1.0, -2.0, 1.0]
        A = np.vstack([A, smooth_weight * (second_diff @ transform)])
        b = np.concatenate([b, np.zeros(knot_count - 2, dtype=float)])

    result = scipy.optimize.lsq_linear(A, b, bounds=(0.0, np.inf))
    values = transform @ result.x
    return PiecewiseCurve(knots, values)


def chunk_rmse_support(chunks: list[MagToTravelChunk], pred_fn, thresholds: tuple[float, ...]) -> tuple[np.ndarray, float]:
    errs = []
    for chunk in chunks:
        zero_pred = float(pred_fn(np.array([chunk.mag[chunk.zv_idx]], dtype=float))[0])
        pred_rel = pred_fn(chunk.mag) - zero_pred
        errs.append(rmse(pred_rel - chunk.x))
    err_arr = np.asarray(errs, dtype=float)
    counts = np.array([int(np.sum(err_arr <= th)) for th in thresholds], dtype=int)
    return counts, float(np.median(err_arr))


def plot_log(
    out_path: Path,
    log_name: str,
    mag: np.ndarray,
    travel: np.ndarray,
    roi_mask: np.ndarray,
    curves: dict[str, np.ndarray],
) -> None:
    roi = np.asarray(roi_mask, dtype=bool)
    mag_roi = mag[roi]
    travel_roi = travel[roi]
    idx = np.argsort(mag_roi)
    mag_s = mag_roi[idx]
    travel_s = travel_roi[idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(mag_roi, travel_roi, s=4, alpha=0.12, color="black", label="GT travel (ROI)")
    ax.plot(mag_s, travel_s, color="black", alpha=0.25, linewidth=1)
    for label, pred in curves.items():
        ax.plot(mag_s, pred[roi][idx], linewidth=2, label=label)
    ax.set_title(f"{log_name}: ROI Travel vs Mag")
    ax.set_xlabel("mag/proj/lpf")
    ax.set_ylabel("travel (mm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_report(log_summaries: list[LogSummary], out_dir: Path) -> None:
    def mean_metric(method_name: str, field_name: str) -> float:
        vals = [getattr(getattr(summary, method_name), field_name) for summary in log_summaries]
        return float(np.mean(np.asarray(vals, dtype=float)))

    method_names = [
        "current",
        "x0_penalty",
        "endpoint_power",
        "endpoint_power_bounded",
        "binned_secant",
        "monotone_knots",
        "oracle_fixed_power_scan",
        "oracle_isotonic",
    ]

    avg_table = [
        (
            name,
            mean_metric(name, "masked_aligned_rmse"),
            mean_metric(name, "corr"),
            mean_metric(name, "slope_ratio_q10_q90"),
        )
        for name in method_names
    ]

    current_support = np.mean(
        np.asarray([summary.current_vs_oracle_chunk_support["current_counts"] for summary in log_summaries], dtype=float),
        axis=0,
    )
    oracle_support = np.mean(
        np.asarray([summary.current_vs_oracle_chunk_support["oracle_counts"] for summary in log_summaries], dtype=float),
        axis=0,
    )

    lines = [
        "# Rear Chunk Curve Method Analysis",
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
                f"- The current chunk-trained power-law still wins among the chunk-only methods I tried. "
                f"Its mean masked aligned RMSE is `{mean_metric('current', 'masked_aligned_rmse'):.3f} mm`."
            ),
            (
                f"- The `x0`-penalty variant helps only modestly on average: "
                f"`{mean_metric('x0_penalty', 'masked_aligned_rmse'):.3f} mm` mean masked aligned RMSE."
            ),
            (
                f"- Endpoint-only power-law fitting is the only chunk-only alternative that sometimes helps, "
                f"but it is inconsistent and slightly worse overall at "
                f"`{mean_metric('endpoint_power', 'masked_aligned_rmse'):.3f} mm`."
            ),
            (
                f"- The more RANSAC-like aggregation ideas did not recover curvature. "
                f"Binned secant aggregation averaged `{mean_metric('binned_secant', 'masked_aligned_rmse'):.3f} mm`, "
                f"and the monotone knot fit averaged `{mean_metric('monotone_knots', 'masked_aligned_rmse'):.3f} mm`."
            ),
            (
                f"- The GT-only ceilings remain much better: fixed-power scan oracle "
                f"`{mean_metric('oracle_fixed_power_scan', 'masked_aligned_rmse'):.3f} mm`, "
                f"isotonic oracle `{mean_metric('oracle_isotonic', 'masked_aligned_rmse'):.3f} mm`."
            ),
            (
                f"- The curvature-identifying signal is weak at the chunk level. "
                f"Across logs, the current learned model and the GT-curved oracle explain nearly the same number of chunks "
                f"under practical inlier thresholds."
            ),
            "",
            "## Interpretation",
            "",
            "- The chunks clearly contain enough first-order information to beat a constant-travel baseline, but not enough clean second-order information to reliably identify curvature.",
            "- Penalizing `x0` toward zero changes which branch the optimizer lands on, but it does not supply the missing curvature evidence. Once the weight is nonzero, all those fits converge to essentially the same near-zero-`x0` branch.",
            "- Endpoint constraints are cleaner than full interior chunk traces, which is why endpoint-only power-law fitting sometimes helps, but even that does not reliably recover the GT curve.",
            "- A plain RANSAC strategy is unlikely to solve this by itself because the true curved model does not win a dramatically larger inlier set than the line-like model.",
            "",
            "## Mean Method Metrics",
            "",
            "| Method | Mean masked aligned RMSE (mm) | Mean corr | Mean slope ratio q90/q10 |",
            "|---|---:|---:|---:|",
        ]
    )
    lines.extend(
        [
            f"| `{name}` | {rmse_val:.3f} | {corr_val:.4f} | {slope_ratio:.3f} |"
            for name, rmse_val, corr_val, slope_ratio in avg_table
        ]
    )
    lines.extend(
        [
            "",
            "Average chunk-support counts for current learned vs GT curved oracle:",
            "",
            "| Chunk RMSE threshold (mm) | Current learned | GT curved oracle |",
            "|---|---:|---:|",
        ]
    )
    lines.extend(
        [
            f"| `{th:g}` | {cur:.1f} | {oracle:.1f} |"
            for th, cur, oracle in zip(CHUNK_SUPPORT_THRESHOLDS, current_support, oracle_support)
        ]
    )
    lines.extend(
        [
            "",
            "## Per-Log Summary",
            "",
            "| Log | Current | x0 penalty | Endpoint power | Binned secant | Monotone knots | Oracle fixed-power | Oracle isotonic |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        [
            (
                f"| `{summary.name}` | "
                f"{summary.current.masked_aligned_rmse:.3f} | "
                f"{summary.x0_penalty.masked_aligned_rmse:.3f} | "
                f"{summary.endpoint_power.masked_aligned_rmse:.3f} | "
                f"{summary.binned_secant.masked_aligned_rmse:.3f} | "
                f"{summary.monotone_knots.masked_aligned_rmse:.3f} | "
                f"{summary.oracle_fixed_power_scan.masked_aligned_rmse:.3f} | "
                f"{summary.oracle_isotonic.masked_aligned_rmse:.3f} |"
            )
            for summary in log_summaries
        ]
    )

    best_log = min(log_summaries, key=lambda summary: summary.current.masked_aligned_rmse).name
    worst_log = max(log_summaries, key=lambda summary: summary.current.masked_aligned_rmse).name
    lines.extend(
        [
            "",
            f"Representative best current-log plot: `{best_log}`",
            "",
            f"![{best_log}]({(out_dir / f'{best_log}_curves.png').resolve()})",
            "",
            f"Representative worst current-log plot: `{worst_log}`",
            "",
            f"![{worst_log}]({(out_dir / f'{worst_log}_curves.png').resolve()})",
            "",
        ]
    )

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[LogSummary] = []

    for log_name in args.logs:
        accel_proj, mag, _t, travel, roi_mask, chunks = load_case(log_name)
        del accel_proj

        if not chunks:
            raise ValueError(f"{log_name}: no filtered chunks available")

        pred_soft_mg = 50.0
        current_coeffs = fit_current_powerlaw(chunks, pred_soft_mg=pred_soft_mg, x0_weight=0.0)
        x0_penalty_coeffs = fit_current_powerlaw(chunks, pred_soft_mg=pred_soft_mg, x0_weight=1.0)
        endpoint_coeffs = fit_endpoint_powerlaw(chunks, pred_soft_mg=pred_soft_mg, guess=current_coeffs)
        endpoint_bounded_coeffs = fit_endpoint_powerlaw_bounded(chunks, pred_soft_mg=pred_soft_mg)

        powerlaw_model = MagToTravelModel(pred_soft_mg=pred_soft_mg)
        current_pred = powerlaw_model.pred_x(mag, current_coeffs)
        x0_penalty_pred = powerlaw_model.pred_x(mag, x0_penalty_coeffs)
        endpoint_pred = powerlaw_model.pred_x(mag, endpoint_coeffs)
        endpoint_bounded_pred = powerlaw_model.pred_x(mag, endpoint_bounded_coeffs)

        secant_curve = fit_binned_secant_curve(chunks)
        if secant_curve is None:
            secant_pred = np.full_like(mag, np.nan, dtype=float)
        else:
            secant_pred = secant_curve.pred(mag)

        knot_curve = fit_monotone_knots_curve(chunks)
        knot_pred = knot_curve.pred(mag)

        mag_roi = mag[roi_mask]
        travel_roi = travel[roi_mask]
        oracle_coeffs, oracle_offset = fit_oracle_fixed_power_scan(mag_roi, travel_roi, pred_soft_mg=pred_soft_mg)
        oracle_fixed_pred = powerlaw_model.pred_x(mag, oracle_coeffs) + oracle_offset

        oracle_iso = fit_oracle_isotonic(mag_roi, travel_roi)
        oracle_iso_pred = oracle_iso.predict(mag)

        current_eval = evaluate_curve(current_pred, mag, travel, roi_mask)
        x0_penalty_eval = evaluate_curve(x0_penalty_pred, mag, travel, roi_mask)
        endpoint_eval = evaluate_curve(endpoint_pred, mag, travel, roi_mask)
        endpoint_bounded_eval = evaluate_curve(endpoint_bounded_pred, mag, travel, roi_mask)
        secant_eval = evaluate_curve(secant_pred, mag, travel, roi_mask)
        knot_eval = evaluate_curve(knot_pred, mag, travel, roi_mask)
        oracle_fixed_eval = evaluate_curve(oracle_fixed_pred, mag, travel, roi_mask)
        oracle_iso_eval = evaluate_curve(oracle_iso_pred, mag, travel, roi_mask)

        current_counts, current_chunk_median = chunk_rmse_support(
            chunks,
            lambda x: powerlaw_model.pred_x(x, current_coeffs),
            CHUNK_SUPPORT_THRESHOLDS,
        )
        oracle_counts, oracle_chunk_median = chunk_rmse_support(
            chunks,
            lambda x: powerlaw_model.pred_x(x, oracle_coeffs) + oracle_offset,
            CHUNK_SUPPORT_THRESHOLDS,
        )

        zero_mags = np.array([chunk.mag[chunk.zv_idx] for chunk in chunks], dtype=float)
        summaries.append(
            LogSummary(
                name=log_name,
                chunk_count=len(chunks),
                chunk_zero_mag_median=float(np.median(zero_mags)),
                current=current_eval,
                x0_penalty=x0_penalty_eval,
                endpoint_power=endpoint_eval,
                endpoint_power_bounded=endpoint_bounded_eval,
                binned_secant=secant_eval,
                monotone_knots=knot_eval,
                oracle_fixed_power_scan=oracle_fixed_eval,
                oracle_isotonic=oracle_iso_eval,
                current_vs_oracle_chunk_support={
                    "thresholds_mm": list(CHUNK_SUPPORT_THRESHOLDS),
                    "current_counts": current_counts.tolist(),
                    "oracle_counts": oracle_counts.tolist(),
                    "current_chunk_median_rmse": current_chunk_median,
                    "oracle_chunk_median_rmse": oracle_chunk_median,
                },
            )
        )

        plot_log(
            out_dir / f"{log_name}_curves.png",
            log_name,
            mag,
            travel,
            roi_mask,
            {
                "current": current_pred,
                "x0_penalty": x0_penalty_pred,
                "endpoint_power": endpoint_pred,
                "oracle_fixed_power": oracle_fixed_pred,
                "oracle_isotonic": oracle_iso_pred,
            },
        )

    payload = {"logs": [asdict(summary) for summary in summaries]}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(summaries, out_dir)


if __name__ == "__main__":
    main()
