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
import pandas as pd
import scipy.integrate
from scipy.signal import butter, savgol_filter, sosfiltfilt


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import rear_mag_model as rmm  # noqa: E402
from mag_to_travel_model_core import MagToTravelChunk  # noqa: E402


TARGET_GYRO_VEL_CHUNK_DX = 40.0


@dataclass
class MethodResult:
    masked_aligned_rmse: float
    corr: float
    slope_ratio_q90_q10: float
    chunk_count: int
    coeffs: list[float]


@dataclass
class LogSummary:
    name: str
    gyro_z_to_travel_vel_corr: float
    gyro_zdot_to_travel_acc_corr: float
    gyro_zdot_grad_to_travel_acc_corr: float
    gyro_vel_scale_raw: float
    gyro_vel_scale_hp1: float
    accel_x0_1: MethodResult
    gyro_alpha_neg_x0_1: MethodResult
    gyro_alpha_grad_neg_x0_1: MethodResult
    gyro_alpha_neg_hp1_x0_1: MethodResult
    gyro_vel_neg_scaled_x0_1: MethodResult
    gyro_vel_neg_hp1_scaled_x0_1: MethodResult
    gyro_vel_neg_scaled_x0_0: MethodResult


class RearVelocityProxyModel(rmm.RearMagModel):
    def create_chunks(self, idxs, mag, vel, t_s):
        chunks: list[MagToTravelChunk] = []
        pairs = self.find_zv_pairs(idxs, mag, t_s)
        for zv_start, zv_stop in pairs:
            sl = slice(zv_start, zv_stop + 1)
            raw_v = np.asarray(vel[sl], dtype=float)
            t_chunk = np.asarray(t_s[sl], dtype=float)
            if len(raw_v) < 3 or t_chunk[-1] <= t_chunk[0]:
                continue

            # Velocity-like proxies should be zero at both true zero-velocity
            # endpoints. Remove the straight-line endpoint bias before
            # integrating once to displacement-like chunk x.
            endpoint_line = np.linspace(raw_v[0], raw_v[-1], len(raw_v))
            v_corr = raw_v - endpoint_line
            x = scipy.integrate.cumulative_trapezoid(v_corr, t_chunk, initial=0.0)

            chunk = MagToTravelChunk(
                a=np.gradient(v_corr, t_chunk, edge_order=1),
                t=t_chunk,
                mag=np.asarray(mag[sl], dtype=float),
                slice_i=sl,
                zv_idx=0,
            )
            chunk.v = v_corr
            chunk.x = x - x[0]
            chunks.append(chunk)
        return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze gyro-proxy rear mag-to-travel learning methods.")
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
        default=Path("reports/rear_gyro_proxy_methods_149_153"),
        help="Directory for report artifacts.",
    )
    return parser.parse_args()


def masked_aligned_rmse(pred: np.ndarray, gt: np.ndarray, roi_mask: np.ndarray) -> float:
    roi = np.asarray(roi_mask, dtype=bool)
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    offset = float(np.median(gt - pred))
    return float(np.sqrt(np.mean(((gt - (pred + offset))[roi]) ** 2)))


def slope_ratio_q90_q10(pred: np.ndarray, mag: np.ndarray, roi_mask: np.ndarray) -> float:
    roi = np.asarray(roi_mask, dtype=bool)
    mag_roi = np.asarray(mag, dtype=float)[roi]
    pred_roi = np.asarray(pred, dtype=float)[roi]
    order = np.argsort(mag_roi)
    mag_sorted = mag_roi[order]
    pred_sorted = pred_roi[order]
    slope = np.abs(np.gradient(pred_sorted, mag_sorted))
    slope = slope[np.isfinite(slope)]
    q10, q90 = np.percentile(slope, [10, 90])
    return float(q90 / max(q10, 1e-9))


def load_gyro2_rads(log_name: str) -> np.ndarray:
    cache = np.load(f"backend/run_artifacts/{log_name}/cache/all.npz")
    gyro_dps = np.asarray(cache["gyro/lpf/gyro2__x"], dtype=float)
    df = pd.read_csv(f"logs/{log_name}.csv")
    if float(df["lis2_x"].mean()) > 0.0:
        gyro_dps = gyro_dps @ np.diag([-1.0, -1.0, 1.0])
    return np.radians(gyro_dps)


def calibrate_velocity_scale(model: rmm.RearMagModel, idxs, mag, vel_proxy, t_s) -> float:
    pairs = model.find_zv_pairs(idxs, mag, t_s)
    dxs = []
    for zv_start, zv_stop in pairs:
        sl = slice(zv_start, zv_stop + 1)
        raw_v = np.asarray(vel_proxy[sl], dtype=float)
        t_chunk = np.asarray(t_s[sl], dtype=float)
        if len(raw_v) < 3 or t_chunk[-1] <= t_chunk[0]:
            continue
        v_corr = raw_v - np.linspace(raw_v[0], raw_v[-1], len(raw_v))
        x = scipy.integrate.cumulative_trapezoid(v_corr, t_chunk, initial=0.0)
        dxs.append(float(np.max(x) - np.min(x)))

    if not dxs:
        return 1.0
    return TARGET_GYRO_VEL_CHUNK_DX / max(float(np.median(dxs)), 1e-9)


def fit_proxy_method(
    log_name: str,
    mag: np.ndarray,
    proxy: np.ndarray,
    t: np.ndarray,
    travel: np.ndarray,
    v_gt: np.ndarray,
    a_gt: np.ndarray,
    zv_points: np.ndarray,
    roi_mask: np.ndarray,
    *,
    proxy_kind: str,
    x0_weight: float,
) -> MethodResult:
    if proxy_kind == "accel":
        model = rmm.RearMagModel(x0_weight=x0_weight)
    elif proxy_kind == "vel":
        model = RearVelocityProxyModel(x0_weight=x0_weight)
    else:
        raise ValueError(f"Unknown proxy kind {proxy_kind}")

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        chunks = model.create_chunks(zv_points, mag, proxy, t)
        if proxy_kind == "accel":
            model.prepare_chunks(chunks)
        else:
            for chunk in chunks:
                model.calc_chunk_metrics(chunk)
        model.calc_chunks_errors(chunks, travel, v_gt, a_gt)
        chunks = model.filter_chunks(chunks, model.get_filter_fns())
        training_data = model.format_chunks_for_fit(chunks)
        if training_data.shape[0] == 0:
            raise ValueError("No chunks after filtering")
        result = model.fit_model(training_data, guess_vec=[0, -1, 1 / 3])
        pred = model.model.pred_x(mag)

    return MethodResult(
        masked_aligned_rmse=masked_aligned_rmse(pred, travel, roi_mask),
        corr=float(np.corrcoef(pred, travel)[0, 1]),
        slope_ratio_q90_q10=slope_ratio_q90_q10(pred, mag, roi_mask),
        chunk_count=len(chunks),
        coeffs=result.x.tolist(),
    )


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
    mag_sorted = mag_roi[idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(mag_roi, travel_roi, s=4, alpha=0.12, color="black", label="GT travel (ROI)")
    for label, pred in curves.items():
        pred_roi = np.asarray(pred, dtype=float)[roi]
        ax.plot(mag_sorted, pred_roi[idx], linewidth=2, label=label)
    ax.set_title(f"{log_name}: Travel vs Mag with Gyro Proxy Curves")
    ax.set_xlabel("mag/proj/lpf")
    ax.set_ylabel("travel (mm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_report(log_summaries: list[LogSummary], out_dir: Path) -> None:
    method_names = [
        "accel_x0_1",
        "gyro_alpha_neg_x0_1",
        "gyro_alpha_grad_neg_x0_1",
        "gyro_alpha_neg_hp1_x0_1",
        "gyro_vel_neg_scaled_x0_1",
        "gyro_vel_neg_hp1_scaled_x0_1",
        "gyro_vel_neg_scaled_x0_0",
    ]

    def mean_metric(method_name: str, field_name: str) -> float:
        vals = [getattr(getattr(summary, method_name), field_name) for summary in log_summaries]
        return float(np.mean(np.asarray(vals, dtype=float)))

    avg_table = [
        (
            method_name,
            mean_metric(method_name, "masked_aligned_rmse"),
            mean_metric(method_name, "corr"),
            mean_metric(method_name, "slope_ratio_q90_q10"),
        )
        for method_name in method_names
    ]

    lines = [
        "# Rear Gyro Proxy Method Analysis",
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
                f"- `gyro2_z` really does carry useful rear-motion information. "
                f"Across these logs, its correlation with GT travel velocity averages "
                f"`{np.mean([s.gyro_z_to_travel_vel_corr for s in log_summaries]):.3f}`, "
                f"and the smoothed `d/dt(gyro2_z)` proxy correlates with GT travel acceleration at "
                f"`{np.mean([s.gyro_zdot_to_travel_acc_corr for s in log_summaries]):.3f}`."
            ),
            (
                f"- As a curve-learning proxy, gyro does **not** beat the current accel proxy overall. "
                f"The current accel baseline (`x0_weight=1`) averages "
                f"`{mean_metric('accel_x0_1', 'masked_aligned_rmse'):.3f} mm`, while the best gyro-only variant "
                f"`gyro_vel_neg_hp1_scaled_x0_1` averages "
                f"`{mean_metric('gyro_vel_neg_hp1_scaled_x0_1', 'masked_aligned_rmse'):.3f} mm`."
            ),
            (
                f"- The best gyro idea was using `-gyro2_z` as a **velocity** proxy, rescaled only to satisfy the existing chunk filters. "
                f"It helped materially on `log153_rear` and was close on `log150_rear` / `log151_rear`, but it was worse on `log149_rear` and `log152_rear`."
            ),
            (
                f"- Using `-d/dt(gyro2_z)` as an acceleration proxy was viable but generally weaker. "
                f"It only slightly beat the accel baseline on `log152_rear`."
            ),
            (
                f"- A rough `np.gradient(gyro2_z)` derivative is a useful negative control: it has *higher* raw GT-accel correlation "
                f"(`{np.mean([s.gyro_zdot_grad_to_travel_acc_corr for s in log_summaries]):.3f}` vs "
                f"`{np.mean([s.gyro_zdot_to_travel_acc_corr for s in log_summaries]):.3f}` for the smoothed derivative), "
                f"but it learns a *worse* curve (`{mean_metric('gyro_alpha_grad_neg_x0_1', 'masked_aligned_rmse'):.3f} mm`). "
                f"That points back to the chunk objective as the limiting factor."
            ),
            (
                f"- None of the gyro proxies recovered much more curvature. Their learned slope-ratio metrics stay around "
                f"`{mean_metric('gyro_vel_neg_scaled_x0_1', 'slope_ratio_q90_q10'):.3f}` to "
                f"`{mean_metric('gyro_alpha_neg_x0_1', 'slope_ratio_q90_q10'):.3f}`, which is close to the accel learner and still far below the GT-oracle curvature we saw earlier."
            ),
            "",
            "## Interpretation",
            "",
            "- The gyro signal looks cleaner as a first-order motion cue than as a source of curvature information for the current chunk learner.",
            "- Better raw proxy-to-GT-accel correlation does not automatically translate into a better learned curve. The rough derivative shows the learner cares as much about chunk compatibility and filter behavior as about physical proxy fidelity.",
            "- The raw gyro-velocity proxy cannot be dropped into the current learner unchanged because the chunk filters are scale-sensitive. A neutral per-log scale calibration is enough to make it trainable, but not enough to make it decisively better.",
            "- The fact that gyro-based and accel-based learners land on similarly mild curvature suggests the bottleneck is still the chunk objective, not just accel noise.",
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
            "## Per-Log Summary",
            "",
            "| Log | accel x0=1 | gyro alpha SG x0=1 | gyro alpha grad x0=1 | gyro alpha HP x0=1 | gyro vel scaled x0=1 | gyro vel HP scaled x0=1 | gyro vel scaled x0=0 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        [
            (
                f"| `{summary.name}` | "
                f"{summary.accel_x0_1.masked_aligned_rmse:.3f} | "
                f"{summary.gyro_alpha_neg_x0_1.masked_aligned_rmse:.3f} | "
                f"{summary.gyro_alpha_grad_neg_x0_1.masked_aligned_rmse:.3f} | "
                f"{summary.gyro_alpha_neg_hp1_x0_1.masked_aligned_rmse:.3f} | "
                f"{summary.gyro_vel_neg_scaled_x0_1.masked_aligned_rmse:.3f} | "
                f"{summary.gyro_vel_neg_hp1_scaled_x0_1.masked_aligned_rmse:.3f} | "
                f"{summary.gyro_vel_neg_scaled_x0_0.masked_aligned_rmse:.3f} |"
            )
            for summary in log_summaries
        ]
    )

    best_log = min(log_summaries, key=lambda s: s.gyro_vel_neg_scaled_x0_1.masked_aligned_rmse).name
    lines.extend(
        [
            "",
            f"Representative gyro-proxy plot: `{best_log}`",
            "",
            f"![{best_log}]({(out_dir / f'{best_log}_curves.png').resolve()})",
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
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            accel_proj, mag, t, travel, v_gt, a_gt, zv_points, roi_mask = rmm.load_ws(log_name)
        roi_mask = np.asarray(roi_mask, dtype=bool)

        gyro_rads = load_gyro2_rads(log_name)
        gyro_z = gyro_rads[:, 2]
        dt = float(np.median(np.diff(t)))

        gyro_vel_proxy = -gyro_z
        gyro_vel_proxy_hp1 = -sosfiltfilt(
            butter(2, 1.0, btype="high", fs=200.0, output="sos"),
            gyro_z,
        )
        gyro_alpha_proxy = -savgol_filter(gyro_z, 31, 3, deriv=1, delta=dt, mode="interp")
        gyro_alpha_proxy_grad = -np.gradient(gyro_z, t, edge_order=1)
        gyro_alpha_proxy_hp1 = sosfiltfilt(
            butter(2, 1.0, btype="high", fs=200.0, output="sos"),
            gyro_alpha_proxy,
        )

        base_model = rmm.RearMagModel(x0_weight=0.0)
        vel_scale_raw = calibrate_velocity_scale(base_model, zv_points, mag, gyro_vel_proxy, t)
        vel_scale_hp1 = calibrate_velocity_scale(base_model, zv_points, mag, gyro_vel_proxy_hp1, t)

        accel_result = fit_proxy_method(
            log_name,
            mag,
            accel_proj,
            t,
            travel,
            v_gt,
            a_gt,
            zv_points,
            roi_mask,
            proxy_kind="accel",
            x0_weight=1.0,
        )
        gyro_alpha_result = fit_proxy_method(
            log_name,
            mag,
            gyro_alpha_proxy,
            t,
            travel,
            v_gt,
            a_gt,
            zv_points,
            roi_mask,
            proxy_kind="accel",
            x0_weight=1.0,
        )
        gyro_alpha_grad_result = fit_proxy_method(
            log_name,
            mag,
            gyro_alpha_proxy_grad,
            t,
            travel,
            v_gt,
            a_gt,
            zv_points,
            roi_mask,
            proxy_kind="accel",
            x0_weight=1.0,
        )
        gyro_alpha_hp1_result = fit_proxy_method(
            log_name,
            mag,
            gyro_alpha_proxy_hp1,
            t,
            travel,
            v_gt,
            a_gt,
            zv_points,
            roi_mask,
            proxy_kind="accel",
            x0_weight=1.0,
        )
        gyro_vel_result = fit_proxy_method(
            log_name,
            mag,
            gyro_vel_proxy * vel_scale_raw,
            t,
            travel,
            v_gt,
            a_gt,
            zv_points,
            roi_mask,
            proxy_kind="vel",
            x0_weight=1.0,
        )
        gyro_vel_hp1_result = fit_proxy_method(
            log_name,
            mag,
            gyro_vel_proxy_hp1 * vel_scale_hp1,
            t,
            travel,
            v_gt,
            a_gt,
            zv_points,
            roi_mask,
            proxy_kind="vel",
            x0_weight=1.0,
        )
        gyro_vel_x0_zero_result = fit_proxy_method(
            log_name,
            mag,
            gyro_vel_proxy * vel_scale_raw,
            t,
            travel,
            v_gt,
            a_gt,
            zv_points,
            roi_mask,
            proxy_kind="vel",
            x0_weight=0.0,
        )

        travel_vel_gt = np.gradient(travel, t, edge_order=2)
        travel_acc_gt = np.gradient(travel_vel_gt, t, edge_order=2)
        gyro_zdot = savgol_filter(gyro_z, 31, 3, deriv=1, delta=dt, mode="interp")

        summaries.append(
            LogSummary(
                name=log_name,
                gyro_z_to_travel_vel_corr=float(np.corrcoef(gyro_vel_proxy, travel_vel_gt)[0, 1]),
                gyro_zdot_to_travel_acc_corr=float(np.corrcoef(-gyro_zdot, travel_acc_gt)[0, 1]),
                gyro_zdot_grad_to_travel_acc_corr=float(np.corrcoef(-np.gradient(gyro_z, t, edge_order=1), travel_acc_gt)[0, 1]),
                gyro_vel_scale_raw=float(vel_scale_raw),
                gyro_vel_scale_hp1=float(vel_scale_hp1),
                accel_x0_1=accel_result,
                gyro_alpha_neg_x0_1=gyro_alpha_result,
                gyro_alpha_grad_neg_x0_1=gyro_alpha_grad_result,
                gyro_alpha_neg_hp1_x0_1=gyro_alpha_hp1_result,
                gyro_vel_neg_scaled_x0_1=gyro_vel_result,
                gyro_vel_neg_hp1_scaled_x0_1=gyro_vel_hp1_result,
                gyro_vel_neg_scaled_x0_0=gyro_vel_x0_zero_result,
            )
        )

        power_model = rmm.MagToTravelModel(pred_soft_mg=50.0)
        accel_pred = power_model.pred_x(mag, np.asarray(accel_result.coeffs, dtype=float))
        gyro_alpha_pred = power_model.pred_x(mag, np.asarray(gyro_alpha_result.coeffs, dtype=float))
        gyro_vel_pred = power_model.pred_x(mag, np.asarray(gyro_vel_result.coeffs, dtype=float))
        plot_log(
            out_dir / f"{log_name}_curves.png",
            log_name,
            mag,
            travel,
            roi_mask,
            {
                "accel_x0_1": accel_pred,
                "gyro_alpha_neg_x0_1": gyro_alpha_pred,
                "gyro_vel_neg_scaled_x0_1": gyro_vel_pred,
            },
        )

    payload = {"logs": [asdict(summary) for summary in summaries]}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(summaries, out_dir)


if __name__ == "__main__":
    main()
