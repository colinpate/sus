#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, savgol_filter, sosfilt, sosfiltfilt
from sklearn.isotonic import IsotonicRegression


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mag_to_travel_model_core import MagToTravelModelCore  # noqa: E402


ACTIVE_THRESH_MS2 = 0.5
SINGLE_AXIS_HP_FC_HZ = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
DIFF_HP_FC_HZ = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
GRAVITY_LP_FC_HZ = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
COMPLEMENTARY_FC_HZ = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
HINGE_AXIS_SENSOR = np.array([0.0, 0.0, 1.0], dtype=float)
NO_GT_Z_HP_FC_HZ = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
NO_GT_SELF_SUP_GRAVITY_FC_HZ = [0.05, 0.1, 0.2]
MAG_DERIV_WINDOWS = [7, 9, 11, 15, 21, 31]
MAG_SELF_CAL_HP_FC_HZ = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
MAG_PROXY_AXIS = 2
MAG_PROXY_SIGN = 1.0
MAG_TRAVEL_SIGN = -1.0
MAG_CHUNK_HALF_WINDOW_S = 0.10


@dataclass
class Metrics:
    n: int
    rmse: float
    mae: float
    corr: float


@dataclass
class LogData:
    name: str
    t: np.ndarray
    fs_hz: float
    accel_lis1_ms2: np.ndarray
    accel_lis2_ms2: np.ndarray
    gyro2_dps: np.ndarray
    travel_mm: np.ndarray
    travel_accel_gt_ms2: np.ndarray
    mag_proj: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze rear IMU travel-acceleration estimators")
    parser.add_argument(
        "--logs",
        nargs="*",
        default=["log136_rear", "log137_rear"],
        help="Log names without extension",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/rear_accel_exploration"),
        help="Directory for summary JSON and plots",
    )
    parser.add_argument(
        "--decimate",
        type=int,
        default=4,
        help="Additional decimation factor applied after cached 200 Hz signals",
    )
    return parser.parse_args()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-9)
    return x / denom


def filter_signal(x: np.ndarray, fs_hz: float, fc_hz: float, btype: str, *, zero_phase: bool) -> np.ndarray:
    sos = butter(4, fc_hz, btype=btype, fs=fs_hz, output="sos")
    if zero_phase:
        return sosfiltfilt(sos, x, axis=0)
    return sosfilt(sos, x, axis=0)


def active_metrics(pred: np.ndarray, gt: np.ndarray, thresh_ms2: float = ACTIVE_THRESH_MS2) -> Metrics:
    mask = np.isfinite(pred) & np.isfinite(gt) & (np.abs(gt) >= thresh_ms2)
    if int(np.sum(mask)) < 10:
        raise ValueError("Not enough active samples for metrics")
    pred_masked = pred[mask]
    gt_masked = gt[mask]
    err = pred[mask] - gt[mask]
    pred_std = float(np.std(pred_masked))
    gt_std = float(np.std(gt_masked))
    if pred_std < 1e-12 or gt_std < 1e-12:
        corr = None
    else:
        corr = float(np.corrcoef(pred_masked, gt_masked)[0, 1])
    return Metrics(
        n=int(np.sum(mask)),
        rmse=float(np.sqrt(np.mean(err**2))),
        mae=float(np.mean(np.abs(err))),
        corr=corr,
    )


def combined_active_metrics(preds: list[np.ndarray], gts: list[np.ndarray]) -> Metrics:
    pred = np.concatenate(preds)
    gt = np.concatenate(gts)
    return active_metrics(pred, gt)


def signal_metrics(pred: np.ndarray, gt: np.ndarray, *, center: bool = False) -> Metrics:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    gt = np.asarray(gt, dtype=float).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(gt)
    if int(np.sum(mask)) < 10:
        raise ValueError("Not enough finite samples for metrics")
    pred_masked = pred[mask]
    gt_masked = gt[mask]
    if center:
        pred_masked = pred_masked - np.mean(pred_masked)
        gt_masked = gt_masked - np.mean(gt_masked)
    err = pred_masked - gt_masked
    pred_std = float(np.std(pred_masked))
    gt_std = float(np.std(gt_masked))
    if pred_std < 1e-12 or gt_std < 1e-12:
        corr = None
    else:
        corr = float(np.corrcoef(pred_masked, gt_masked)[0, 1])
    return Metrics(
        n=int(np.sum(mask)),
        rmse=float(np.sqrt(np.mean(err**2))),
        mae=float(np.mean(np.abs(err))),
        corr=corr,
    )


def gyro_needs_flip(df: pd.DataFrame, accel_sensor: str) -> bool:
    return float(df[f"{accel_sensor}_x"].mean()) > 0.0


def load_log(log_name: str, decimate: int) -> LogData:
    cache = np.load(f"backend/run_artifacts/{log_name}/cache/all.npz")
    df = pd.read_csv(f"logs/{log_name}.csv")

    sl = slice(None, None, decimate)
    t = cache["accel/lpf/lis2__t"][sl]
    fs_hz = float(1.0 / np.median(np.diff(t)))
    accel_lis1_ms2 = cache["accel/lpf/lis1__x"][sl]
    accel_lis2_ms2 = cache["accel/lpf/lis2__x"][sl]
    gyro2_dps = cache["gyro/lpf/gyro2__x"][sl]
    if gyro_needs_flip(df, "lis2"):
        gyro2_dps = gyro2_dps @ np.diag([-1.0, -1.0, 1.0])

    travel_mm = cache["travel__x"][sl, 0]
    vel_gt = np.gradient(travel_mm, t, edge_order=2)
    travel_accel_gt_ms2 = np.gradient(vel_gt, t, edge_order=2) / 1000.0
    mag_proj = cache["mag/proj/lpf__x"][sl, 0]

    return LogData(
        name=log_name,
        t=t,
        fs_hz=fs_hz,
        accel_lis1_ms2=accel_lis1_ms2,
        accel_lis2_ms2=accel_lis2_ms2,
        gyro2_dps=gyro2_dps,
        travel_mm=travel_mm,
        travel_accel_gt_ms2=travel_accel_gt_ms2,
        mag_proj=mag_proj,
    )


def second_derivative_mm_to_ms2(x_mm: np.ndarray, t: np.ndarray, window_length: int) -> np.ndarray:
    x_mm = np.asarray(x_mm, dtype=float).reshape(-1)
    if len(x_mm) < 5:
        raise ValueError("Need at least 5 samples for second derivative")
    window = int(window_length)
    if window % 2 == 0:
        window += 1
    window = max(5, window)
    max_window = len(x_mm) if len(x_mm) % 2 == 1 else len(x_mm) - 1
    window = min(window, max_window)
    if window < 5:
        raise ValueError("Signal too short for Savitzky-Golay derivative")
    dt_s = float(np.median(np.diff(t)))
    return savgol_filter(
        x_mm,
        window_length=window,
        polyorder=3,
        deriv=2,
        delta=dt_s,
        mode="interp",
    ) / 1000.0


def find_mag_zv_points(mag_proj: np.ndarray) -> np.ndarray:
    mag = np.asarray(mag_proj, dtype=float).reshape(-1)
    local_max = (np.diff(np.sign(np.diff(mag))) < 0).nonzero()[0] + 1
    local_min = (np.diff(np.sign(np.diff(mag))) > 0).nonzero()[0] + 1
    return np.sort(np.concatenate((local_max, local_min)))


def estimate_mag_baseline(mag_proj: np.ndarray, accel_proxy_ms2: np.ndarray, fs_hz: float) -> float:
    still_len = max(1, int(round(0.1 * fs_hz)))
    still_mags = []
    for start in range(0, len(mag_proj) - still_len, still_len):
        mag_chunk = mag_proj[start : start + still_len]
        accel_chunk_mm_s2 = accel_proxy_ms2[start : start + still_len] * 1000.0
        if np.max(np.abs(accel_chunk_mm_s2)) < 1000.0:
            still_mags.append(mag_chunk)

    if not still_mags:
        return float(np.percentile(mag_proj, 10.0))

    still_vals = np.concatenate(still_mags)
    return float(np.median(still_vals) + np.std(still_vals))


def mag_chunk_len(fs_hz: float, half_window_s: float = MAG_CHUNK_HALF_WINDOW_S) -> int:
    return max(3, int(round(fs_hz * half_window_s)))


def fit_mag_self_calibrated_travel(log: LogData, accel_hp_fc_hz: float) -> tuple[np.ndarray, dict[str, Any]]:
    accel_proxy_ms2 = MAG_PROXY_SIGN * filter_signal(
        log.accel_lis2_ms2[:, MAG_PROXY_AXIS],
        log.fs_hz,
        accel_hp_fc_hz,
        "high",
        zero_phase=True,
    )
    mag_zv_points = find_mag_zv_points(log.mag_proj)
    baseline = estimate_mag_baseline(log.mag_proj, accel_proxy_ms2, log.fs_hz)
    model_core = MagToTravelModelCore(chunk_len=mag_chunk_len(log.fs_hz))

    with contextlib.redirect_stdout(io.StringIO()):
        training_data = model_core.create_training_data(
            mag=log.mag_proj,
            accel=accel_proxy_ms2,
            train_mask=np.zeros_like(log.mag_proj, dtype=bool),
            t=log.t,
            baseline_min_mag=baseline,
            idxs=mag_zv_points,
        )
        if training_data.shape[0] == 0:
            raise ValueError(f"No self-calibration chunks found for {log.name}")
        result = model_core.train(training_data)

    pred_travel_mm = model_core.model.pred_x(log.mag_proj)
    meta = {
        "accel_hp_fc_hz": accel_hp_fc_hz,
        "proxy_axis": MAG_PROXY_AXIS,
        "proxy_sign": MAG_PROXY_SIGN,
        "travel_sign": MAG_TRAVEL_SIGN,
        "mag_baseline": baseline,
        "chunk_half_window_s": MAG_CHUNK_HALF_WINDOW_S,
        "chunk_len_samples": mag_chunk_len(log.fs_hz),
        "n_training_chunks": int(len(model_core.chunks)),
        "n_zv_points": int(len(mag_zv_points)),
        "model_coeffs": result.x.tolist(),
    }
    return MAG_TRAVEL_SIGN * pred_travel_mm, meta


def predict_mag_isotonic_upper_bound(
    target_log: LogData,
    logs: list[LogData],
    derivative_window: int,
) -> np.ndarray:
    if len(logs) == 1:
        train_log = target_log
    else:
        train_log = next(log for log in logs if log.name != target_log.name)
    increasing = float(np.corrcoef(train_log.mag_proj, train_log.travel_mm)[0, 1]) > 0.0
    model = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
    model.fit(train_log.mag_proj, train_log.travel_mm)
    pred_travel_mm = model.predict(target_log.mag_proj)
    return second_derivative_mm_to_ms2(pred_travel_mm, target_log.t, derivative_window)


def predict_mag_self_calibrated(log: LogData, accel_hp_fc_hz: float, derivative_window: int) -> np.ndarray:
    pred_travel_mm, _ = fit_mag_self_calibrated_travel(log, accel_hp_fc_hz)
    return second_derivative_mm_to_ms2(pred_travel_mm, log.t, derivative_window)


def estimate_gravity_complementary(accel_ms2: np.ndarray, gyro_dps: np.ndarray, fs_hz: float, fc_hz: float) -> np.ndarray:
    dt_s = 1.0 / fs_hz
    omega_rads = np.radians(gyro_dps)
    gain = 2.0 * np.pi * fc_hz

    gravity = -accel_ms2[0]
    gravity = gravity / max(np.linalg.norm(gravity), 1e-9)
    gravity_hist = np.zeros_like(accel_ms2)

    for i in range(len(accel_ms2)):
        if i > 0:
            gravity = gravity - dt_s * np.cross(omega_rads[i], gravity)
            gravity = gravity / max(np.linalg.norm(gravity), 1e-9)

        accel_dir = -accel_ms2[i]
        accel_norm = np.linalg.norm(accel_dir)
        if accel_norm > 1e-9:
            accel_dir = accel_dir / accel_norm
            err = np.cross(gravity, accel_dir)
            gravity = gravity - dt_s * gain * err
            gravity = gravity / max(np.linalg.norm(gravity), 1e-9)

        gravity_hist[i] = gravity

    return gravity_hist


def gravity_basis_features(
    accel_ms2: np.ndarray,
    gravity_hat: np.ndarray,
    hinge_axis_sensor: np.ndarray = HINGE_AXIS_SENSOR,
) -> np.ndarray:
    hinge = hinge_axis_sensor / max(np.linalg.norm(hinge_axis_sensor), 1e-9)
    hinge_hist = np.broadcast_to(hinge, gravity_hat.shape)
    tangent_hat = normalize_rows(np.cross(hinge_hist, gravity_hat))
    normal_hat = normalize_rows(np.cross(gravity_hat, tangent_hat))
    linear_accel = accel_ms2 + 9.81 * gravity_hat

    return np.column_stack(
        (
            np.sum(linear_accel * gravity_hat, axis=1),
            np.sum(linear_accel * tangent_hat, axis=1),
            np.sum(linear_accel * normal_hat, axis=1),
        )
    )


def basis_predictions(
    data: LogData,
    fc_hz: float,
    mode: str,
    weights: np.ndarray | None = None,
    hinge_axis_sensor: np.ndarray = HINGE_AXIS_SENSOR,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "zero_phase":
        gravity_hat = normalize_rows(
            -filter_signal(data.accel_lis2_ms2, data.fs_hz, fc_hz, "low", zero_phase=True)
        )
    elif mode == "causal":
        gravity_hat = normalize_rows(
            -filter_signal(data.accel_lis2_ms2, data.fs_hz, fc_hz, "low", zero_phase=False)
        )
    elif mode == "complementary":
        gravity_hat = estimate_gravity_complementary(
            data.accel_lis2_ms2,
            data.gyro2_dps,
            data.fs_hz,
            fc_hz,
        )
    else:
        raise ValueError(f"Unknown gravity mode {mode}")

    features = gravity_basis_features(data.accel_lis2_ms2, gravity_hat, hinge_axis_sensor=hinge_axis_sensor)
    if weights is None:
        weights, *_ = np.linalg.lstsq(features, data.travel_accel_gt_ms2, rcond=None)
    pred = features @ weights
    return pred, weights


def combine_features(feature_list: list[np.ndarray], gt_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate(feature_list, axis=0), np.concatenate(gt_list, axis=0)


def evaluate_single_axis_hp(logs: list[LogData]) -> dict[str, Any]:
    result: dict[str, Any] = {"kind": "single_axis_hp", "per_log_best": {}}
    for log in logs:
        best: dict[str, Any] | None = None
        for axis in range(3):
            for fc_hz in SINGLE_AXIS_HP_FC_HZ:
                pred = filter_signal(log.accel_lis2_ms2[:, axis], log.fs_hz, fc_hz, "high", zero_phase=True)
                metrics = active_metrics(pred, log.travel_accel_gt_ms2)
                row = {
                    "axis": axis,
                    "fc_hz": fc_hz,
                    "metrics": asdict(metrics),
                }
                if best is None or metrics.rmse < best["metrics"]["rmse"]:
                    best = row
        result["per_log_best"][log.name] = best
    return result


def evaluate_dual_accel_diff_leave_one_out(logs: list[LogData]) -> dict[str, Any]:
    assert len(logs) == 2, "This helper currently expects exactly two logs"
    best: dict[str, Any] | None = None
    trials: list[dict[str, Any]] = []
    for fc_hz in DIFF_HP_FC_HZ:
        eval_rows = []
        for train_idx, test_idx in ((0, 1), (1, 0)):
            train = logs[train_idx]
            test = logs[test_idx]
            train_feat = filter_signal(
                train.accel_lis2_ms2 - train.accel_lis1_ms2,
                train.fs_hz,
                fc_hz,
                "high",
                zero_phase=True,
            )
            weights, *_ = np.linalg.lstsq(train_feat, train.travel_accel_gt_ms2, rcond=None)
            test_feat = filter_signal(
                test.accel_lis2_ms2 - test.accel_lis1_ms2,
                test.fs_hz,
                fc_hz,
                "high",
                zero_phase=True,
            )
            pred = test_feat @ weights
            eval_rows.append(
                {
                    "train": train.name,
                    "test": test.name,
                    "weights": weights.tolist(),
                    "metrics": asdict(active_metrics(pred, test.travel_accel_gt_ms2)),
                }
            )

        mean_rmse = float(np.mean([row["metrics"]["rmse"] for row in eval_rows]))
        trial = {"fc_hz": fc_hz, "leave_one_out": eval_rows, "mean_rmse": mean_rmse}
        trials.append(trial)
        if best is None or mean_rmse < best["mean_rmse"]:
            best = trial

    return {"kind": "dual_accel_diff_hp", "best": best, "trials": trials}


def evaluate_gravity_basis(logs: list[LogData], mode: str) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for fc_hz in (COMPLEMENTARY_FC_HZ if mode == "complementary" else GRAVITY_LP_FC_HZ):
        feature_list = []
        gt_list = []
        per_log_feat: dict[str, np.ndarray] = {}
        for log in logs:
            if mode == "zero_phase":
                gravity_hat = normalize_rows(
                    -filter_signal(log.accel_lis2_ms2, log.fs_hz, fc_hz, "low", zero_phase=True)
                )
            elif mode == "causal":
                gravity_hat = normalize_rows(
                    -filter_signal(log.accel_lis2_ms2, log.fs_hz, fc_hz, "low", zero_phase=False)
                )
            else:
                gravity_hat = estimate_gravity_complementary(
                    log.accel_lis2_ms2,
                    log.gyro2_dps,
                    log.fs_hz,
                    fc_hz,
                )

            features = gravity_basis_features(log.accel_lis2_ms2, gravity_hat)
            per_log_feat[log.name] = features
            feature_list.append(features)
            gt_list.append(log.travel_accel_gt_ms2)

        global_feat, global_gt = combine_features(feature_list, gt_list)
        weights, *_ = np.linalg.lstsq(global_feat, global_gt, rcond=None)

        per_log_metrics = {}
        for log in logs:
            pred = per_log_feat[log.name] @ weights
            per_log_metrics[log.name] = asdict(active_metrics(pred, log.travel_accel_gt_ms2))

        global_metrics = asdict(active_metrics(global_feat @ weights, global_gt))

        leave_one_out = []
        if len(logs) == 2:
            for train_idx, test_idx in ((0, 1), (1, 0)):
                train = logs[train_idx]
                test = logs[test_idx]
                weights_lolo, *_ = np.linalg.lstsq(
                    per_log_feat[train.name], train.travel_accel_gt_ms2, rcond=None
                )
                pred = per_log_feat[test.name] @ weights_lolo
                leave_one_out.append(
                    {
                        "train": train.name,
                        "test": test.name,
                        "weights": weights_lolo.tolist(),
                        "metrics": asdict(active_metrics(pred, test.travel_accel_gt_ms2)),
                    }
                )

        trial = {
            "fc_hz": fc_hz,
            "weights": weights.tolist(),
            "global_metrics": global_metrics,
            "per_log_metrics": per_log_metrics,
            "leave_one_out": leave_one_out,
        }
        trials.append(trial)

        if best is None or global_metrics["rmse"] < best["global_metrics"]["rmse"]:
            best = trial

    return {
        "kind": f"gravity_basis_{mode}",
        "hinge_axis_sensor": HINGE_AXIS_SENSOR.tolist(),
        "best": best,
        "trials": trials,
    }


def evaluate_no_gt_fixed_z_hp(logs: list[LogData]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    trials: list[dict[str, Any]] = []

    for fc_hz in NO_GT_Z_HP_FC_HZ:
        preds = []
        gts = []
        per_log = {}
        for log in logs:
            pred = filter_signal(log.accel_lis2_ms2[:, 2], log.fs_hz, fc_hz, "high", zero_phase=True)
            preds.append(pred)
            gts.append(log.travel_accel_gt_ms2)
            per_log[log.name] = asdict(active_metrics(pred, log.travel_accel_gt_ms2))

        trial = {
            "fc_hz": fc_hz,
            "global_metrics": asdict(combined_active_metrics(preds, gts)),
            "per_log_metrics": per_log,
        }
        trials.append(trial)
        if best is None or trial["global_metrics"]["rmse"] < best["global_metrics"]["rmse"]:
            best = trial

    return {"kind": "no_gt_fixed_z_hp", "best": best, "trials": trials}


def self_supervised_alpha_prediction(log: LogData, gravity_fc_hz: float) -> tuple[np.ndarray, np.ndarray]:
    gravity_hat = normalize_rows(
        -filter_signal(log.accel_lis2_ms2, log.fs_hz, gravity_fc_hz, "low", zero_phase=True)
    )
    features = gravity_basis_features(log.accel_lis2_ms2, gravity_hat)
    gyro_z_rads = np.radians(log.gyro2_dps[:, 2])
    dt_s = float(np.median(np.diff(log.t)))
    alpha = savgol_filter(
        gyro_z_rads,
        window_length=31,
        polyorder=3,
        deriv=1,
        delta=dt_s,
        mode="interp",
    )

    cov = np.cov(features, rowvar=False)
    direction = np.linalg.solve(cov + (1e-6 * np.eye(features.shape[1])), features.T @ alpha)
    direction = direction / max(np.linalg.norm(direction), 1e-9)
    pred = features @ direction

    # Sign is not observable from IMU-only self-supervision alone.
    # Use the convention that predicted travel acceleration should have positive correlation
    # with positive hinge angular acceleration.
    pred_alpha_corr = np.corrcoef(pred, alpha)[0, 1]
    if np.isfinite(pred_alpha_corr) and pred_alpha_corr < 0:
        pred = -pred
        direction = -direction

    return pred, direction


def evaluate_no_gt_self_supervised_alpha(logs: list[LogData]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    trials: list[dict[str, Any]] = []

    for fc_hz in NO_GT_SELF_SUP_GRAVITY_FC_HZ:
        preds = []
        gts = []
        per_log = {}
        for log in logs:
            pred, direction = self_supervised_alpha_prediction(log, fc_hz)
            preds.append(pred)
            gts.append(log.travel_accel_gt_ms2)
            per_log[log.name] = {
                "direction": direction.tolist(),
                "metrics": asdict(active_metrics(pred, log.travel_accel_gt_ms2)),
            }

        trial = {
            "gravity_fc_hz": fc_hz,
            "global_metrics": asdict(combined_active_metrics(preds, gts)),
            "per_log": per_log,
        }
        trials.append(trial)
        if best is None or trial["global_metrics"]["rmse"] < best["global_metrics"]["rmse"]:
            best = trial

    return {"kind": "no_gt_self_supervised_alpha", "best": best, "trials": trials}


def evaluate_mag_supervised_isotonic(logs: list[LogData]) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for derivative_window in MAG_DERIV_WINDOWS:
        same_log = {}
        same_preds = []
        same_gts = []

        for log in logs:
            increasing = float(np.corrcoef(log.mag_proj, log.travel_mm)[0, 1]) > 0.0
            model = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
            pred_travel_mm = model.fit_transform(log.mag_proj, log.travel_mm)
            pred_accel_ms2 = second_derivative_mm_to_ms2(pred_travel_mm, log.t, derivative_window)
            same_log[log.name] = {
                "increasing": increasing,
                "travel_metrics": asdict(signal_metrics(pred_travel_mm, log.travel_mm)),
                "accel_metrics": asdict(active_metrics(pred_accel_ms2, log.travel_accel_gt_ms2)),
            }
            same_preds.append(pred_accel_ms2)
            same_gts.append(log.travel_accel_gt_ms2)

        leave_one_out = []
        lolo_preds = []
        lolo_gts = []
        if len(logs) == 2:
            for train_idx, test_idx in ((0, 1), (1, 0)):
                train = logs[train_idx]
                test = logs[test_idx]
                increasing = float(np.corrcoef(train.mag_proj, train.travel_mm)[0, 1]) > 0.0
                model = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
                model.fit(train.mag_proj, train.travel_mm)
                pred_travel_mm = model.predict(test.mag_proj)
                pred_accel_ms2 = second_derivative_mm_to_ms2(pred_travel_mm, test.t, derivative_window)
                leave_one_out.append(
                    {
                        "train": train.name,
                        "test": test.name,
                        "increasing": increasing,
                        "travel_metrics": asdict(signal_metrics(pred_travel_mm, test.travel_mm)),
                        "accel_metrics": asdict(active_metrics(pred_accel_ms2, test.travel_accel_gt_ms2)),
                    }
                )
                lolo_preds.append(pred_accel_ms2)
                lolo_gts.append(test.travel_accel_gt_ms2)

        global_metrics = (
            combined_active_metrics(lolo_preds, lolo_gts) if lolo_preds else combined_active_metrics(same_preds, same_gts)
        )
        trial = {
            "derivative_window": derivative_window,
            "same_log": same_log,
            "leave_one_out": leave_one_out,
            "global_accel_metrics": asdict(global_metrics),
        }
        trials.append(trial)
        if best is None or global_metrics.rmse < best["global_accel_metrics"]["rmse"]:
            best = trial

    return {
        "kind": "mag_supervised_isotonic",
        "note": "Upper bound only: fits mag-to-travel from GT travel.",
        "best": best,
        "trials": trials,
    }


def evaluate_mag_self_calibrated(logs: list[LogData]) -> dict[str, Any]:
    cached_travel_preds: dict[tuple[str, float], tuple[np.ndarray, dict[str, Any]]] = {}
    for fc_hz in MAG_SELF_CAL_HP_FC_HZ:
        for log in logs:
            cached_travel_preds[(log.name, fc_hz)] = fit_mag_self_calibrated_travel(log, fc_hz)

    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for fc_hz in MAG_SELF_CAL_HP_FC_HZ:
        for derivative_window in MAG_DERIV_WINDOWS:
            pred_accels = []
            gt_accels = []
            pred_travels = []
            gt_travels = []
            per_log = {}

            for log in logs:
                pred_travel_mm, meta = cached_travel_preds[(log.name, fc_hz)]
                pred_accel_ms2 = second_derivative_mm_to_ms2(pred_travel_mm, log.t, derivative_window)
                pred_accels.append(pred_accel_ms2)
                gt_accels.append(log.travel_accel_gt_ms2)
                pred_travels.append(pred_travel_mm)
                gt_travels.append(log.travel_mm)
                per_log[log.name] = {
                    **meta,
                    "travel_metrics_centered": asdict(signal_metrics(pred_travel_mm, log.travel_mm, center=True)),
                    "accel_metrics": asdict(active_metrics(pred_accel_ms2, log.travel_accel_gt_ms2)),
                }

            trial = {
                "accel_hp_fc_hz": fc_hz,
                "derivative_window": derivative_window,
                "global_travel_metrics_centered": asdict(
                    signal_metrics(
                        np.concatenate([pred - np.mean(pred) for pred in pred_travels]),
                        np.concatenate([gt - np.mean(gt) for gt in gt_travels]),
                    )
                ),
                "global_accel_metrics": asdict(combined_active_metrics(pred_accels, gt_accels)),
                "per_log": per_log,
            }
            trials.append(trial)
            if best is None or trial["global_accel_metrics"]["rmse"] < best["global_accel_metrics"]["rmse"]:
                best = trial

    return {
        "kind": "mag_self_calibrated",
        "note": (
            "No travel GT in the fit. Uses high-passed axle IMU z acceleration with a fixed sign "
            "convention for compression-positive motion, then learns a monotonic mag-to-travel curve per ride."
        ),
        "best": best,
        "trials": trials,
    }


def select_activity_window(gt: np.ndarray, fs_hz: float, window_s: float = 20.0) -> slice:
    win = max(1, int(round(window_s * fs_hz)))
    energy = np.convolve(gt**2, np.ones(win), mode="same")
    center = int(np.argmax(energy))
    start = max(0, center - win // 2)
    stop = min(len(gt), start + win)
    start = max(0, stop - win)
    return slice(start, stop)


def plot_candidate_comparison(
    out_dir: Path,
    log: LogData,
    basis_zero_phase: dict[str, Any],
    no_gt_fixed_z_best: dict[str, Any],
    mag_supervised_isotonic_best: dict[str, Any],
    mag_self_calibrated_best: dict[str, Any],
    logs: list[LogData],
) -> None:
    gt = log.travel_accel_gt_ms2
    sl = select_activity_window(gt, log.fs_hz)

    pred_zero, _ = basis_predictions(
        log,
        float(basis_zero_phase["fc_hz"]),
        "zero_phase",
        weights=np.asarray(basis_zero_phase["weights"]),
    )
    pred_fixed_z = filter_signal(
        log.accel_lis2_ms2[:, 2],
        log.fs_hz,
        float(no_gt_fixed_z_best["fc_hz"]),
        "high",
        zero_phase=True,
    )
    pred_mag_upper = predict_mag_isotonic_upper_bound(
        log,
        logs,
        int(mag_supervised_isotonic_best["derivative_window"]),
    )
    pred_mag_self = predict_mag_self_calibrated(
        log,
        float(mag_self_calibrated_best["accel_hp_fc_hz"]),
        int(mag_self_calibrated_best["derivative_window"]),
    )

    plt.figure(figsize=(13, 6))
    plt.plot(log.t[sl], gt[sl], label="ground truth", linewidth=2.0, color="black")
    plt.plot(log.t[sl], pred_zero[sl], label="gravity-basis zero-phase", linewidth=1.2)
    plt.plot(log.t[sl], pred_mag_upper[sl], label="mag isotonic upper bound", linewidth=1.2)
    plt.plot(log.t[sl], pred_mag_self[sl], label="mag self-calibrated", linewidth=1.2)
    plt.plot(log.t[sl], pred_fixed_z[sl], label="fixed z-axis HP", linewidth=1.0, alpha=0.9)
    plt.title(f"{log.name}: travel acceleration estimate comparison")
    plt.xlabel("time (s)")
    plt.ylabel("travel acceleration (m/s²)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{log.name}_comparison.png", dpi=160)
    plt.close()


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Metrics):
        return asdict(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {key: to_jsonable(val) for key, val in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(val) for val in obj]
    return obj


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logs = [load_log(log_name, args.decimate) for log_name in args.logs]

    summary: dict[str, Any] = {
        "logs": {
            log.name: {
                "n_samples": int(len(log.t)),
                "fs_hz": float(log.fs_hz),
                "travel_accel_zero_predictor": asdict(active_metrics(np.zeros_like(log.travel_accel_gt_ms2), log.travel_accel_gt_ms2)),
            }
            for log in logs
        },
        "single_axis_hp": evaluate_single_axis_hp(logs),
        "dual_accel_diff_hp": evaluate_dual_accel_diff_leave_one_out(logs),
        "gravity_basis_zero_phase": evaluate_gravity_basis(logs, "zero_phase"),
        "gravity_basis_causal": evaluate_gravity_basis(logs, "causal"),
        "gravity_basis_complementary": evaluate_gravity_basis(logs, "complementary"),
        "mag_supervised_isotonic": evaluate_mag_supervised_isotonic(logs),
        "mag_self_calibrated": evaluate_mag_self_calibrated(logs),
        "no_gt_fixed_z_hp": evaluate_no_gt_fixed_z_hp(logs),
        "no_gt_self_supervised_alpha": evaluate_no_gt_self_supervised_alpha(logs),
    }

    with open(args.out_dir / "summary.json", "w") as fo:
        json.dump(to_jsonable(summary), fo, indent=2)

    for log in logs:
        plot_candidate_comparison(
            args.out_dir,
            log,
            summary["gravity_basis_zero_phase"]["best"],
            summary["no_gt_fixed_z_hp"]["best"],
            summary["mag_supervised_isotonic"]["best"],
            summary["mag_self_calibrated"]["best"],
            logs,
        )

    print(f"Wrote summary to {args.out_dir / 'summary.json'}")
    for log in logs:
        print(f"Wrote plot to {args.out_dir / f'{log.name}_comparison.png'}")


if __name__ == "__main__":
    main()
