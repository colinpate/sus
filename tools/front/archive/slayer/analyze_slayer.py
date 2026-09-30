#!/usr/bin/env python3
"""Reproducible sensor-health and RMSE analysis for the Slayer import."""

from __future__ import annotations

import csv
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from scipy.signal import butter, sosfiltfilt


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "reports" / "slayer_deep_dive"
REGISTRY_PATH = ROOT / "logs" / "registry.toml"
CACHE_ROOT = ROOT / "backend" / "run_artifacts"
SLAYER_LOGS = [f"log-{number:04d}" for number in range(144, 156)]
SUCCESSFUL_SLAYER = [
    "log-0145",
    "log-0146",
    "log-0148",
    "log-0149",
    "log-0150",
    "log-0151",
    "log-0152",
    "log-0155",
]
PREDICTIONS = {
    "mag_raw": "travel/mag_model",
    "mag_adjusted": "travel/mag_model/adj",
    "fusion1_baseline": "travel/fusion1",
    "nuisance_delta_lifted": "travel/solved/mag_nuisance/delta_lifted",
    "nuisance_mag_observation": "travel/mag_nuisance/corrected",
    "nuisance_fusion2": "travel/solved",
}


def csv_path(entry: dict) -> Path:
    filename = Path(entry["file"])
    candidates = [ROOT / "logs" / filename, ROOT / "logs" / "converted" / filename]
    return next(path for path in candidates if path.is_file())


def max_true_run(mask: np.ndarray, sample_interval_s: float) -> float:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return 0.0
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[cuts + 1]]
    ends = np.r_[idx[cuts], idx[-1]]
    return float(np.max(ends - starts + 1) * sample_interval_s)


def count_true_runs(mask: np.ndarray) -> int:
    idx = np.flatnonzero(mask)
    return 0 if len(idx) == 0 else int(1 + np.sum(np.diff(idx) > 1))


def sensor_audit(log_id: str, entry: dict) -> dict[str, object]:
    df = pd.read_csv(csv_path(entry))
    t = df["t_s"].to_numpy(float)
    dt = float(np.median(np.diff(t)))

    def vec(columns: list[str]) -> np.ndarray:
        return df[columns].to_numpy(float)

    a1_raw = vec(["lis1_x", "lis1_y", "lis1_z"])
    a2_raw = vec(["lis2_x", "lis2_y", "lis2_z"])
    g1_raw = vec(["gyro1_dps10_x", "gyro1_dps10_y", "gyro1_dps10_z"])
    g2_raw = vec(["gyro2_dps10_x", "gyro2_dps10_y", "gyro2_dps10_z"])
    mag_raw = vec(["mmc_mG_x", "mmc_mG_y", "mmc_mG_z"])
    lis_mag_raw = vec(["lis3mdl_mG_x", "lis3mdl_mG_y", "lis3mdl_mG_z"])
    angle = df["angle_raw"].to_numpy(int)

    a1_zero = np.all(a1_raw == 0, axis=1)
    a2_zero = np.all(a2_raw == 0, axis=1)
    g1_zero = np.all(g1_raw == 0, axis=1)
    g2_zero = np.all(g2_raw == 0, axis=1)
    mag_zero = np.all(mag_raw == 0, axis=1)
    lis_mag_zero = np.all(lis_mag_raw == 0, axis=1)
    angle_rail = np.isin(angle, (0, 4095))
    angle_wrapped_side = angle > 2048

    # Remove per-axis bias before using gyro motion as a ride/no-ride indicator.
    gyro_dps = g1_raw * 0.1
    gyro_dynamic = gyro_dps - np.median(gyro_dps, axis=0)
    gyro_dynamic_rms = float(np.sqrt(np.mean(np.sum(gyro_dynamic**2, axis=1))))
    signed_counts = ((angle.astype(float) + 2048.0) % 4096.0) - 2048.0
    angle_span_deg = float(
        (np.percentile(signed_counts, 99) - np.percentile(signed_counts, 1))
        * 360.0
        / 4096.0
    )
    ride_like = gyro_dynamic_rms > 5.0 and angle_span_deg > 2.0

    return {
        "log": log_id,
        "records": len(df),
        "duration_s": float(t[-1] - t[0]),
        "sample_rate_hz": 1.0 / dt,
        "ride_like": ride_like,
        "gyro1_dynamic_rms_dps": gyro_dynamic_rms,
        "angle_robust_span_deg": angle_span_deg,
        "angle_unique_values": int(np.unique(angle).size),
        "angle_wrap_side_pct": 100.0 * float(np.mean(angle_wrapped_side)),
        "angle_rail_pct": 100.0 * float(np.mean(angle_rail)),
        "angle_rail_longest_s": max_true_run(angle_rail, dt),
        "lis1_zero_pct": 100.0 * float(np.mean(a1_zero)),
        "gyro1_zero_pct": 100.0 * float(np.mean(g1_zero)),
        "primary_mag_zero_pct": 100.0 * float(np.mean(mag_zero)),
        "lis2_zero_pct": 100.0 * float(np.mean(a2_zero)),
        "gyro2_zero_pct": 100.0 * float(np.mean(g2_zero)),
        "lower_imu_zero_runs": count_true_runs(a2_zero),
        "lower_imu_longest_zero_s": max_true_run(a2_zero, dt),
        "lower_accel_gyro_zero_masks_equal": bool(np.array_equal(a2_zero, g2_zero)),
        "secondary_mag_zero_pct": 100.0 * float(np.mean(lis_mag_zero)),
    }


def corrected_slayer_travel(
    df: pd.DataFrame, *, interpolate_rails: bool = False
) -> tuple[np.ndarray, float]:
    """Apply the current geometry after wrapping encoder counts around zero."""
    time_s = df["t_s"].to_numpy(float)
    fs_hz = 1.0 / float(np.median(np.diff(time_s)))
    raw = df["angle_raw"].to_numpy(float)
    signed_counts = ((raw + 2048.0) % 4096.0) - 2048.0
    if interpolate_rails:
        rail_bad = binary_dilation(np.isin(raw, (0, 4095)), structure=np.ones(7, dtype=bool))
        signed_counts = np.interp(time_s, time_s[~rail_bad], signed_counts[~rail_bad])
    angle_rad = signed_counts * 2.0 * np.pi / 4096.0

    # Match AngleLoader(lag=-1) followed by the pipeline's 20 Hz, order-4 LPF
    # and 200 -> 100 Hz decimation.
    angle_rad = np.roll(angle_rad, 1)
    sos = butter(N=4, Wn=20.0, btype="low", fs=fs_hz, output="sos")
    angle_lpf = sosfiltfilt(sos, angle_rad, axis=0)[::2]

    angle_signed = -angle_lpf
    top_zeroangle = float(np.percentile(angle_signed, 99.5))
    hypotenuse = 150.0
    top_adjacent = 138.5
    top_angle = np.arccos(top_adjacent / hypotenuse)
    net_angle = -(angle_signed - top_zeroangle) + top_angle
    travel = 2.0 * (top_adjacent - hypotenuse * np.cos(net_angle))
    return travel, top_zeroangle


def active_mask_for_travel(travel: np.ndarray) -> np.ndarray:
    """Reproduce FindBoringRegions for an alternate ground-truth series."""
    travel = np.asarray(travel, dtype=float).reshape(-1)
    active = np.ones(len(travel), dtype=bool)
    chunks: list[tuple[int, int]] = []
    chunk_start = 0
    chunk_min = np.inf
    chunk_max = -np.inf
    chunk_has_finite = False
    for idx, value in enumerate(travel):
        if np.isfinite(value):
            chunk_min = min(chunk_min, value)
            chunk_max = max(chunk_max, value)
            chunk_has_finite = True
        if idx <= chunk_start or not chunk_has_finite:
            continue
        if (chunk_max - chunk_min) > 10.0 or chunk_max > 200.0:
            chunk_end = idx + 1
            if chunk_end - chunk_start >= 100:
                chunks.append((max(0, chunk_start + 10), min(len(travel), chunk_end - 10)))
            chunk_start = chunk_end
            chunk_min = np.inf
            chunk_max = -np.inf
            chunk_has_finite = False
    chunk_end = len(travel)
    if (
        chunk_has_finite
        and (chunk_max - chunk_min) <= 10.0
        and chunk_max <= 200.0
        and chunk_end - chunk_start >= 100
    ):
        chunks.append((max(0, chunk_start + 10), min(len(travel), chunk_end - 10)))
    for start, stop in chunks:
        active[start:stop] = False
    return active


def rmse(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, centered: bool) -> float:
    pred = np.asarray(pred, float).reshape(-1)
    truth = np.asarray(truth, float).reshape(-1)
    mask = np.asarray(mask, bool).reshape(-1) & np.isfinite(pred) & np.isfinite(truth)
    pred = pred[mask]
    truth = truth[mask]
    if centered:
        pred = pred - np.mean(pred)
        truth = truth - np.mean(truth)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def cache_series(cache: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    return np.asarray(cache[f"{key}__x"], float).reshape(-1)


def project_bad_angle_mask(cache: np.lib.npyio.NpzFile, target_key: str = "travel") -> np.ndarray:
    target_t = np.asarray(cache[f"{target_key}__t"], float).reshape(-1)
    if "angle/bad_mask__x" not in cache:
        return np.zeros(len(target_t), dtype=bool)
    source_t = np.asarray(cache["angle/bad_mask__t"], float).reshape(-1)
    source_bad = np.asarray(cache["angle/bad_mask__x"], bool).reshape(-1)
    bad_idx = np.flatnonzero(source_bad)
    projected = np.zeros(len(target_t), dtype=bool)
    if len(bad_idx) == 0:
        return projected
    cuts = np.flatnonzero(np.diff(bad_idx) > 1)
    starts = np.r_[bad_idx[0], bad_idx[cuts + 1]]
    ends = np.r_[bad_idx[cuts], bad_idx[-1]]
    for start, stop in zip(starts, ends):
        left = np.searchsorted(target_t, source_t[start] - 0.08, side="left")
        right = np.searchsorted(target_t, source_t[stop] + 0.08, side="right")
        projected[left:right] = True
    return projected


def current_metric_rows(log_id: str, group: str) -> list[dict[str, object]]:
    cache_path = CACHE_ROOT / log_id / "cache" / "all.npz"
    if not cache_path.is_file():
        return []
    rows: list[dict[str, object]] = []
    with np.load(cache_path) as cache:
        truth = cache_series(cache, "travel")
        mask = np.asarray(cache["active_mask"], bool).reshape(-1)
        if "angle/bad_mask__x" in cache:
            mask &= ~project_bad_angle_mask(cache)
        for method, key in PREDICTIONS.items():
            if f"{key}__x" not in cache:
                continue
            pred = cache_series(cache, key)
            for centered in (False, True):
                rows.append(
                    {
                        "log": log_id,
                        "cohort": group,
                        "ground_truth": "current_pipeline",
                        "method": method,
                        "centered": centered,
                        "rmse_mm": rmse(pred, truth, mask, centered),
                        "sample_count": int(np.sum(mask & np.isfinite(pred) & np.isfinite(truth))),
                    }
                )
    return rows


def corrected_slayer_metric_rows(
    log_id: str, entry: dict, audit: dict[str, object]
) -> list[dict[str, object]]:
    df = pd.read_csv(csv_path(entry))
    corrected_truth, top_zeroangle = corrected_slayer_travel(df)
    rail_interpolated_truth, _ = corrected_slayer_travel(df, interpolate_rails=True)
    cache_path = CACHE_ROOT / log_id / "cache" / "all.npz"
    rows: list[dict[str, object]] = []
    with np.load(cache_path) as cache:
        assert len(corrected_truth) == len(cache_series(cache, "travel"))
        lower_zero_raw = np.all(
            df[["lis2_x", "lis2_y", "lis2_z"]].to_numpy(float) == 0, axis=1
        )
        # Give the 20 Hz filters 0.20 s on each side of a dropout transition.
        lower_bad_raw = binary_dilation(lower_zero_raw, structure=np.ones(81, dtype=bool))
        lower_healthy = ~lower_bad_raw[::2]
        variants = (
            ("encoder_zero_unwrapped", corrected_truth),
            ("encoder_zero_unwrapped_rails_interpolated", rail_interpolated_truth),
        )
        for ground_truth, truth in variants:
            mask = active_mask_for_travel(truth)
            for method, key in PREDICTIONS.items():
                pred = cache_series(cache, key)
                for centered in (False, True):
                    for suffix, valid in (("", mask), ("_lower_imu_healthy", mask & lower_healthy)):
                        rows.append(
                            {
                                "log": log_id,
                                "cohort": "slayer",
                                "ground_truth": ground_truth + suffix,
                                "method": method,
                                "centered": centered,
                                "rmse_mm": rmse(pred, truth, valid, centered),
                                "sample_count": int(
                                    np.sum(valid & np.isfinite(pred) & np.isfinite(truth))
                                ),
                            }
                        )
        mask = active_mask_for_travel(corrected_truth)
        audit["corrected_top_zeroangle_rad"] = top_zeroangle
        audit["current_travel_gt_over_200_pct"] = 100.0 * float(
            np.mean(cache_series(cache, "travel") > 200.0)
        )
        audit["corrected_travel_gt_over_200_pct"] = 100.0 * float(
            np.mean(corrected_truth > 200.0)
        )
        audit["current_travel_gt_p99_mm"] = float(
            np.percentile(cache_series(cache, "travel"), 99)
        )
        audit["corrected_travel_gt_p99_mm"] = float(np.percentile(corrected_truth, 99))
        audit["corrected_active_pct"] = 100.0 * float(np.mean(mask))
    return rows


def cohort_for(entry: dict) -> str:
    sets = set(entry.get("sets", []))
    if "stumpjumper-front-pod-v1" in sets:
        return "stumpjumper-pod-v1"
    if "stumpjumper-front-pod-v2" in sets:
        return "stumpjumper-pod-v2"
    if "harry" in sets:
        return "harry"
    if "jamaal" in sets:
        return "jamaal"
    return "other"


def nuisance_rows(log_ids: list[str], audits: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for log_id in log_ids:
        cache_path = CACHE_ROOT / log_id / "cache" / "all.npz"
        if not cache_path.is_file():
            continue
        with np.load(cache_path) as cache:
            if "mag/nuisance/summary" not in cache:
                continue
            summary = np.asarray(cache["mag/nuisance/summary"], float).reshape(-1)
            xyz_path = np.asarray(cache["mag/nuisance/xyz_path"], float)
            travel_grid = xyz_path[:, 0]
            slope = np.linalg.norm(np.gradient(xyz_path[:, 1:], travel_grid, axis=0), axis=1)
            low = (travel_grid >= 0.0) & (travel_grid <= 30.0)
            base = cache_series(cache, "travel/fusion1")
            lifted = cache_series(cache, "travel/solved/mag_nuisance/delta_lifted")
            delta = lifted - base
            row = {
                "log": log_id,
                "xyz_bin_count": summary[2],
                "update_fraction": summary[3],
                "correction_vector_rms_mg": summary[4],
                "applied_update_rms_mm": summary[6],
                "final_iteration_change_mm": summary[7],
                "low_travel_xyz_slope_mg_per_mm": float(np.median(slope[low])),
                "delta_lifted_signal_rms_mm": float(np.sqrt(np.mean(delta**2))),
                "mag_model_offset_mm": float(np.asarray(cache["mag_model_offset_mm"]).reshape(-1)[0]),
                "mag_reference_x_mm": float(np.asarray(cache["mag_travel_ref_point"]).reshape(-1)[0]),
                "mag_reference_mg": float(np.asarray(cache["mag_travel_ref_point"]).reshape(-1)[1]),
            }
            ref_diag = reference_diagnostic(cache)
            row.update(ref_diag)
            if log_id in SUCCESSFUL_SLAYER:
                frame = pd.read_csv(csv_path(REGISTRY["logs"][log_id]))
                corrected_truth, _ = corrected_slayer_travel(frame)
                row.update(reference_diagnostic(cache, corrected_truth, prefix="corrected_"))
            if log_id in audits:
                row["lis2_zero_pct"] = audits[log_id]["lis2_zero_pct"]
            rows.append(row)
    return rows


def reference_diagnostic(
    cache: np.lib.npyio.NpzFile,
    alternate_truth: np.ndarray | None = None,
    *,
    prefix: str = "current_",
) -> dict[str, object]:
    """Recreate the absolute-reference candidate and selected-point statistics."""
    accel = cache_series(cache, "accel/lpfhp/proj") * 1000.0
    mag = cache_series(cache, "mag/norm/corr/lpf")
    truth = cache_series(cache, "travel") if alternate_truth is None else np.asarray(alternate_truth, float)
    time_s = np.asarray(cache["mag/norm/corr/lpf__t"], float).reshape(-1)
    dt_s = np.diff(time_s, prepend=time_s[0] - 0.01)
    baseline = float(np.asarray(cache["mag_baseline"]).reshape(-1)[0])
    still_len, bump_len, stride = 10, 30, 5
    chunk_len = still_len + bump_len
    x_chunks: list[np.ndarray] = []
    mag_chunks: list[np.ndarray] = []
    truth_chunks: list[np.ndarray] = []
    skip = 0
    for idx in range(0, len(mag) - chunk_len, stride):
        if skip > 0:
            skip -= 1
            continue
        for chunk_slice in (slice(idx, idx + chunk_len), slice(idx + chunk_len, idx, -1)):
            chunk_mag = mag[chunk_slice]
            chunk_accel = accel[chunk_slice]
            mag_still = chunk_mag[:still_len]
            accel_still = chunk_accel[:still_len]
            accel_bump = chunk_accel[still_len : still_len + bump_len]
            dt_bump = dt_s[chunk_slice][still_len : still_len + bump_len]
            mag_bump = chunk_mag[still_len : still_len + bump_len]
            if np.mean(mag_still) > baseline:
                continue
            if np.max(np.abs(accel_still)) > 1000.0:
                continue
            if np.max(mag_bump) < np.mean(mag_still) + 1000.0:
                continue
            velocity = np.cumsum(accel_bump * dt_bump)
            displacement = np.cumsum(velocity * dt_bump)
            if np.max(displacement) < 20.0:
                continue
            skip = 3
            x_chunks.append(displacement)
            mag_chunks.append(mag_bump)
            truth_chunks.append(truth[chunk_slice][still_len : still_len + bump_len])
    result: dict[str, object] = {
        f"{prefix}reference_chunks": len(x_chunks),
        f"{prefix}reference_selected_points": 0,
        f"{prefix}reference_error_mm": float("nan"),
    }
    if not x_chunks:
        return result
    x_points = np.concatenate(x_chunks)
    mag_points = np.concatenate(mag_chunks)
    truth_points = np.concatenate(truth_chunks)
    center = max(baseline + 2000.0, float(np.median(mag_points)))
    selected = (mag_points > center - 1000.0) & (mag_points < center + 1000.0)
    result[f"{prefix}reference_selected_points"] = int(np.sum(selected))
    if np.any(selected):
        result[f"{prefix}reference_error_mm"] = float(
            np.median(x_points[selected]) - np.median(truth_points[selected])
        )
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_cohorts(metrics: pd.DataFrame, centered: bool) -> pd.DataFrame:
    current = metrics[
        (metrics["ground_truth"] == "current_pipeline")
        & (metrics["centered"] == centered)
        & metrics["method"].isin(["fusion1_baseline", "nuisance_delta_lifted", "nuisance_fusion2"])
    ]
    pivot = current.pivot_table(index=["cohort", "log"], columns="method", values="rmse_mm").reset_index()
    pivot["delta_lift_change_mm"] = pivot["nuisance_delta_lifted"] - pivot["fusion1_baseline"]
    pivot["fusion2_change_mm"] = pivot["nuisance_fusion2"] - pivot["fusion1_baseline"]
    grouped = pivot.groupby("cohort", sort=False)
    out = grouped.agg(
        logs=("log", "count"),
        baseline_mean_rmse_mm=("fusion1_baseline", "mean"),
        baseline_median_rmse_mm=("fusion1_baseline", "median"),
        delta_lift_mean_rmse_mm=("nuisance_delta_lifted", "mean"),
        delta_lift_mean_change_mm=("delta_lift_change_mm", "mean"),
        fusion2_mean_rmse_mm=("nuisance_fusion2", "mean"),
        fusion2_mean_change_mm=("fusion2_change_mm", "mean"),
        delta_lift_regressions=("delta_lift_change_mm", lambda x: int(np.sum(x > 0))),
        fusion2_regressions=("fusion2_change_mm", lambda x: int(np.sum(x > 0))),
    ).reset_index()
    out.insert(1, "centered", centered)
    return out


def corrected_slayer_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics[
        (metrics["cohort"] == "slayer")
        & (metrics["ground_truth"].str.startswith("encoder_zero_unwrapped"))
        & metrics["method"].isin(["fusion1_baseline", "nuisance_delta_lifted", "nuisance_fusion2"])
    ]
    pivot = selected.pivot_table(
        index=["ground_truth", "centered", "log"], columns="method", values="rmse_mm"
    ).reset_index()
    pivot["delta_lift_change_mm"] = pivot["nuisance_delta_lifted"] - pivot["fusion1_baseline"]
    pivot["fusion2_change_mm"] = pivot["nuisance_fusion2"] - pivot["fusion1_baseline"]
    return (
        pivot.groupby(["ground_truth", "centered"])
        .agg(
            logs=("log", "count"),
            baseline_mean_rmse_mm=("fusion1_baseline", "mean"),
            baseline_median_rmse_mm=("fusion1_baseline", "median"),
            delta_lift_mean_rmse_mm=("nuisance_delta_lifted", "mean"),
            delta_lift_mean_change_mm=("delta_lift_change_mm", "mean"),
            fusion2_mean_rmse_mm=("nuisance_fusion2", "mean"),
            fusion2_mean_change_mm=("fusion2_change_mm", "mean"),
            delta_lift_regressions=("delta_lift_change_mm", lambda x: int(np.sum(x > 0))),
            fusion2_regressions=("fusion2_change_mm", lambda x: int(np.sum(x > 0))),
        )
        .reset_index()
    )


def make_plots(audit_df: pd.DataFrame, metrics: pd.DataFrame, cohort_summary: pd.DataFrame) -> None:
    slayer = audit_df.set_index("log").loc[SLAYER_LOGS]
    y = np.arange(len(slayer))
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(y - 0.22, slayer["lis2_zero_pct"], height=0.22, label="lower accelerometer zero")
    ax.barh(y, slayer["gyro2_zero_pct"], height=0.22, label="lower gyro zero")
    ax.barh(y + 0.22, slayer["angle_wrap_side_pct"], height=0.22, label="encoder samples >2048")
    ax.set_yticks(y, slayer.index)
    ax.invert_yaxis()
    ax.set_xlabel("Percent of raw samples")
    ax.set_title("Slayer sensor failure and encoder-wrap indicators")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "sensor_health.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 7))
    for idx, log_id in enumerate(SLAYER_LOGS):
        entry = REGISTRY["logs"][log_id]
        df = pd.read_csv(csv_path(entry), usecols=["t_s", "lis2_x", "lis2_y", "lis2_z"])
        zero = np.all(df[["lis2_x", "lis2_y", "lis2_z"]].to_numpy() == 0, axis=1)
        t = df["t_s"].to_numpy(float)
        ax.fill_between(t - t[0], idx - 0.35, idx + 0.35, where=zero, step="mid", color="#c83e4d")
    ax.set_yticks(range(len(SLAYER_LOGS)), SLAYER_LOGS)
    ax.invert_yaxis()
    ax.set_xlabel("Time since start (s)")
    ax.set_title("Lower-IMU zero-output intervals")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "lower_imu_dropout_timeline.png", dpi=180)
    plt.close(fig)

    example = "log-0150"
    df = pd.read_csv(csv_path(REGISTRY["logs"][example]))
    corrected, _ = corrected_slayer_travel(df)
    with np.load(CACHE_ROOT / example / "cache" / "all.npz") as cache:
        t = np.asarray(cache["travel__t"], float).reshape(-1)
        current = cache_series(cache, "travel")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t - t[0], current, lw=0.8, label="current pipeline ground truth", alpha=0.8)
    ax.plot(t - t[0], corrected, lw=0.9, label="zero-wrapped encoder ground truth")
    ax.axhline(200, color="black", lw=0.8, ls="--", label="nominal travel")
    ax.set_ylim(-35, 610)
    ax.set_xlabel("Time since start (s)")
    ax.set_ylabel("Travel (mm)")
    ax.set_title(f"Encoder wrap artifact in {example}")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "angle_wrap_example.png", dpi=180)
    plt.close(fig)

    chosen = metrics[
        (metrics["cohort"] == "slayer")
        & (metrics["ground_truth"] == "encoder_zero_unwrapped")
        & (metrics["centered"] == True)  # noqa: E712
        & metrics["method"].isin(["fusion1_baseline", "nuisance_delta_lifted", "nuisance_fusion2"])
    ]
    pivot = chosen.pivot(index="log", columns="method", values="rmse_mm").loc[SUCCESSFUL_SLAYER]
    x = np.arange(len(pivot))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for offset, (method, label) in zip(
        (-width, 0, width),
        [
            ("fusion1_baseline", "baseline fusion"),
            ("nuisance_delta_lifted", "nuisance delta-lift"),
            ("nuisance_fusion2", "nuisance refusion"),
        ],
    ):
        ax.bar(x + offset, pivot[method], width, label=label)
    ax.set_xticks(x, pivot.index, rotation=30)
    ax.set_ylabel("Centered RMSE (mm)")
    ax.set_title("Slayer RMSE after correcting the encoder wrap")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "slayer_corrected_rmse.png", dpi=180)
    plt.close(fig)

    cohorts = cohort_summary[cohort_summary["centered"] == True].set_index("cohort")  # noqa: E712
    order = [name for name in ["stumpjumper-pod-v1", "stumpjumper-pod-v2", "jamaal", "harry", "slayer"] if name in cohorts.index]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(order))
    ax.bar(x - 0.18, cohorts.loc[order, "baseline_mean_rmse_mm"], 0.36, label="baseline fusion")
    ax.bar(x + 0.18, cohorts.loc[order, "fusion2_mean_rmse_mm"], 0.36, label="nuisance refusion")
    ax.set_xticks(x, order, rotation=20, ha="right")
    ax.set_ylabel("Mean centered RMSE (mm)")
    ax.set_title("Current-pipeline RMSE by front-default cohort")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cohort_rmse.png", dpi=180)
    plt.close(fig)


with REGISTRY_PATH.open("rb") as handle:
    REGISTRY = tomllib.load(handle)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audits = {
        log_id: sensor_audit(log_id, REGISTRY["logs"][log_id])
        for log_id in SLAYER_LOGS
    }

    metric_rows: list[dict[str, object]] = []
    front_default_logs: list[str] = []
    for log_id, entry in REGISTRY["logs"].items():
        if "front-default" not in entry.get("sets", []):
            continue
        front_default_logs.append(log_id)
        metric_rows.extend(current_metric_rows(log_id, cohort_for(entry)))
    for log_id in SUCCESSFUL_SLAYER:
        metric_rows.extend(current_metric_rows(log_id, "slayer"))
        metric_rows.extend(corrected_slayer_metric_rows(log_id, REGISTRY["logs"][log_id], audits[log_id]))

    audit_rows = [audits[log_id] for log_id in SLAYER_LOGS]
    write_csv(OUT / "sensor_audit.csv", audit_rows)
    write_csv(OUT / "rmse_per_log.csv", metric_rows)

    metrics = pd.DataFrame(metric_rows)
    cohort_summary = pd.concat(
        [aggregate_cohorts(metrics, False), aggregate_cohorts(metrics, True)],
        ignore_index=True,
    )
    cohort_summary.to_csv(OUT / "cohort_summary_current_ground_truth.csv", index=False)
    corrected_summary = corrected_slayer_summary(metrics)
    corrected_summary.to_csv(OUT / "slayer_corrected_ground_truth_summary.csv", index=False)

    all_nuisance_ids = front_default_logs + SUCCESSFUL_SLAYER
    nuisance = nuisance_rows(all_nuisance_ids, audits)
    write_csv(OUT / "nuisance_diagnostics.csv", nuisance)
    make_plots(pd.DataFrame(audit_rows), metrics, cohort_summary)


if __name__ == "__main__":
    main()
