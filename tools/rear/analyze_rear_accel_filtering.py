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

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import numpy as np
from scipy.signal import butter, sosfiltfilt

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from accel_zv import correct_accel_with_zv, filter_mag_zv_points  # noqa: E402
from mag_to_travel_model_core import MagToTravelChunk, MagToTravelModel  # noqa: E402
from rear_mag_model import RearMagModel  # noqa: E402
from travel_solver_core import SolverInputs, SolverWeights, solve_travel  # noqa: E402


TRAVEL_BIN_EDGES = np.linspace(0.0, 150.0, 6)
MIN_BIN_SAMPLES = 100
ANGLE_ERROR_HALO_S = 0.08


@dataclass(frozen=True)
class FilterVariant:
    name: str
    highpass_fc_hz: float
    highpass_order: int = 2
    lowpass_fc_hz: float | None = 40.0
    lowpass_order: int = 4
    zv_smooth_bias_s: float = 0.05
    zv_min_prominence_mg: float = 0.0
    zv_min_separation_s: float = 0.0
    use_zv_correction: bool = True


@dataclass
class LogData:
    name: str
    t: np.ndarray
    raw_accel: np.ndarray
    lpf_accel: np.ndarray
    cached_axis: np.ndarray
    mag: np.ndarray
    travel: np.ndarray
    valid_mask: np.ndarray
    zv_points: np.ndarray
    travel_accel: np.ndarray


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


def load_log(log_name: str) -> LogData:
    ws = np.load(REPO_ROOT / "backend" / "run_artifacts" / log_name / "cache" / "all.npz")
    t = flatten_1d(ws["accel/lpf/lis2__t"])
    raw_accel = np.asarray(ws["accel/lis2__x"], dtype=float)
    lpf_accel = np.asarray(ws["accel/lpf/lis2__x"], dtype=float)
    travel = flatten_1d(ws["travel__x"])
    velocity = np.gradient(travel, t, edge_order=2)
    travel_accel = np.gradient(velocity, t, edge_order=2) / 1000.0

    angle_bad = np.zeros(len(t), dtype=bool)
    if "angle/bad_mask__x" in ws and "angle/bad_mask__t" in ws:
        angle_bad = project_bad_angle_mask(ws["angle/bad_mask__t"], ws["angle/bad_mask__x"], t)

    finite = (
        np.isfinite(travel)
        & np.all(np.isfinite(raw_accel), axis=1)
        & np.isfinite(flatten_1d(ws["mag/proj/lpf__x"]))
    )
    return LogData(
        name=log_name,
        t=t,
        raw_accel=raw_accel,
        lpf_accel=lpf_accel,
        cached_axis=np.asarray(ws["accel_trav_vec"], dtype=float).reshape(-1),
        mag=flatten_1d(ws["mag/proj/lpf__x"]),
        travel=travel,
        valid_mask=np.asarray(ws["boring_mask"]).astype(bool).reshape(-1) & finite & ~angle_bad,
        zv_points=np.asarray(ws["mag_zv_points"], dtype=int),
        travel_accel=travel_accel,
    )


def infer_fs_hz(t: np.ndarray) -> float:
    dt = np.diff(flatten_1d(t))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError("Cannot infer sampling frequency from empty/non-monotonic time vector")
    return 1.0 / float(np.median(dt))


def apply_filter(x: np.ndarray, fs_hz: float, fc_hz: float, btype: str, order: int) -> np.ndarray:
    sos = butter(N=order, Wn=fc_hz, btype=btype, fs=fs_hz, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def filtered_accel(log: LogData, variant: FilterVariant) -> np.ndarray:
    fs_hz = infer_fs_hz(log.t)
    if variant.lowpass_fc_hz is None:
        lowpassed = log.lpf_accel
    else:
        lowpassed = apply_filter(
            log.raw_accel,
            fs_hz,
            variant.lowpass_fc_hz,
            "low",
            variant.lowpass_order,
        )
    return apply_filter(
        lowpassed,
        fs_hz,
        variant.highpass_fc_hz,
        "high",
        variant.highpass_order,
    )


def rear_accel_axis(a_hp: np.ndarray, fs_hz: float) -> np.ndarray:
    a_norm = np.linalg.norm(a_hp, axis=1)
    mask = a_norm > 10.0
    dilation = max(1, int(round(fs_hz)))
    mask = np.convolve(mask.astype(float), np.ones(dilation), mode="same") > 0
    samples = a_hp[mask]
    if len(samples) == 0:
        raise ValueError("No high-accel samples available for rear accel axis")

    directed = samples[samples[:, 1] < 0]
    if len(directed) < 10:
        directed = samples

    axis = np.mean(directed, axis=0)
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Rear accel axis is degenerate")
    return axis / norm


def prepare_projected_accel(log: LogData, variant: FilterVariant) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    a_hp = filtered_accel(log, variant)
    axis = rear_accel_axis(a_hp, infer_fs_hz(log.t))
    projected = a_hp @ axis

    idxs = filter_mag_zv_points(
        log.zv_points,
        log.mag,
        log.t,
        min_prominence_mg=variant.zv_min_prominence_mg,
        min_separation_s=variant.zv_min_separation_s,
    )
    if not variant.use_zv_correction:
        return projected, idxs, {
            "zv_count": float(len(idxs)),
            "bias_std": 0.0,
            "bias_abs_p95": 0.0,
            "axis_dot_cached": float(np.dot(axis, log.cached_axis)),
        }

    corrected, stats = correct_accel_with_zv(
        projected,
        log.t,
        idxs,
        mode="smoothed_bias",
        smooth_bias_s=variant.zv_smooth_bias_s,
    )
    stats["axis_dot_cached"] = float(np.dot(axis, log.cached_axis))
    return corrected, idxs, stats


def active_accel_metrics(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(pred) & np.isfinite(gt) & valid_mask & (np.abs(gt) >= 0.5)
    if not np.any(mask):
        return {"accel_n": 0, "accel_rmse": float("nan"), "accel_mae": float("nan"), "accel_corr": float("nan")}
    err = pred[mask] - gt[mask]
    corr = float("nan")
    if np.std(pred[mask]) > 1e-12 and np.std(gt[mask]) > 1e-12:
        corr = float(np.corrcoef(pred[mask], gt[mask])[0, 1])
    return {
        "accel_n": int(np.sum(mask)),
        "accel_rmse": float(np.sqrt(np.mean(err**2))),
        "accel_mae": float(np.mean(np.abs(err))),
        "accel_corr": corr,
    }


def centered_error(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pred_roi = np.asarray(pred, dtype=float)[mask]
    gt_roi = np.asarray(gt, dtype=float)[mask]
    return (pred_roi - np.mean(pred_roi)) - (gt_roi - np.mean(gt_roi))


def bin_stats(err: np.ndarray, gt_roi: np.ndarray) -> tuple[float, float, list[float], list[int]]:
    rmses: list[float] = []
    counts: list[int] = []
    eligible_mses: list[float] = []
    for idx, (lo, hi) in enumerate(zip(TRAVEL_BIN_EDGES[:-1], TRAVEL_BIN_EDGES[1:])):
        mask = (gt_roi >= lo) & ((gt_roi <= hi) if idx == len(TRAVEL_BIN_EDGES) - 2 else (gt_roi < hi))
        count = int(np.sum(mask))
        counts.append(count)
        if count:
            rmse = float(np.sqrt(np.mean(err[mask] ** 2)))
            rmses.append(rmse)
            if count >= MIN_BIN_SAMPLES:
                eligible_mses.append(float(np.mean(err[mask] ** 2)))
        else:
            rmses.append(float("nan"))
    bin_rmse = float(np.sqrt(np.mean(eligible_mses))) if eligible_mses else float("nan")
    worst_bin_rmse = float(np.nanmax(np.asarray(rmses, dtype=float)))
    return bin_rmse, worst_bin_rmse, rmses, counts


def travel_metrics(pred: np.ndarray, log: LogData, prefix: str) -> dict[str, float]:
    err = centered_error(pred, log.travel, log.valid_mask)
    gt_roi = log.travel[log.valid_mask]
    bin_rmse, worst_bin_rmse, bin_rmses, bin_counts = bin_stats(err, gt_roi)
    row: dict[str, float] = {
        f"{prefix}_rmse": float(np.sqrt(np.mean(err**2))),
        f"{prefix}_mae": float(np.mean(np.abs(err))),
        f"{prefix}_bin_rmse": bin_rmse,
        f"{prefix}_worst_bin_rmse": worst_bin_rmse,
    }
    for idx, value in enumerate(bin_rmses):
        row[f"{prefix}_bin{idx}_rmse"] = value
    for idx, value in enumerate(bin_counts):
        row[f"{prefix}_bin{idx}_n"] = value
    return row


def subsample_chunks(
    chunks: list[MagToTravelChunk],
    *,
    max_chunks: int | None,
    seed: int,
) -> list[MagToTravelChunk]:
    if max_chunks is None or len(chunks) <= max_chunks:
        return chunks
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(chunks), size=max_chunks, replace=False))
    return [chunks[int(i)] for i in keep]


def stable_seed(*parts: str) -> int:
    import hashlib

    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def fit_mag_model(
    log: LogData,
    accel: np.ndarray,
    zv_points: np.ndarray,
    variant: FilterVariant,
    *,
    max_chunks: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    model = RearMagModel(
        x0_weight=0.0,
        chunking_method="centered_zv",
        chunk_rad=20,
        chunk_min_dx=10.0,
        chunk_max_dx=150.0,
    )
    chunks = model.create_chunks(zv_points, log.mag, accel, log.t)
    model.prepare_chunks(chunks)
    chunks = model.filter_chunks(chunks, model.get_filter_fns())
    chunks = subsample_chunks(
        chunks,
        max_chunks=max_chunks,
        seed=stable_seed(log.name, variant.name),
    )
    model.chunks = chunks
    if not chunks:
        raise ValueError(f"No chunks for {variant.name} on {log.name}")

    with contextlib.redirect_stdout(io.StringIO()):
        input_arr = model.format_chunks_for_fit(chunks)
        result = model.train(input_arr, guess_vec=[0.0, -1.0, 1.0 / 3.0])

    fitted = MagToTravelModel(pred_soft_mg=model.pred_soft_mg)
    fitted.set_coeffs(result.x)
    pred = fitted.pred_x(log.mag)
    return pred - np.percentile(pred, 8.0), result.x.copy(), len(chunks)


def maybe_solve(
    log: LogData,
    accel: np.ndarray,
    zv_points: np.ndarray,
    mag_pred: np.ndarray,
    *,
    run_solver: bool,
    max_nfev: int,
) -> np.ndarray | None:
    if not run_solver:
        return None
    inputs = SolverInputs(
        time_s=log.t,
        accel_mm_s2=accel * 1000.0,
        mag=None,
        mag_preds_mm=mag_pred,
        mag_zv_points=zv_points,
        mag_baseline=None,
    )
    result = solve_travel(
        inputs,
        SolverWeights(mag_x=400.0),
        max_nfev=max_nfev,
        verbose=0,
    )
    return result.x


def evaluate_variant_on_log(
    variant: FilterVariant,
    log: LogData,
    *,
    max_chunks: int | None,
    run_solver: bool,
    solver_max_nfev: int,
) -> dict[str, object]:
    accel, zv_points, filter_stats = prepare_projected_accel(log, variant)
    mag_pred, coeffs, chunk_count = fit_mag_model(
        log,
        accel,
        zv_points,
        variant,
        max_chunks=max_chunks,
    )
    solver_pred = maybe_solve(
        log,
        accel,
        zv_points,
        mag_pred,
        run_solver=run_solver,
        max_nfev=solver_max_nfev,
    )

    row: dict[str, object] = {
        "variant": variant.name,
        "log": log.name,
        "lowpass_fc_hz": variant.lowpass_fc_hz if variant.lowpass_fc_hz is not None else "cached",
        "lowpass_order": variant.lowpass_order,
        "highpass_fc_hz": variant.highpass_fc_hz,
        "highpass_order": variant.highpass_order,
        "zv_smooth_bias_s": variant.zv_smooth_bias_s,
        "use_zv_correction": variant.use_zv_correction,
        "chunks": chunk_count,
        "zv_points": int(filter_stats["zv_count"]),
        "bias_std": float(filter_stats["bias_std"]),
        "bias_abs_p95": float(filter_stats["bias_abs_p95"]),
        "axis_dot_cached": float(filter_stats["axis_dot_cached"]),
        "coeff_x0": float(coeffs[0]),
        "coeff_y_scale": float(coeffs[1]),
        "coeff_power": float(coeffs[2]),
    }
    row.update(active_accel_metrics(accel, log.travel_accel, log.valid_mask))
    row.update(travel_metrics(mag_pred, log, "mag"))
    if solver_pred is not None:
        row.update(travel_metrics(solver_pred, log, "solver"))
    return row


def default_variants() -> list[FilterVariant]:
    variants = [
        FilterVariant("current_lpf40_hpf2_o2_s50", highpass_fc_hz=2.0),
        FilterVariant("hpf0p5_o2_s50", highpass_fc_hz=0.5),
        FilterVariant("hpf0p75_o2_s50", highpass_fc_hz=0.75),
        FilterVariant("hpf1_o2_s50", highpass_fc_hz=1.0),
        FilterVariant("hpf1p25_o2_s50", highpass_fc_hz=1.25),
        FilterVariant("hpf1p5_o2_s50", highpass_fc_hz=1.5),
        FilterVariant("hpf2p5_o2_s50", highpass_fc_hz=2.5),
        FilterVariant("hpf3_o2_s50", highpass_fc_hz=3.0),
        FilterVariant("hpf4_o2_s50", highpass_fc_hz=4.0),
        FilterVariant("hpf1_o1_s50", highpass_fc_hz=1.0, highpass_order=1),
        FilterVariant("hpf1_o4_s50", highpass_fc_hz=1.0, highpass_order=4),
        FilterVariant("hpf2_o1_s50", highpass_fc_hz=2.0, highpass_order=1),
        FilterVariant("hpf2_o4_s50", highpass_fc_hz=2.0, highpass_order=4),
        FilterVariant("hpf1_o2_s25", highpass_fc_hz=1.0, zv_smooth_bias_s=0.025),
        FilterVariant("hpf1_o2_s100", highpass_fc_hz=1.0, zv_smooth_bias_s=0.10),
        FilterVariant("hpf1_o2_s200", highpass_fc_hz=1.0, zv_smooth_bias_s=0.20),
        FilterVariant("lpf20_hpf1_o2_s50", lowpass_fc_hz=20.0, highpass_fc_hz=1.0),
        FilterVariant("lpf30_hpf1_o2_s50", lowpass_fc_hz=30.0, highpass_fc_hz=1.0),
        FilterVariant("lpf60_hpf1_o2_s50", lowpass_fc_hz=60.0, highpass_fc_hz=1.0),
        FilterVariant("lpf80_hpf1_o2_s50", lowpass_fc_hz=80.0, highpass_fc_hz=1.0),
        FilterVariant("hpf1_o2_no_zv_corr", highpass_fc_hz=1.0, use_zv_correction=False),
    ]
    return variants


def mean_metric(rows: list[dict[str, object]], key: str) -> float:
    vals = np.asarray([float(row[key]) for row in rows if key in row], dtype=float)
    if not np.any(np.isfinite(vals)):
        return float("nan")
    return float(np.nanmean(vals))


def aggregate(rows: list[dict[str, object]], variants: list[FilterVariant]) -> list[dict[str, object]]:
    out = []
    for variant in variants:
        subset = [row for row in rows if row["variant"] == variant.name]
        if not subset:
            continue
        row: dict[str, object] = {
            "variant": variant.name,
            "lowpass_fc_hz": variant.lowpass_fc_hz if variant.lowpass_fc_hz is not None else "cached",
            "highpass_fc_hz": variant.highpass_fc_hz,
            "highpass_order": variant.highpass_order,
            "zv_smooth_bias_s": variant.zv_smooth_bias_s,
            "use_zv_correction": variant.use_zv_correction,
            "mean_accel_rmse": mean_metric(subset, "accel_rmse"),
            "mean_accel_corr": mean_metric(subset, "accel_corr"),
            "mean_mag_rmse": mean_metric(subset, "mag_rmse"),
            "mean_mag_bin_rmse": mean_metric(subset, "mag_bin_rmse"),
            "mean_mag_worst_bin_rmse": mean_metric(subset, "mag_worst_bin_rmse"),
            "mean_solver_rmse": mean_metric(subset, "solver_rmse"),
            "mean_solver_bin_rmse": mean_metric(subset, "solver_bin_rmse"),
            "mean_solver_worst_bin_rmse": mean_metric(subset, "solver_worst_bin_rmse"),
            "mean_chunks": mean_metric(subset, "chunks"),
            "mean_zv_points": mean_metric(subset, "zv_points"),
            "mean_bias_abs_p95": mean_metric(subset, "bias_abs_p95"),
            "mean_axis_dot_cached": mean_metric(subset, "axis_dot_cached"),
        }
        score_base = row["mean_solver_bin_rmse"] if np.isfinite(float(row["mean_solver_bin_rmse"])) else row["mean_mag_bin_rmse"]
        row["score"] = float(score_base) + 0.03 * float(row["mean_accel_rmse"])
        out.append(row)
    return sorted(out, key=lambda row: float(row["score"]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 3) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_f):
        return "nan"
    return f"{value_f:.{digits}f}"


def write_report(
    path: Path,
    logs: list[str],
    rows: list[dict[str, object]],
    aggregate_rows: list[dict[str, object]],
    *,
    max_chunks: int | None,
    run_solver: bool,
) -> None:
    best = aggregate_rows[0]
    current = next((row for row in aggregate_rows if row["variant"] == "current_lpf40_hpf2_o2_s50"), None)
    lines = [
        "# Rear Accel Filtering Sweep",
        "",
        "Logs:",
        "",
    ]
    lines.extend(f"- `{log}`" for log in logs)
    lines.extend(
        [
            "",
            "## Method",
            "",
            (
                "Each variant recomputes the rear LIS2 lowpass/highpass, re-estimates the rear travel axis "
                "from high-accel samples, projects to 1D, applies the current smoothed-bias mag-ZV correction, "
                "then retrains the rear mag model with `centered_zv` chunks and `x0_weight=0.0`."
            ),
            "",
            f"- Chunk cap during this run: `{max_chunks if max_chunks is not None else 'none'}`.",
            f"- Solver evaluated: `{run_solver}`.",
            "",
            "## Main Findings",
            "",
            (
                f"- Best score in this run: `{best['variant']}` with mean mag bin RMSE "
                f"`{fmt(best['mean_mag_bin_rmse'])} mm` and mean accel RMSE "
                f"`{fmt(best['mean_accel_rmse'])} m/s^2`."
            ),
        ]
    )
    if current is not None:
        lines.append(
            f"- Current 2 Hz HPF reference: mean mag bin RMSE `{fmt(current['mean_mag_bin_rmse'])} mm`, "
            f"mean accel RMSE `{fmt(current['mean_accel_rmse'])} m/s^2`."
        )
    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
            "| Variant | LPF | HPF | Order | ZV smooth | Accel RMSE | Accel Corr | Mag RMSE | Mag Bin | Mag Worst | Solver RMSE | Solver Bin | Axis Dot | Chunks | Bias p95 | Score |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            f"| `{row['variant']}` | {row['lowpass_fc_hz']} | {fmt(row['highpass_fc_hz'], 2)} | "
            f"{int(row['highpass_order'])} | {fmt(1000 * float(row['zv_smooth_bias_s']), 0)} ms | "
            f"{fmt(row['mean_accel_rmse'])} | {fmt(row['mean_accel_corr'])} | "
            f"{fmt(row['mean_mag_rmse'])} | {fmt(row['mean_mag_bin_rmse'])} | "
            f"{fmt(row['mean_mag_worst_bin_rmse'])} | {fmt(row['mean_solver_rmse'])} | "
            f"{fmt(row['mean_solver_bin_rmse'])} | {fmt(row['mean_axis_dot_cached'])} | "
            f"{fmt(row['mean_chunks'], 0)} | {fmt(row['mean_bias_abs_p95'])} | {fmt(row['score'])} |"
        )

    selected_names = [str(best["variant"]), "current_lpf40_hpf2_o2_s50", "hpf1_o2_s50"]
    available = {str(row["variant"]) for row in rows}
    selected_names = [name for name in dict.fromkeys(selected_names) if name in available]
    lines.extend(
        [
            "",
            "## Selected Per-Log Metrics",
            "",
            "| Log | Variant | Accel RMSE | Accel Corr | Mag RMSE | Mag Bin | Mag Worst | Solver RMSE | Solver Bin | Axis Dot | Chunks |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for log_name in logs:
        for name in selected_names:
            match = next(row for row in rows if row["log"] == log_name and row["variant"] == name)
            lines.append(
                f"| `{log_name}` | `{name}` | {fmt(match['accel_rmse'])} | {fmt(match['accel_corr'])} | "
                f"{fmt(match['mag_rmse'])} | {fmt(match['mag_bin_rmse'])} | "
                f"{fmt(match['mag_worst_bin_rmse'])} | {fmt(match.get('solver_rmse', float('nan')))} | "
                f"{fmt(match.get('solver_bin_rmse', float('nan')))} | {fmt(match['axis_dot_cached'])} | "
                f"{int(match['chunks'])} |"
            )

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rear accel filtering variants.")
    parser.add_argument("--logs", nargs="*", default=[f"log{i}_rear" for i in range(148, 155)])
    parser.add_argument("--out-dir", type=Path, default=Path("reports/rear_accel_filtering_148_154"))
    parser.add_argument("--max-chunks", type=int, default=3000)
    parser.add_argument("--run-solver", action="store_true")
    parser.add_argument("--solver-max-nfev", type=int, default=40)
    parser.add_argument("--variants", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = default_variants()
    if args.variants:
        names = set(args.variants)
        variants = [variant for variant in variants if variant.name in names]
        missing = names - {variant.name for variant in variants}
        if missing:
            raise ValueError(f"Unknown variants: {sorted(missing)}")

    logs = [load_log(log_name) for log_name in args.logs]
    rows: list[dict[str, object]] = []
    for variant in variants:
        print(f"Evaluating {variant.name}", flush=True)
        for log in logs:
            row = evaluate_variant_on_log(
                variant,
                log,
                max_chunks=args.max_chunks,
                run_solver=args.run_solver,
                solver_max_nfev=args.solver_max_nfev,
            )
            rows.append(row)
            solver_txt = ""
            if args.run_solver:
                solver_txt = f" solver_bin={float(row['solver_bin_rmse']):.3f}"
            print(
                f"  {log.name}: accel={float(row['accel_rmse']):.3f} "
                f"mag_bin={float(row['mag_bin_rmse']):.3f}{solver_txt} "
                f"axis_dot={float(row['axis_dot_cached']):.3f} "
                f"chunks={int(row['chunks'])}",
                flush=True,
            )

    agg = aggregate(rows, variants)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_log.csv", rows)
    write_csv(args.out_dir / "aggregate.csv", agg)
    write_report(
        args.out_dir / "report.md",
        args.logs,
        rows,
        agg,
        max_chunks=args.max_chunks,
        run_solver=args.run_solver,
    )
    print(f"Wrote {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
