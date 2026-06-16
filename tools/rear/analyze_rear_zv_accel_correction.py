#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import numpy as np
import scipy.optimize

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


@dataclass(frozen=True)
class Variant:
    name: str
    correction_mode: str | None
    min_prominence_mg: float = 0.0
    min_separation_s: float = 0.0
    smooth_bias_s: float = 0.05
    chunking_method: str = "centered_zv"
    power_prior: float = 1.0 / 3.0


@dataclass
class LogData:
    name: str
    t: np.ndarray
    accel: np.ndarray
    mag: np.ndarray
    travel: np.ndarray
    roi_mask: np.ndarray
    zv_points: np.ndarray
    travel_accel: np.ndarray


def load_log(log_name: str) -> LogData:
    ws = np.load(REPO_ROOT / "backend" / "run_artifacts" / log_name / "cache" / "all.npz")
    t = np.asarray(ws["accel/lphp/proj__t"], dtype=float)
    travel = np.asarray(ws["travel__x"][:, 0], dtype=float)
    velocity = np.gradient(travel, t, edge_order=2)
    travel_accel = np.gradient(velocity, t, edge_order=2) / 1000.0
    return LogData(
        name=log_name,
        t=t,
        accel=np.asarray(ws["accel/lphp/proj__x"][:, 0], dtype=float),
        mag=np.asarray(ws["mag/proj/lpf__x"][:, 0], dtype=float),
        travel=travel,
        roi_mask=np.asarray(ws["boring_mask"], dtype=bool),
        zv_points=np.asarray(ws["mag_zv_points"], dtype=int),
        travel_accel=travel_accel,
    )


def prepare_accel(log: LogData, variant: Variant) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    idxs = filter_mag_zv_points(
        log.zv_points,
        log.mag,
        log.t,
        min_prominence_mg=variant.min_prominence_mg,
        min_separation_s=variant.min_separation_s,
    )

    if variant.correction_mode is None:
        return log.accel, idxs, {
            "zv_count": float(len(idxs)),
            "bias_std": 0.0,
            "bias_abs_p95": 0.0,
        }

    corrected, stats = correct_accel_with_zv(
        log.accel,
        log.t,
        idxs,
        mode=variant.correction_mode,
        smooth_bias_s=variant.smooth_bias_s,
    )
    return corrected, idxs, stats


def active_accel_metrics(pred: np.ndarray, gt: np.ndarray, roi_mask: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(pred) & np.isfinite(gt) & roi_mask & (np.abs(gt) >= 0.5)
    err = pred[mask] - gt[mask]
    corr = np.nan
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
    err = centered_error(pred, log.travel, log.roi_mask)
    gt_roi = log.travel[log.roi_mask]
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


def configure_model(variant: Variant) -> RearMagModel:
    return RearMagModel(
        x0_weight=1.0,
        chunking_method=variant.chunking_method,
        chunk_rad=20,
        chunk_min_dx=10.0,
        chunk_max_dx=150.0,
    )


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
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def fit_weighted_model(
    train_model: RearMagModel,
    input_arr: np.ndarray,
    *,
    power_prior: float,
) -> tuple[MagToTravelModel, np.ndarray]:
    model = MagToTravelModel(pred_soft_mg=train_model.pred_soft_mg)

    def residuals(vec: np.ndarray) -> np.ndarray:
        zero_x_preds = model.pred_x(input_arr[:, 0, 0], vec)
        x_acc_preds = input_arr[:, 1, 1:] + zero_x_preds[:, np.newaxis]
        x_mag_preds = model.pred_x(input_arr[:, 0, 1:], vec)
        fit_res = (x_acc_preds - x_mag_preds) * input_arr[:, 2, 1:]
        prior_res = np.asarray(
            [
                (vec[2] - power_prior) * train_model.power_weight,
                vec[0] * train_model.x0_weight,
            ],
            dtype=float,
        )
        return np.concatenate([fit_res.ravel(), prior_res])

    result = scipy.optimize.least_squares(
        residuals,
        x0=np.asarray([0.0, -1.0, 1.0 / 3.0], dtype=float),
        method="trf",
        verbose=0,
        max_nfev=1000,
    )
    model.set_coeffs(result.x)
    train_model.model = model
    return model, result.x.copy()


def fit_mag_model(
    log: LogData,
    accel: np.ndarray,
    zv_points: np.ndarray,
    variant: Variant,
    *,
    max_chunks: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    model = configure_model(variant)
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
    fitted, coeffs = fit_weighted_model(model, input_arr, power_prior=variant.power_prior)
    pred = fitted.pred_x(log.mag)
    return pred - np.percentile(pred, 8.0), coeffs, len(chunks)


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
    variant: Variant,
    log: LogData,
    *,
    max_chunks: int | None,
    run_solver: bool,
    solver_max_nfev: int,
) -> dict[str, object]:
    accel, zv_points, correction_stats = prepare_accel(log, variant)
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
        "mode": variant.correction_mode or "raw",
        "min_prominence_mg": variant.min_prominence_mg,
        "min_separation_s": variant.min_separation_s,
        "chunking": variant.chunking_method,
        "chunks": chunk_count,
        "zv_points": int(correction_stats["zv_count"]),
        "bias_std": float(correction_stats["bias_std"]),
        "bias_abs_p95": float(correction_stats["bias_abs_p95"]),
        "coeff_x0": float(coeffs[0]),
        "coeff_y_scale": float(coeffs[1]),
        "coeff_power": float(coeffs[2]),
    }
    row.update(active_accel_metrics(accel, log.travel_accel, log.roi_mask))
    row.update(travel_metrics(mag_pred, log, "mag"))
    if solver_pred is not None:
        row.update(travel_metrics(solver_pred, log, "solver"))
    return row


def default_variants() -> list[Variant]:
    return [
        Variant("raw_centered", None),
        Variant("raw_zv_s80ms", None, min_separation_s=0.08),
        Variant("raw_zv_p200_s50ms", None, min_prominence_mg=200.0, min_separation_s=0.05),
        Variant("zv_linear_all", "linear_velocity"),
        Variant("zv_linear_s50ms", "linear_velocity", min_separation_s=0.05),
        Variant("zv_linear_s80ms", "linear_velocity", min_separation_s=0.08),
        Variant("zv_linear_s100ms", "linear_velocity", min_separation_s=0.10),
        Variant("zv_linear_p25_s20ms", "linear_velocity", min_prominence_mg=25.0, min_separation_s=0.02),
        Variant("zv_linear_p50_s20ms", "linear_velocity", min_prominence_mg=50.0, min_separation_s=0.02),
        Variant("zv_linear_p50_s50ms", "linear_velocity", min_prominence_mg=50.0, min_separation_s=0.05),
        Variant("zv_linear_p100_s50ms", "linear_velocity", min_prominence_mg=100.0, min_separation_s=0.05),
        Variant("zv_linear_p200_s50ms", "linear_velocity", min_prominence_mg=200.0, min_separation_s=0.05),
        Variant("zv_linear_p500_s80ms", "linear_velocity", min_prominence_mg=500.0, min_separation_s=0.08),
        Variant("zv_pchip_all", "pchip_velocity"),
        Variant("zv_pchip_p200_s50ms", "pchip_velocity", min_prominence_mg=200.0, min_separation_s=0.05),
        Variant("zv_smooth_all", "smoothed_bias"),
        Variant("zv_smooth_p25_s20ms", "smoothed_bias", min_prominence_mg=25.0, min_separation_s=0.02),
        Variant("zv_smooth_p50_s20ms", "smoothed_bias", min_prominence_mg=50.0, min_separation_s=0.02),
        Variant("zv_smooth_p100_s50ms", "smoothed_bias", min_prominence_mg=100.0, min_separation_s=0.05),
        Variant("raw_paired", None, chunking_method="paired_zv"),
        Variant("zv_linear_all_paired", "linear_velocity", chunking_method="paired_zv"),
    ]


def mean_metric(rows: list[dict[str, object]], key: str) -> float:
    vals = np.asarray([float(row[key]) for row in rows if key in row], dtype=float)
    if not np.any(np.isfinite(vals)):
        return float("nan")
    return float(np.nanmean(vals))


def aggregate(rows: list[dict[str, object]], variants: list[Variant]) -> list[dict[str, object]]:
    out = []
    for variant in variants:
        subset = [row for row in rows if row["variant"] == variant.name]
        if not subset:
            continue
        row: dict[str, object] = {
            "variant": variant.name,
            "mode": variant.correction_mode or "raw",
            "chunking": variant.chunking_method,
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
        }
        if np.isfinite(float(row["mean_solver_bin_rmse"])):
            score = float(row["mean_solver_bin_rmse"])
        else:
            score = float(row["mean_mag_bin_rmse"])
        row["score"] = score + 0.03 * float(row["mean_accel_rmse"])
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
    lines = [
        "# Rear ZV Acceleration Correction",
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
                "For each acceleration signal, integrate projected acceleration once, sample the raw integrated "
                "velocity at mag ZV points, interpolate that drift velocity, and subtract its derivative from "
                "the full acceleration series."
            ),
            "",
            f"- Chunk cap during this run: `{max_chunks if max_chunks is not None else 'none'}`.",
            f"- Solver evaluated: `{run_solver}`.",
            "",
            "## Main Findings",
            "",
            (
                f"- Best score in this run: `{best['variant']}` with mean accel RMSE "
                f"`{float(best['mean_accel_rmse']):.3f} m/s^2` and mean mag bin RMSE "
                f"`{float(best['mean_mag_bin_rmse']):.3f} mm`."
            ),
            (
                "- The all-extrema linear velocity-drift correction is the most important baseline because it "
                "uses exactly the same ZV evidence as `centered_zv`, but applies it to the whole acceleration "
                "series before any downstream integration."
            ),
            (
                "- Prominence/separation filters trade off noisy anchor removal against losing real small-amplitude "
                "turning points. Compare `mean_zv_points`, `mean_accel_rmse`, and bin RMSE together rather than "
                "optimizing only one metric."
            ),
            "",
            "## Aggregate Metrics",
            "",
            "| Variant | Mode | Chunking | Accel RMSE | Accel Corr | Mag RMSE | Mag Bin | Mag Worst | Solver RMSE | Solver Bin | Chunks | ZV | Bias p95 | Score |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            f"| `{row['variant']}` | `{row['mode']}` | `{row['chunking']}` | "
            f"{float(row['mean_accel_rmse']):.3f} | {float(row['mean_accel_corr']):.3f} | "
            f"{float(row['mean_mag_rmse']):.3f} | {float(row['mean_mag_bin_rmse']):.3f} | "
            f"{float(row['mean_mag_worst_bin_rmse']):.3f} | {float(row['mean_solver_rmse']):.3f} | "
            f"{float(row['mean_solver_bin_rmse']):.3f} | {float(row['mean_chunks']):.0f} | "
            f"{float(row['mean_zv_points']):.0f} | {float(row['mean_bias_abs_p95']):.3f} | "
            f"{float(row['score']):.3f} |"
        )

    selected_names = [best["variant"], "raw_centered", "zv_linear_all"]
    available_names = {str(row["variant"]) for row in rows}
    selected_names = [
        name
        for name in dict.fromkeys(str(name) for name in selected_names)
        if name in available_names
    ]
    lines.extend(
        [
            "",
            "## Selected Per-Log Metrics",
            "",
            "| Log | Variant | Accel RMSE | Accel Corr | Mag RMSE | Mag Bin | Mag Worst | Solver RMSE | Solver Bin | Chunks | ZV |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for log_name in logs:
        for name in selected_names:
            match = next(row for row in rows if row["log"] == log_name and row["variant"] == name)
            lines.append(
                f"| `{log_name}` | `{name}` | {float(match['accel_rmse']):.3f} | "
                f"{float(match['accel_corr']):.3f} | {float(match['mag_rmse']):.3f} | "
                f"{float(match['mag_bin_rmse']):.3f} | {float(match['mag_worst_bin_rmse']):.3f} | "
                f"{float(match.get('solver_rmse', float('nan'))):.3f} | "
                f"{float(match.get('solver_bin_rmse', float('nan'))):.3f} | "
                f"{int(match['chunks'])} | {int(match['zv_points'])} |"
            )

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rear acceleration ZV correction variants.")
    parser.add_argument(
        "--logs",
        nargs="*",
        default=[f"log{i}_rear" for i in range(148, 155)],
    )
    parser.add_argument("--out-dir", type=Path, default=Path("reports/rear_zv_accel_correction_148_154"))
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--run-solver", action="store_true")
    parser.add_argument("--solver-max-nfev", type=int, default=40)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Optional subset of variant names from default_variants().",
    )
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
                f"chunks={int(row['chunks'])} zv={int(row['zv_points'])}",
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
