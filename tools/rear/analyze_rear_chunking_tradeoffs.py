#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import numpy as np
import scipy.optimize


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import rear_mag_model as rmm  # noqa: E402
from mag_to_travel_model_core import MagToTravelChunk, MagToTravelModel  # noqa: E402


TRAVEL_BIN_EDGES = np.linspace(0.0, 150.0, 6)
MIN_BIN_SAMPLES = 100


@dataclass(frozen=True)
class Variant:
    name: str
    chunking: str
    chunk_weighting: str = "uniform"
    chunk_rad: int = 20
    chunk_min_dx: float = 10.0
    chunk_max_dx: float = 150.0
    min_chunk_dt: float = 0.1
    max_chunk_dt: float = 0.2
    min_chunk_db: float = 500.0
    pair_mode: str = "first_valid"
    min_abs_b_x_corr: float | None = None
    min_db_per_dx: float | None = None
    x0_weight: float = 1.0
    power_weight: float = 1000.0
    power_prior: float = 1.0 / 3.0
    pred_soft_mg: float = 50.0


@dataclass
class LogData:
    name: str
    mag: np.ndarray
    accel: np.ndarray
    t: np.ndarray
    travel: np.ndarray
    roi_mask: np.ndarray
    zv_points: np.ndarray


def quiet_call(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def load_log(log_name: str) -> LogData:
    path = REPO_ROOT / "backend" / "run_artifacts" / log_name / "cache" / "all.npz"
    ws = np.load(path)
    return LogData(
        name=log_name,
        mag=np.asarray(ws["mag/proj/lpf__x"][:, 0], dtype=float),
        accel=np.asarray(ws["accel/lphp/proj__x"][:, 0], dtype=float),
        t=np.asarray(ws["mag/proj/lpf__t"], dtype=float),
        travel=np.asarray(ws["travel__x"][:, 0], dtype=float),
        roi_mask=np.asarray(ws["boring_mask"], dtype=bool),
        zv_points=np.asarray(ws["mag_zv_points"], dtype=int),
    )


def configure_model(variant: Variant, *, chunking_method: str | None = None) -> rmm.RearMagModel:
    model = rmm.RearMagModel(
        x0_weight=variant.x0_weight,
        power_weight=variant.power_weight,
        pred_soft_mg=variant.pred_soft_mg,
        chunking_method=chunking_method or variant.chunking,
        chunk_rad=variant.chunk_rad,
        chunk_min_dx=variant.chunk_min_dx,
        chunk_max_dx=variant.chunk_max_dx,
    )
    model.min_chunk_dt = variant.min_chunk_dt
    model.max_chunk_dt = variant.max_chunk_dt
    model.min_chunk_db = variant.min_chunk_db
    model.pair_mode = variant.pair_mode
    model.min_abs_b_x_corr = variant.min_abs_b_x_corr
    model.min_db_per_dx = variant.min_db_per_dx
    return model


def create_chunks_for_variant(variant: Variant, log: LogData) -> tuple[rmm.RearMagModel, list[MagToTravelChunk], list[str]]:
    model = configure_model(
        variant,
        chunking_method="paired_zv" if variant.chunking == "hybrid" else variant.chunking,
    )
    if variant.chunking == "hybrid":
        chunks: list[MagToTravelChunk] = []
        sources: list[str] = []
        for method in ("paired_zv", "centered_zv"):
            source_model = configure_model(variant, chunking_method=method)
            source_chunks = source_model.create_chunks(log.zv_points, log.mag, log.accel, log.t)
            chunks.extend(source_chunks)
            sources.extend([method] * len(source_chunks))
    else:
        chunks = model.create_chunks(log.zv_points, log.mag, log.accel, log.t)
        sources = [variant.chunking] * len(chunks)

    model.prepare_chunks(chunks)
    keep_chunks: list[MagToTravelChunk] = []
    keep_sources: list[str] = []
    filters = model.get_filter_fns()
    for chunk, source in zip(chunks, sources):
        if all(filter_fn(chunk) for filter_fn in filters):
            keep_chunks.append(chunk)
            keep_sources.append(source)
    model.chunks = keep_chunks
    return model, keep_chunks, keep_sources


def build_chunk_weights(chunks: list[MagToTravelChunk], sources: list[str], weighting: str) -> np.ndarray:
    weights = np.ones(len(chunks), dtype=float)
    if len(chunks) == 0 or weighting == "uniform":
        return weights

    if weighting == "equal_source":
        source_values, source_counts = np.unique(np.asarray(sources), return_counts=True)
        count_by_source = dict(zip(source_values.tolist(), source_counts.tolist(), strict=True))
        weights = np.asarray([1.0 / count_by_source[source] for source in sources], dtype=float)
        return weights / max(float(np.mean(weights)), 1e-12)

    if weighting in {"magbin", "equal_source_magbin"}:
        mags = np.asarray([np.median(chunk.mag) for chunk in chunks], dtype=float)
        edges = np.percentile(mags, [0, 20, 40, 60, 80, 100])
        # Degenerate bins can happen on tiny subsets. Make the right edge inclusive.
        bin_ids = np.digitize(mags, edges[1:-1], right=False)
        for bin_id in np.unique(bin_ids):
            mask = bin_ids == bin_id
            weights[mask] /= max(int(np.sum(mask)), 1)
        if weighting == "equal_source_magbin":
            source_values, source_counts = np.unique(np.asarray(sources), return_counts=True)
            count_by_source = dict(zip(source_values.tolist(), source_counts.tolist(), strict=True))
            weights *= np.asarray([1.0 / count_by_source[source] for source in sources], dtype=float)
        return weights / max(float(np.mean(weights)), 1e-12)

    raise ValueError(f"Unknown chunk weighting {weighting}")


def fit_weighted_model(
    train_model: rmm.RearMagModel,
    input_arr: np.ndarray,
    chunk_weights: np.ndarray,
    *,
    power_prior: float,
    guess_vec: np.ndarray,
) -> tuple[MagToTravelModel, np.ndarray]:
    model = MagToTravelModel(pred_soft_mg=train_model.pred_soft_mg)
    chunk_weights = np.asarray(chunk_weights, dtype=float)
    chunk_weights = chunk_weights / max(float(np.mean(chunk_weights)), 1e-12)
    chunk_sqrt_weights = np.sqrt(chunk_weights)

    def residuals(vec: np.ndarray) -> np.ndarray:
        zero_x_mags = input_arr[:, 0, 0]
        zero_x_preds = model.pred_x(zero_x_mags, vec)
        x_acc_preds = input_arr[:, 1, 1:] + zero_x_preds[:, np.newaxis]
        x_mag_preds = model.pred_x(input_arr[:, 0, 1:], vec)
        mask = input_arr[:, 2, 1:]
        fit_res = (x_acc_preds - x_mag_preds) * mask * chunk_sqrt_weights[:, np.newaxis]
        prior_res = np.array(
            [
                (vec[2] - power_prior) * train_model.power_weight,
                vec[0] * train_model.x0_weight,
            ],
            dtype=float,
        )
        return np.concatenate([fit_res.ravel(), prior_res])

    result = scipy.optimize.least_squares(
        residuals,
        x0=np.asarray(guess_vec, dtype=float),
        method="trf",
        verbose=0,
        max_nfev=1000,
    )
    model.set_coeffs(result.x)
    train_model.model = model
    return model, result.x.copy()


def centered_error(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pred_roi = np.asarray(pred, dtype=float)[mask]
    gt_roi = np.asarray(gt, dtype=float)[mask]
    return (pred_roi - np.mean(pred_roi)) - (gt_roi - np.mean(gt_roi))


def bin_stats(err: np.ndarray, gt_roi: np.ndarray) -> tuple[list[float], list[int], float, float]:
    rmses: list[float] = []
    counts: list[int] = []
    eligible_mses: list[float] = []
    for idx, (lo, hi) in enumerate(zip(TRAVEL_BIN_EDGES[:-1], TRAVEL_BIN_EDGES[1:])):
        mask = (gt_roi >= lo) & ((gt_roi <= hi) if idx == len(TRAVEL_BIN_EDGES) - 2 else (gt_roi < hi))
        count = int(np.sum(mask))
        counts.append(count)
        if count:
            rmse = float(np.sqrt(np.mean(err[mask] ** 2)))
        else:
            rmse = float("nan")
        rmses.append(rmse)
        if count >= MIN_BIN_SAMPLES:
            eligible_mses.append(float(np.mean(err[mask] ** 2)))
    bin_rmse = float(np.sqrt(np.mean(eligible_mses))) if eligible_mses else float("nan")
    worst_bin_rmse = float(np.nanmax(np.asarray(rmses, dtype=float)))
    return rmses, counts, bin_rmse, worst_bin_rmse


def evaluate_variant_on_log(variant: Variant, log: LogData) -> dict[str, object]:
    model, chunks, sources = create_chunks_for_variant(variant, log)
    if not chunks:
        raise ValueError(f"No chunks for {variant.name} on {log.name}")
    input_arr = quiet_call(model.format_chunks_for_fit, chunks)
    weights = build_chunk_weights(chunks, sources, variant.chunk_weighting)
    fitted, coeffs = fit_weighted_model(
        model,
        input_arr,
        weights,
        power_prior=variant.power_prior,
        guess_vec=np.asarray([0.0, -1.0, 1.0 / 3.0], dtype=float),
    )
    pred = fitted.pred_x(log.mag)
    # This mirrors GetRearMagToTravelModel. Centered metrics are invariant to
    # this offset, but keeping it here makes raw inspection less surprising.
    pred_adj = pred - np.percentile(pred, 8)
    err = centered_error(pred_adj, log.travel, log.roi_mask)
    gt_roi = log.travel[log.roi_mask]
    bin_rmses, bin_counts, bin_rmse, worst_bin_rmse = bin_stats(err, gt_roi)
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    source_counts = {source: sources.count(source) for source in sorted(set(sources))}
    return {
        "variant": variant.name,
        "log": log.name,
        "rmse": rmse,
        "mae": mae,
        "bin_rmse": bin_rmse,
        "worst_bin_rmse": worst_bin_rmse,
        "chunks": len(chunks),
        "coeff_x0": float(coeffs[0]),
        "coeff_y_scale": float(coeffs[1]),
        "coeff_power": float(coeffs[2]),
        "source_counts": source_counts,
        **{f"bin{i}_rmse": bin_rmses[i] for i in range(len(bin_rmses))},
        **{f"bin{i}_n": bin_counts[i] for i in range(len(bin_counts))},
    }


def fit_variant_prediction(variant: Variant, log: LogData) -> tuple[np.ndarray, np.ndarray, int]:
    model, chunks, sources = create_chunks_for_variant(variant, log)
    input_arr = quiet_call(model.format_chunks_for_fit, chunks)
    weights = build_chunk_weights(chunks, sources, variant.chunk_weighting)
    fitted, coeffs = fit_weighted_model(
        model,
        input_arr,
        weights,
        power_prior=variant.power_prior,
        guess_vec=np.asarray([0.0, -1.0, 1.0 / 3.0], dtype=float),
    )
    pred = fitted.pred_x(log.mag)
    return pred - np.percentile(pred, 8), coeffs, len(chunks)


def evaluate_prediction_row(
    *,
    variant_name: str,
    log: LogData,
    pred_adj: np.ndarray,
    chunks: int,
    coeffs: np.ndarray,
    source_counts: dict[str, int] | None = None,
) -> dict[str, object]:
    err = centered_error(pred_adj, log.travel, log.roi_mask)
    gt_roi = log.travel[log.roi_mask]
    bin_rmses, bin_counts, bin_rmse, worst_bin_rmse = bin_stats(err, gt_roi)
    return {
        "variant": variant_name,
        "log": log.name,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bin_rmse": bin_rmse,
        "worst_bin_rmse": worst_bin_rmse,
        "chunks": chunks,
        "coeff_x0": float(coeffs[0]) if len(coeffs) > 0 else float("nan"),
        "coeff_y_scale": float(coeffs[1]) if len(coeffs) > 1 else float("nan"),
        "coeff_power": float(coeffs[2]) if len(coeffs) > 2 else float("nan"),
        "source_counts": source_counts or {},
        **{f"bin{i}_rmse": bin_rmses[i] for i in range(len(bin_rmses))},
        **{f"bin{i}_n": bin_counts[i] for i in range(len(bin_counts))},
    }


def evaluate_mag_gated_blend_on_log(
    variant_name: str,
    log: LogData,
    *,
    low_mag_percentile: float,
    high_mag_percentile: float,
) -> dict[str, object]:
    paired_variant = Variant(name="paired_for_gate", chunking="paired_zv")
    centered_variant = Variant(
        name="centered_for_gate",
        chunking="centered_zv",
        power_prior=0.28,
    )
    paired_pred, _, paired_chunks = fit_variant_prediction(paired_variant, log)
    centered_pred, _, centered_chunks = fit_variant_prediction(centered_variant, log)

    mag_roi = log.mag[log.roi_mask]
    low_mag = float(np.percentile(mag_roi, low_mag_percentile))
    high_mag = float(np.percentile(mag_roi, high_mag_percentile))
    centered_weight = np.clip((high_mag - log.mag) / max(high_mag - low_mag, 1e-9), 0.0, 1.0)
    pred_adj = (1.0 - centered_weight) * paired_pred + centered_weight * centered_pred

    return evaluate_prediction_row(
        variant_name=variant_name,
        log=log,
        pred_adj=pred_adj,
        chunks=paired_chunks + centered_chunks,
        coeffs=np.asarray([np.nan, np.nan, np.nan], dtype=float),
        source_counts={"paired_zv": paired_chunks, "centered_zv": centered_chunks},
    )


def mean_metric(rows: list[dict[str, object]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    if not np.any(np.isfinite(values)):
        return float("nan")
    return float(np.nanmean(values))


def mean_abs_metric(rows: list[dict[str, object]], key: str) -> float:
    values = np.asarray([abs(float(row[key])) for row in rows], dtype=float)
    if not np.any(np.isfinite(values)):
        return float("nan")
    return float(np.nanmean(values))


def aggregate_rows(rows: list[dict[str, object]], variant: Variant) -> dict[str, object]:
    variant_rows = [row for row in rows if row["variant"] == variant.name]
    out: dict[str, object] = {
        "variant": variant.name,
        "chunking": variant.chunking,
        "weighting": variant.chunk_weighting,
        "mean_rmse": mean_metric(variant_rows, "rmse"),
        "mean_bin_rmse": mean_metric(variant_rows, "bin_rmse"),
        "mean_worst_bin_rmse": mean_metric(variant_rows, "worst_bin_rmse"),
        "max_worst_bin_rmse": float(np.nanmax([float(row["worst_bin_rmse"]) for row in variant_rows])),
        "mean_chunks": mean_metric(variant_rows, "chunks"),
        "mean_power": mean_metric(variant_rows, "coeff_power"),
        "mean_scale_abs": mean_abs_metric(variant_rows, "coeff_y_scale"),
    }
    for i in range(len(TRAVEL_BIN_EDGES) - 1):
        out[f"mean_bin{i}_rmse"] = mean_metric(variant_rows, f"bin{i}_rmse")
    out["score"] = (
        float(out["mean_rmse"])
        + 0.50 * float(out["mean_bin_rmse"])
        + 0.25 * float(out["mean_worst_bin_rmse"])
    )
    return out


def default_variants() -> list[Variant]:
    base = Variant(name="paired_default", chunking="paired_zv")
    return [
        base,
        replace(base, name="centered_default", chunking="centered_zv"),
        replace(base, name="hybrid_uniform", chunking="hybrid"),
        replace(base, name="hybrid_equal_source", chunking="hybrid", chunk_weighting="equal_source"),
        replace(base, name="paired_magbin_weighted", chunk_weighting="magbin"),
        replace(base, name="centered_magbin_weighted", chunking="centered_zv", chunk_weighting="magbin"),
        replace(base, name="hybrid_magbin_weighted", chunking="hybrid", chunk_weighting="magbin"),
        replace(base, name="hybrid_equal_source_magbin", chunking="hybrid", chunk_weighting="equal_source_magbin"),
        replace(base, name="paired_pair_max_db_dt", pair_mode="max_db_per_dt"),
        replace(base, name="paired_pair_max_abs_db", pair_mode="max_abs_db"),
        replace(base, name="paired_min_db_250", min_chunk_db=250.0),
        replace(base, name="paired_min_db_750", min_chunk_db=750.0),
        replace(base, name="paired_dx_180", chunk_max_dx=180.0),
        replace(base, name="paired_corr_0p7", min_abs_b_x_corr=0.7),
        replace(base, name="centered_corr_0p7", chunking="centered_zv", min_abs_b_x_corr=0.7),
        replace(base, name="centered_rad_12", chunking="centered_zv", chunk_rad=12),
        replace(base, name="centered_rad_30", chunking="centered_zv", chunk_rad=30),
        replace(base, name="paired_power_prior_0p28", power_prior=0.28),
        replace(base, name="paired_power_prior_0p38", power_prior=0.38),
        replace(base, name="centered_power_prior_0p28", chunking="centered_zv", power_prior=0.28),
        replace(base, name="centered_power_prior_0p38", chunking="centered_zv", power_prior=0.38),
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, logs: list[str], variants: list[Variant], rows: list[dict[str, object]], aggregate: list[dict[str, object]]) -> None:
    sorted_by_score = sorted(aggregate, key=lambda row: float(row["score"]))
    sorted_by_worst = sorted(aggregate, key=lambda row: float(row["mean_worst_bin_rmse"]))
    sorted_by_rmse = sorted(aggregate, key=lambda row: float(row["mean_rmse"]))

    lines = [
        "# Rear Mag Chunking Tradeoff Analysis",
        "",
        "Logs:",
        "",
    ]
    lines.extend(f"- `{log}`" for log in logs)
    lines.extend(
        [
            "",
            "Metrics are centered on `boring_mask` to match `tools/stats_aggregator.py`.",
            "`bin_rmse` is the equal-weight average over the five 0-150 mm travel bins; `worst_bin_rmse` is the largest of those bins.",
            "",
            "## Main Findings",
            "",
            (
                "- The best balanced result is `mag_gated_blend_p5_p50`: train one `paired_zv` curve and one "
                "`centered_zv` curve with a `0.28` power prior, use the centered curve at the low-mag/high-travel "
                "end, and fade to the paired curve by the median mag."
            ),
            (
                "- Simply mixing both chunk sets in one least-squares fit does not combine the advantages. "
                "`hybrid_uniform` improves sample RMSE versus `paired_default`, but its worst-bin RMSE stays much "
                "closer to paired than centered."
            ),
            (
                "- `centered_zv` is the best single-model family for equal-bin and high-travel error. "
                "A lower power prior (`0.28`) improves it a bit more."
            ),
            (
                "- `paired_zv` remains strong in the dense low/mid travel range, but it leaves large 90-150 mm "
                "tail errors. Mag-bin chunk weighting and pair selection changes did not materially fix that."
            ),
            (
                "- A strict paired-chunk correlation filter (`min_abs_b_x_corr=0.7`) gets the lowest mean worst-bin "
                "RMSE among single fits, but sample RMSE is too high to be a good default."
            ),
            "",
            "## Ranking Summary",
            "",
            f"- Best composite score: `{sorted_by_score[0]['variant']}`.",
            f"- Best mean worst-bin RMSE: `{sorted_by_worst[0]['variant']}`.",
            f"- Best mean sample RMSE: `{sorted_by_rmse[0]['variant']}`.",
            "",
            "## Aggregate Metrics",
            "",
            "| Variant | Chunking | Weighting | RMSE | Bin RMSE | Worst Bin | Max Worst | Chunks | Power | |Scale| | Score |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted_by_score:
        lines.append(
            f"| `{row['variant']}` | `{row['chunking']}` | `{row['weighting']}` | "
            f"{float(row['mean_rmse']):.3f} | {float(row['mean_bin_rmse']):.3f} | "
            f"{float(row['mean_worst_bin_rmse']):.3f} | {float(row['max_worst_bin_rmse']):.3f} | "
            f"{float(row['mean_chunks']):.0f} | {float(row['mean_power']):.3f} | "
            f"{float(row['mean_scale_abs']):.1f} | {float(row['score']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Travel-Bin Means",
            "",
            "| Variant | 0-30 | 30-60 | 60-90 | 90-120 | 120-150 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted_by_score:
        lines.append(
            f"| `{row['variant']}` | "
            f"{float(row['mean_bin0_rmse']):.3f} | {float(row['mean_bin1_rmse']):.3f} | "
            f"{float(row['mean_bin2_rmse']):.3f} | {float(row['mean_bin3_rmse']):.3f} | "
            f"{float(row['mean_bin4_rmse']):.3f} |"
        )

    best_names = [sorted_by_score[0]["variant"], "paired_default", "centered_default", "hybrid_equal_source"]
    best_names = list(dict.fromkeys(str(name) for name in best_names))
    lines.extend(
        [
            "",
            "## Selected Per-Log Metrics",
            "",
            "| Log | Variant | RMSE | Bin RMSE | Worst Bin | b0 | b1 | b2 | b3 | b4 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for log in logs:
        for name in best_names:
            row = next(r for r in rows if r["log"] == log and r["variant"] == name)
            lines.append(
                f"| `{log}` | `{name}` | {float(row['rmse']):.3f} | {float(row['bin_rmse']):.3f} | "
                f"{float(row['worst_bin_rmse']):.3f} | {float(row['bin0_rmse']):.3f} | "
                f"{float(row['bin1_rmse']):.3f} | {float(row['bin2_rmse']):.3f} | "
                f"{float(row['bin3_rmse']):.3f} | {float(row['bin4_rmse']):.3f} |"
            )

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rear mag chunking/filter/training tradeoffs.")
    parser.add_argument(
        "--logs",
        nargs="*",
        default=["log148_rear", "log149_rear", "log150_rear", "log151_rear", "log152_rear", "log153_rear"],
    )
    parser.add_argument("--out-dir", type=Path, default=Path("reports/rear_chunking_tradeoffs_148_153"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs = [load_log(log_name) for log_name in args.logs]
    variants = default_variants()
    rows: list[dict[str, object]] = []
    for variant in variants:
        print(f"Evaluating {variant.name}")
        for log in logs:
            row = evaluate_variant_on_log(variant, log)
            rows.append(row)
            print(
                f"  {log.name}: rmse={row['rmse']:.3f} bin={row['bin_rmse']:.3f} "
                f"worst={row['worst_bin_rmse']:.3f} chunks={row['chunks']}"
            )

    gated_variant = Variant(
        name="mag_gated_blend_p5_p50",
        chunking="paired_zv+centered_zv",
        chunk_weighting="mag_gate_p5_p50",
    )
    print(f"Evaluating {gated_variant.name}")
    for log in logs:
        row = evaluate_mag_gated_blend_on_log(
            gated_variant.name,
            log,
            low_mag_percentile=5.0,
            high_mag_percentile=50.0,
        )
        rows.append(row)
        print(
            f"  {log.name}: rmse={row['rmse']:.3f} bin={row['bin_rmse']:.3f} "
            f"worst={row['worst_bin_rmse']:.3f} chunks={row['chunks']}"
        )

    report_variants = variants + [gated_variant]
    aggregate = [aggregate_rows(rows, variant) for variant in report_variants]
    aggregate.sort(key=lambda row: float(row["score"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_log.csv", rows)
    write_csv(args.out_dir / "aggregate.csv", aggregate)
    write_report(args.out_dir / "report.md", args.logs, report_variants, rows, aggregate)

    print()
    print("Top variants by composite score:")
    for row in aggregate[:8]:
        print(
            f"{row['variant']:28s} rmse={row['mean_rmse']:.3f} bin={row['mean_bin_rmse']:.3f} "
            f"worst={row['mean_worst_bin_rmse']:.3f} score={row['score']:.3f}"
        )
    print(f"Wrote {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
