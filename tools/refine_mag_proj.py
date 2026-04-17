import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import spearmanr

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.stats_aggregator import DEFAULT_LOGS
#DEFAULT_LOGS = ["log022", "log029", "log030", "log031", "log038", "log056_ccdh", "log078", "log079", "log080", "log085", "log086", "log088", "log091", '']
DEFAULT_METHODS = [
    "oracle_travel_ridge",
    "himag_mean",
    "himag_mean_norm",
    "accel_chunk_peak_mean",
    "accel_chunk_peak_pca",
]


@dataclass
class LogData:
    log_name: str
    mag: np.ndarray
    travel: np.ndarray
    active_mask: np.ndarray
    accel_hp_proj: np.ndarray
    time_s: np.ndarray


@dataclass
class MotionChunk:
    still_mag_mean: np.ndarray
    bump_mag: np.ndarray
    pseudo_travel_mm: np.ndarray
    sign: float
    score_mm: float


@dataclass
class MethodEstimate:
    method_name: str
    vector: np.ndarray
    detail_items: dict[str, float | int | str] = field(default_factory=dict)


@dataclass
class CurveFit:
    coeffs: np.ndarray
    preds: np.ndarray
    active_rmse: float
    active_corr: float
    all_rmse: float
    all_corr: float


@dataclass
class MethodEval:
    method_name: str
    vector: np.ndarray
    active_corr_abs: float
    all_corr_abs: float
    oracle_align_abs: float
    curve_all_rmse: float | None
    curve_all_corr: float | None
    detail_items: dict[str, float | int | str]


def flatten_1d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr.reshape(-1)


def safe_corr(a: np.ndarray, b: np.ndarray, use_spearman=True) -> float:
    a = flatten_1d(a)
    b = flatten_1d(b)
    if a.shape != b.shape:
        raise ValueError(f"Correlation inputs must match in shape, got {a.shape} vs {b.shape}")
    if len(a) < 2:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    if use_spearman:
        corr = spearmanr(a, b).correlation
        return float(corr) if np.isfinite(corr) else float("nan")
    else:
        return np.corrcoef(a, b)[0, 1]


def safe_abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    corr = safe_corr(a, b)
    return float(abs(corr)) if np.isfinite(corr) else float("nan")


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a near-zero vector")
    return vector / norm


def pred_x(mag_proj: np.ndarray, x0: float, y_scale: float, power: float, y0: float = 0.0) -> np.ndarray:
    shifted = np.maximum(flatten_1d(mag_proj) - x0, 0.0)
    return (shifted**power) * y_scale + y0


def fit_curve(mag_proj: np.ndarray, travel: np.ndarray, eval_mag_proj: np.ndarray, eval_travel: np.ndarray) -> CurveFit:
    mag_proj = flatten_1d(mag_proj)
    travel = flatten_1d(travel)
    eval_mag_proj = flatten_1d(eval_mag_proj)
    eval_travel = flatten_1d(eval_travel)

    if mag_proj.shape != travel.shape:
        raise ValueError(f"Curve-fit inputs must match in shape, got {mag_proj.shape} vs {travel.shape}")
    if len(mag_proj) < 8:
        raise ValueError(f"Need at least 8 samples to fit the mag curve, got {len(mag_proj)}")

    x0_guess = float(np.percentile(mag_proj, 1))
    shifted_p95 = max(float(np.percentile(np.maximum(mag_proj - x0_guess, 0.0), 95)), 1.0)
    travel_span = max(float(np.ptp(travel)), 1.0)
    y_scale_guess = max(travel_span / (shifted_p95 ** (1.0 / 3.0)), 1e-3)
    y0_guess = float(np.percentile(travel, 1))

    mag_span = max(float(np.ptp(mag_proj)), 1.0)
    x0_lower = float(np.min(mag_proj) - 0.25 * mag_span)
    x0_upper = float(np.percentile(mag_proj, 60))
    y0_lower = float(np.min(travel) - 0.25 * travel_span)
    y0_upper = float(np.max(travel) + 0.25 * travel_span)
    y_scale_upper = max(y_scale_guess * 10.0, 1.0)

    def residuals(params: np.ndarray) -> np.ndarray:
        return pred_x(mag_proj, *params) - travel

    result = least_squares(
        fun=residuals,
        x0=np.array([x0_guess, y_scale_guess, 1.0 / 3.0, y0_guess]),
        bounds=(
            np.array([x0_lower, 0.0, 0.05, y0_lower]),
            np.array([x0_upper, y_scale_upper, 3.0, y0_upper]),
        ),
        method="trf",
        max_nfev=4000,
    )

    preds = pred_x(eval_mag_proj, *result.x)
    active_pred = pred_x(mag_proj, *result.x)
    return CurveFit(
        coeffs=result.x,
        preds=preds,
        active_rmse=float(np.sqrt(np.mean((active_pred - travel) ** 2))),
        active_corr=safe_corr(active_pred, travel),
        all_rmse=float(np.sqrt(np.mean((preds - eval_travel) ** 2))),
        all_corr=safe_corr(preds, eval_travel),
    )


def load_log_data(log_name: str, cache_root: Path) -> LogData:
    cache_path = cache_root / log_name / "cache" / "all.npz"
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    cache = np.load(cache_path)
    mag = np.asarray(cache["mag/lpf__x"], dtype=float)
    travel = flatten_1d(cache["travel__x"])
    active_mask = cache["boring_mask"].astype(bool)
    accel_hp_proj = flatten_1d(cache["accel/lpfhp/proj__x"])
    time_s = flatten_1d(cache["mag/lpf__t"])

    if mag.ndim != 2 or mag.shape[1] != 3:
        raise ValueError(f"{log_name}: expected mag/lpf__x shape (N, 3), got {mag.shape}")
    n = len(travel)
    if mag.shape[0] != n or len(active_mask) != n or len(accel_hp_proj) != n or len(time_s) != n:
        raise ValueError(f"{log_name}: cache arrays do not align in length")

    finite_mask = np.isfinite(travel) & np.isfinite(accel_hp_proj) & np.all(np.isfinite(mag), axis=1)
    active_mask = active_mask & finite_mask
    if np.sum(active_mask) < 8:
        raise ValueError(f"{log_name}: not enough finite active samples")

    return LogData(
        log_name=log_name,
        mag=mag,
        travel=travel,
        active_mask=active_mask,
        accel_hp_proj=accel_hp_proj,
        time_s=time_s,
    )


def fit_projection_from_target(mag: np.ndarray, target: np.ndarray, mask: np.ndarray, ridge: float) -> np.ndarray:
    mag = np.asarray(mag, dtype=float)
    target = flatten_1d(target)
    mask = np.asarray(mask, dtype=bool).reshape(-1)

    if np.sum(mask) < 8:
        raise ValueError(f"Need at least 8 masked samples, got {np.sum(mask)}")

    x_train = mag[mask]
    y_train = target[mask]
    x_centered = x_train - np.mean(x_train, axis=0, keepdims=True)
    y_centered = y_train - np.mean(y_train)

    cov = (x_centered.T @ x_centered) / max(len(x_train) - 1, 1)
    cov_scale = max(float(np.trace(cov) / cov.shape[0]), 1.0)
    cross_cov = (x_centered.T @ y_centered) / max(len(x_train) - 1, 1)

    vector = np.linalg.solve(cov + np.eye(3) * ridge * cov_scale, cross_cov)
    return normalize_vector(vector)


def extract_motion_chunks(
    data: LogData,
    still_len_s: float,
    bump_len_s: float,
    stride_s: float,
    still_a_max: float,
    bump_dx_min_mm: float,
    skip_chunks: int,
) -> list[MotionChunk]:
    dt_med = float(np.median(np.diff(data.time_s)))
    fs_hz = 1.0 / max(dt_med, 1e-6)
    still_len = max(int(still_len_s * fs_hz), 2)
    bump_len = max(int(bump_len_s * fs_hz), 4)
    stride = max(int(stride_s * fs_hz), 1)
    chunk_len = still_len + bump_len

    dt_s = np.diff(data.time_s, prepend=data.time_s[0] - dt_med)
    chunks: list[MotionChunk] = []
    skip = 0
    for start in range(0, len(data.accel_hp_proj) - chunk_len, stride):
        if skip > 0:
            skip -= 1
            continue

        kept = False
        for reverse in (False, True):
            idxs = np.arange(start, start + chunk_len)
            if reverse:
                idxs = idxs[::-1]

            still_idxs = idxs[:still_len]
            bump_idxs = idxs[still_len:]

            if np.max(np.abs(data.accel_hp_proj[still_idxs])) > still_a_max:
                continue

            accel_bump = data.accel_hp_proj[bump_idxs]
            dt_bump = dt_s[bump_idxs]
            vel_proxy = np.cumsum(accel_bump * dt_bump)
            pos_proxy_mm = np.cumsum(vel_proxy * dt_bump) * 1000.0

            peak_idx = int(np.argmax(np.abs(pos_proxy_mm)))
            score_mm = float(abs(pos_proxy_mm[peak_idx]))
            sign = float(np.sign(pos_proxy_mm[peak_idx]))

            if score_mm < bump_dx_min_mm or sign == 0.0:
                continue

            chunks.append(
                MotionChunk(
                    still_mag_mean=np.mean(data.mag[still_idxs], axis=0),
                    bump_mag=data.mag[bump_idxs],
                    pseudo_travel_mm=pos_proxy_mm,
                    sign=sign,
                    score_mm=score_mm,
                )
            )
            skip = skip_chunks
            kept = True
            break

        if kept:
            continue

    return chunks


def get_peak_deltas(chunks: list[MotionChunk]) -> tuple[np.ndarray, np.ndarray]:
    if not chunks:
        raise ValueError("No motion chunks found")

    deltas = []
    weights = []
    for chunk in chunks:
        peak_idx = int(np.argmax(np.abs(chunk.pseudo_travel_mm)))
        delta = (chunk.bump_mag[peak_idx] - chunk.still_mag_mean) * chunk.sign
        deltas.append(delta)
        weights.append(chunk.score_mm)

    return np.asarray(deltas, dtype=float), np.asarray(weights, dtype=float)


def estimate_oracle_travel_ridge(data: LogData, args: argparse.Namespace) -> MethodEstimate:
    vector = fit_projection_from_target(data.mag, data.travel, data.active_mask, ridge=args.ridge)
    return MethodEstimate(
        method_name="oracle_travel_ridge",
        vector=vector,
        detail_items={"active_samples": int(np.sum(data.active_mask))},
    )


def estimate_himag_mean(data: LogData, args: argparse.Namespace, normalize=False) -> MethodEstimate:
    mag_norm = np.linalg.norm(data.mag, axis=1)
    train_mask = mag_norm >= args.pipeline_mag_threshold
    if np.sum(train_mask) < 8:
        raise ValueError(
            f"Only {np.sum(train_mask)} samples above --pipeline-mag-threshold={args.pipeline_mag_threshold:.0f}"
        )
    if normalize:
        mag_i = data.mag[train_mask] / (np.linalg.norm(data.mag[train_mask], axis=1, keepdims=True) + 1e-8)
    else:
        mag_i = data.mag[train_mask]
    vector = normalize_vector(np.mean(mag_i, axis=0))
    return MethodEstimate(
        method_name="himag_mean" + ("_norm" if normalize else ""),
        vector=vector,
        detail_items={
            "threshold_mg": int(args.pipeline_mag_threshold),
            "train_samples": int(np.sum(train_mask)),
        },
    )


def estimate_himag_mean_norm(data: LogData, args: argparse.Namespace) -> MethodEstimate:
    return estimate_himag_mean(data, args, normalize=True)


def estimate_accel_chunk_peak_mean(data: LogData, args: argparse.Namespace) -> MethodEstimate:
    chunks = extract_motion_chunks(
        data=data,
        still_len_s=args.still_len_s,
        bump_len_s=args.bump_len_s,
        stride_s=args.stride_s,
        still_a_max=args.still_a_max,
        bump_dx_min_mm=args.bump_dx_min_mm,
        skip_chunks=args.skip_chunks,
    )
    deltas, weights = get_peak_deltas(chunks)
    vector = normalize_vector(np.sum(deltas * weights[:, np.newaxis], axis=0))
    return MethodEstimate(
        method_name="accel_chunk_peak_mean",
        vector=vector,
        detail_items={"chunks": len(chunks), "median_dx_mm": float(np.median(weights))},
    )


def estimate_accel_chunk_peak_pca(data: LogData, args: argparse.Namespace) -> MethodEstimate:
    chunks = extract_motion_chunks(
        data=data,
        still_len_s=args.still_len_s,
        bump_len_s=args.bump_len_s,
        stride_s=args.stride_s,
        still_a_max=args.still_a_max,
        bump_dx_min_mm=args.bump_dx_min_mm,
        skip_chunks=args.skip_chunks,
    )
    deltas, weights = get_peak_deltas(chunks)
    weighted_cov = (deltas * weights[:, np.newaxis]).T @ deltas / max(float(np.sum(weights)), 1.0)
    eigvals, eigvecs = np.linalg.eigh(weighted_cov)
    vector = eigvecs[:, int(np.argmax(eigvals))]
    if np.dot(vector, np.sum(deltas * weights[:, np.newaxis], axis=0)) < 0:
        vector *= -1.0
    return MethodEstimate(
        method_name="accel_chunk_peak_pca",
        vector=normalize_vector(vector),
        detail_items={"chunks": len(chunks), "median_dx_mm": float(np.median(weights))},
    )


METHOD_REGISTRY = {
    "oracle_travel_ridge": estimate_oracle_travel_ridge,
    "himag_mean": estimate_himag_mean,
    "himag_mean_norm": estimate_himag_mean_norm,

    "accel_chunk_peak_mean": estimate_accel_chunk_peak_mean,
    "accel_chunk_peak_pca": estimate_accel_chunk_peak_pca,
}


def orient_vector_for_eval(vector: np.ndarray, data: LogData) -> np.ndarray:
    active_proj = data.mag[data.active_mask] @ vector
    corr = safe_corr(active_proj, data.travel[data.active_mask])
    if np.isfinite(corr) and corr < 0:
        return -vector
    return vector


def evaluate_method(
    data: LogData,
    estimate: MethodEstimate,
    oracle_vector: np.ndarray,
    skip_curve_fit: bool,
) -> MethodEval:
    vector_eval = orient_vector_for_eval(estimate.vector, data)
    mag_proj = data.mag @ vector_eval
    active_proj = mag_proj[data.active_mask]
    active_travel = data.travel[data.active_mask]

    curve = None
    if not skip_curve_fit:
        curve = fit_curve(
            active_proj,
            active_travel,
            mag_proj,
            data.travel,
        )

    return MethodEval(
        method_name=estimate.method_name,
        vector=vector_eval,
        active_corr_abs=safe_abs_corr(active_proj, active_travel),
        all_corr_abs=safe_abs_corr(mag_proj, data.travel),
        oracle_align_abs=float(abs(np.dot(vector_eval, oracle_vector))),
        curve_all_rmse=None if curve is None else curve.all_rmse,
        curve_all_corr=None if curve is None else float(abs(curve.all_corr)),
        detail_items=estimate.detail_items,
    )


def format_detail_items(detail_items: dict[str, float | int | str]) -> str:
    if not detail_items:
        return ""
    parts = []
    for key, value in detail_items.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.1f}")
        else:
            parts.append(f"{key}={value}")
    return " " + " ".join(parts)


def summarize_log(data: LogData, args: argparse.Namespace) -> dict[str, MethodEval]:
    estimates = {}
    for method_name in args.methods:
        estimate_fn = METHOD_REGISTRY[method_name]
        estimates[method_name] = estimate_fn(data, args)

    if "oracle_travel_ridge" in estimates:
        oracle_vector = estimates["oracle_travel_ridge"].vector
    else:
        oracle_vector = estimate_oracle_travel_ridge(data, args).vector

    evals = {
        method_name: evaluate_method(
            data=data,
            estimate=estimate,
            oracle_vector=oracle_vector,
            skip_curve_fit=args.skip_curve_fit,
        )
        for method_name, estimate in estimates.items()
    }

    print(f"\n{data.log_name}")

    for method_name in args.methods:
        result = evals[method_name]
        curve_bits = ""
        if result.curve_all_rmse is not None and result.curve_all_corr is not None:
            curve_bits = f" curve_rmse={result.curve_all_rmse:.3f} curve_corr={result.curve_all_corr:.4f}"
        print(
            f"  {method_name:22s}"
            f" abs_corr={result.active_corr_abs:.4f}"
            f" all_corr={result.all_corr_abs:.4f}"
            f" oracle_align={result.oracle_align_abs:.4f}"
            f"{curve_bits}"
            f"{format_detail_items(result.detail_items)}"
        )

    unsup_names = [name for name in args.methods if name != "oracle_travel_ridge"]
    if unsup_names:
        best_name = max(unsup_names, key=lambda name: evals[name].active_corr_abs)
        oracle_corr = evals["oracle_travel_ridge"].active_corr_abs if "oracle_travel_ridge" in evals else float("nan")
        oracle_ratio = (
            100.0 * evals[best_name].active_corr_abs / oracle_corr
            if np.isfinite(oracle_corr) and oracle_corr > 1e-9
            else float("nan")
        )
        vec = evals[best_name].vector
        print(
            f"  best_no_travel={best_name}"
            f" abs_corr={evals[best_name].active_corr_abs:.4f}"
            f" oracle_pct={oracle_ratio:.1f}%"
            f" vector=[{vec[0]: .6f} {vec[1]: .6f} {vec[2]: .6f}]"
        )

    return evals


def print_summary(all_results: dict[str, dict[str, MethodEval]], methods: list[str]) -> None:
    if not all_results:
        return
    
    print(
        "  note:"
        " sign is allowed to flip during evaluation so abs correlation reflects vector quality,"
        " not sign ambiguity"
    )

    print("\nSummary")
    oracle_values = {}
    if "oracle_travel_ridge" in methods:
        oracle_values = {
            log_name: log_results["oracle_travel_ridge"].active_corr_abs
            for log_name, log_results in all_results.items()
            if "oracle_travel_ridge" in log_results
        }

    summary_rows = []
    for method_name in methods:
        evals = [log_results[method_name] for log_results in all_results.values() if method_name in log_results]
        if not evals:
            continue

        mean_abs_corr = float(np.mean([ev.active_corr_abs for ev in evals]))
        mean_all_corr = float(np.mean([ev.all_corr_abs for ev in evals]))
        mean_oracle_align = float(np.mean([ev.oracle_align_abs for ev in evals]))
        oracle_ratios = []
        for log_name, log_results in all_results.items():
            if method_name not in log_results or log_name not in oracle_values:
                continue
            oracle_corr = oracle_values[log_name]
            if oracle_corr > 1e-9:
                oracle_ratios.append(log_results[method_name].active_corr_abs / oracle_corr)
        mean_oracle_ratio = float(np.mean(oracle_ratios)) if oracle_ratios else float("nan")

        curve_rmse_values = [ev.curve_all_rmse for ev in evals if ev.curve_all_rmse is not None]
        curve_corr_values = [ev.curve_all_corr for ev in evals if ev.curve_all_corr is not None]
        summary_rows.append(
            (
                mean_abs_corr,
                method_name,
                mean_all_corr,
                mean_oracle_align,
                mean_oracle_ratio,
                float(np.mean(curve_rmse_values)) if curve_rmse_values else None,
                float(np.mean(curve_corr_values)) if curve_corr_values else None,
            )
        )

    for _, method_name, mean_all_corr, mean_oracle_align, mean_oracle_ratio, curve_rmse, curve_corr in sorted(
        summary_rows, reverse=True
    ):
        curve_bits = ""
        if curve_rmse is not None and curve_corr is not None:
            curve_bits = f" mean_curve_rmse={curve_rmse:.3f} mean_curve_corr={curve_corr:.4f}"
        ratio_bits = ""
        if np.isfinite(mean_oracle_ratio):
            ratio_bits = f" oracle_pct={100.0 * mean_oracle_ratio:.1f}%"
        mean_abs_corr = next(row[0] for row in summary_rows if row[1] == method_name)
        print(
            f"  {method_name:22s}"
            f" mean_abs_corr={mean_abs_corr:.4f}"
            f" mean_all_corr={mean_all_corr:.4f}"
            f" mean_oracle_align={mean_oracle_align:.4f}"
            f"{ratio_bits}"
            f"{curve_bits}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate magnetometer projection-vector estimators that do not use travel for training, "
            "while using travel offline to score how well they perform"
        )
    )
    parser.add_argument(
        "logs",
        nargs="*",
        default=DEFAULT_LOGS,
        help="Log names without extension",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("backend/run_artifacts"),
        help="Root containing pipeline cache folders",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=sorted(METHOD_REGISTRY.keys()),
        help="Estimator methods to run",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-6,
        help="Small ridge penalty used in the supervised oracle fit",
    )
    parser.add_argument(
        "--pipeline-mag-threshold",
        type=float,
        default=3000.0,
        help="Magnitude threshold for the pipeline-style high-mag mean baseline, in mG",
    )
    parser.add_argument(
        "--still-len-s",
        type=float,
        default=0.1,
        help="Required quiet time before an accel-defined motion chunk",
    )
    parser.add_argument(
        "--bump-len-s",
        type=float,
        default=0.3,
        help="Length of the accel-defined motion chunk after the quiet window",
    )
    parser.add_argument(
        "--stride-s",
        type=float,
        default=0.05,
        help="Stride for searching accel-defined motion chunks",
    )
    parser.add_argument(
        "--still-a-max",
        type=float,
        default=1.0,
        help="Max allowed abs accel/proj during the quiet window, in m/s^2",
    )
    parser.add_argument(
        "--bump-dx-min-mm",
        type=float,
        default=20.0,
        help="Minimum double-integrated accel displacement proxy to accept a chunk, in mm",
    )
    parser.add_argument(
        "--skip-chunks",
        type=int,
        default=3,
        help="How many following search strides to skip after a chunk is accepted",
    )
    parser.add_argument(
        "--skip-curve-fit",
        action="store_true",
        help="Only report vector quality metrics and skip the eval-only travel curve fit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_results: dict[str, dict[str, MethodEval]] = {}
    for log_name in args.logs:
        data = load_log_data(log_name, args.cache_root)
        all_results[log_name] = summarize_log(data, args)
    print_summary(all_results, args.methods)


if __name__ == "__main__":
    main()
