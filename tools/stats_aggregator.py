from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.angle_corruption import project_mask_to_timeline

DEFAULT_LOGS = [
    "log022",
    "log029",
    "log030",
    "log031",
    "log038",
    "log056_ccdh",
    "log060_upperpred",
    "log078",
    "log079",
    "log080",
    "log085",
    "log088",
    "log091",
    "log096",
    "log098",
    "log099",
    "log103",
    "log104",
]

COMPARISONS = (
    ("travel/mag_model", "travel"),
    ("travel/mag_model/adj", "travel"),
    ("travel/solved", "travel"),
)


def flatten_1d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr.reshape(-1)


def load_cache(log_name: str, cache_root: Path) -> np.lib.npyio.NpzFile:
    cache_path = cache_root / log_name / "cache" / "all.npz"
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    return np.load(cache_path)


def infer_dt_seconds(time_s: np.ndarray) -> float:
    time_s = flatten_1d(time_s)
    if len(time_s) < 2:
        return 0.0
    diffs = np.diff(time_s)
    finite_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(finite_diffs) == 0:
        return 0.0
    return float(np.median(finite_diffs))


def get_error_stats(x: np.ndarray, gt: np.ndarray, center: bool = False, thresh: float | None = None) -> tuple[float, float, float]:
    x = flatten_1d(x)
    gt = flatten_1d(gt)
    if x.shape != gt.shape:
        raise ValueError(f"Error inputs must match in shape, got {x.shape} vs {gt.shape}")
    if thresh is not None:
        err_mask = np.abs(gt) > thresh
    if center:
        x = x - np.mean(x)
        gt = gt - np.mean(gt)
    err = x - gt
    if thresh is not None:
        err = err[err_mask]
    return (
        float(np.sqrt(np.mean(err**2))),
        float(np.mean(np.abs(err))),
        float(np.mean(err)),
    )


def get_error_vector(x: np.ndarray, gt: np.ndarray, center: bool = False) -> np.ndarray:
    x = flatten_1d(x)
    gt = flatten_1d(gt)
    if x.shape != gt.shape:
        raise ValueError(f"Error inputs must match in shape, got {x.shape} vs {gt.shape}")
    if center:
        x = x - np.mean(x)
        gt = gt - np.mean(gt)
    return x - gt


def build_mask(cache: np.lib.npyio.NpzFile, pred_key: str, gt_key: str, error_threshold: float | None = None) -> np.ndarray:
    boring_mask = np.asarray(cache["boring_mask"]).astype(bool).reshape(-1)
    pred = flatten_1d(cache[f"{pred_key}__x"])
    gt = flatten_1d(cache[f"{gt_key}__x"])
    finite_mask = np.isfinite(pred) & np.isfinite(gt)
    mask = boring_mask & finite_mask
    mask = mask & ~build_angle_bad_mask(cache, cache[f"{gt_key}__t"])
    if error_threshold is not None:
        mask = mask & (abs(gt) > error_threshold)
    if pred.shape != gt.shape or pred.shape != boring_mask.shape:
        raise ValueError(
            f"Cache arrays do not align for {pred_key} vs {gt_key}: "
            f"pred={pred.shape}, gt={gt.shape}, mask={boring_mask.shape}"
        )
    return mask


def build_angle_bad_mask(cache: np.lib.npyio.NpzFile, target_t: np.ndarray) -> np.ndarray:
    target_t = flatten_1d(target_t)
    if "angle/bad_mask__x" not in cache or "angle/bad_mask__t" not in cache:
        return np.zeros(len(target_t), dtype=bool)

    return project_mask_to_timeline(
        cache["angle/bad_mask__t"],
        flatten_1d(cache["angle/bad_mask__x"]).astype(bool),
        target_t,
    )


def summarize_log(log_name: str, cache_root: Path, center_errors: bool, error_threshold: float | None) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    cache = load_cache(log_name, cache_root)
    time_s = flatten_1d(cache["travel__t"])
    boring_mask = np.asarray(cache["boring_mask"]).astype(bool).reshape(-1)
    if time_s.shape != boring_mask.shape:
        raise ValueError(f"{log_name}: travel time and boring mask do not align")

    dt_s = infer_dt_seconds(time_s)
    total_seconds = len(time_s) * dt_s
    boring_seconds = int(np.sum(boring_mask)) * dt_s

    summary_row: dict[str, object] = {
        "log": log_name,
        "samples": len(time_s),
        "dt_ms": dt_s * 1000.0,
        "total_s": total_seconds,
        "boring_s": boring_seconds,
        "boring_pct": (100.0 * boring_seconds / total_seconds) if total_seconds > 0 else float("nan"),
    }

    error_rows: dict[str, dict[str, object]] = {}
    for pred_key, gt_key in COMPARISONS:
        pred = flatten_1d(cache[f"{pred_key}__x"])
        gt = flatten_1d(cache[f"{gt_key}__x"])
        mask = build_mask(cache, pred_key, gt_key)
        masked_pred = pred[mask]
        masked_gt = gt[mask]
        if len(masked_pred) == 0:
            raise ValueError(f"{log_name}: no finite boring-mask samples for {pred_key} vs {gt_key}")

        rmse, mae, me = get_error_stats(masked_pred, masked_gt, center=center_errors, thresh=error_threshold)
        error_rows[pred_key] = {
            "log": log_name,
            "n": int(len(masked_pred)),
            "rmse": rmse,
            "mae": mae,
            "me": me,
        }

    return summary_row, error_rows


def dense_index_mask(length: int, indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    idx = np.asarray(indices, dtype=int).reshape(-1)
    idx = idx[(idx >= 0) & (idx < length)]
    mask[idx] = True
    return mask


def make_masked_error(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, center_errors: bool) -> np.ndarray:
    masked_pred = flatten_1d(pred)[mask]
    masked_gt = flatten_1d(gt)[mask]
    return get_error_vector(masked_pred, masked_gt, center=center_errors)


def masked_rmse(err: np.ndarray, cond: np.ndarray) -> float:
    cond = np.asarray(cond, dtype=bool).reshape(-1)
    if cond.shape != err.shape or not np.any(cond):
        return float("nan")
    return float(np.sqrt(np.mean(err[cond] ** 2)))


def masked_ratio_pct(cond: np.ndarray) -> float:
    cond = np.asarray(cond, dtype=bool).reshape(-1)
    if len(cond) == 0:
        return float("nan")
    return 100.0 * float(np.mean(cond))


def maybe_percentile(values: np.ndarray, q: float) -> float:
    values = flatten_1d(values)
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(values, q))


def diagnostic_rows(
    log_name: str, cache_root: Path, center_errors: bool
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, np.ndarray]]:
    cache = load_cache(log_name, cache_root)

    boring_mask = np.asarray(cache["boring_mask"]).astype(bool).reshape(-1)
    travel = flatten_1d(cache["travel__x"])
    mag = flatten_1d(cache["mag/proj/corr/lpf__x"])
    accel_hp_abs = np.abs(flatten_1d(cache["accel/lpfhp/proj__x"]))
    bad_mag_mask = flatten_1d(cache["mag/proj/bad_mask__x"]).astype(bool)
    zv_mask = dense_index_mask(len(travel), cache["mag_zv_points"])
    mag_baseline = float(flatten_1d(cache["mag_baseline"])[0])
    mag_anchor_thresh = max(500.0, mag_baseline)

    if not (
        travel.shape == boring_mask.shape == mag.shape == accel_hp_abs.shape == bad_mag_mask.shape == zv_mask.shape
    ):
        raise ValueError(f"{log_name}: diagnostic arrays do not align")

    angle_bad_mask = build_angle_bad_mask(cache, cache["travel__t"])
    mask = boring_mask & np.isfinite(travel) & np.isfinite(mag) & np.isfinite(accel_hp_abs) & ~angle_bad_mask
    if not np.any(mask):
        raise ValueError(f"{log_name}: no finite diagnostic samples on boring_mask")

    masked_travel = travel[mask]
    masked_mag = mag[mask]
    masked_accel_hp_abs = accel_hp_abs[mask]
    masked_bad_mag = bad_mag_mask[mask]
    masked_zv = zv_mask[mask]

    mag_model_err = make_masked_error(cache["travel/mag_model__x"], travel, mask, center_errors)
    mag_adj_err = make_masked_error(cache["travel/mag_model/adj__x"], travel, mask, center_errors)
    solved_err = make_masked_error(cache["travel/solved__x"], travel, mask, center_errors)

    low_travel = masked_travel < 30
    high_travel = masked_travel > 100 # maybe_percentile(masked_travel, 80.0)
    high_accel = masked_accel_hp_abs > maybe_percentile(masked_accel_hp_abs, 80.0)
    low_mag = masked_mag < maybe_percentile(masked_mag, 20.0)
    high_mag = masked_mag > maybe_percentile(masked_mag, 80.0)
    anchor_on = masked_mag > mag_anchor_thresh
    anchor_off = ~anchor_on
    anchor_abs = np.abs(masked_mag) > mag_anchor_thresh

    stage_row = {
        "log": log_name,
        "n": int(np.sum(mask)),
        "mag_model_rmse": masked_rmse(mag_model_err, np.ones_like(masked_travel, dtype=bool)),
        "mag_adj_rmse": masked_rmse(mag_adj_err, np.ones_like(masked_travel, dtype=bool)),
        "solved_rmse": masked_rmse(solved_err, np.ones_like(masked_travel, dtype=bool)),
        "solver_delta": masked_rmse(solved_err, np.ones_like(masked_travel, dtype=bool)) - masked_rmse(mag_adj_err, np.ones_like(masked_travel, dtype=bool)),
        "mag_anchor_pct": 100.0 * float(np.mean(anchor_on)),
        "mag_abs_anchor_pct": 100.0 * float(np.mean(anchor_abs)),
        "bad_mag_pct": 100.0 * float(np.mean(masked_bad_mag)),
        "zvs_per_1k": 1000.0 * float(np.mean(masked_zv)),
    }

    condition_row = {
        "log": log_name,
        "low_trav": masked_rmse(solved_err, low_travel),
        "high_trav": masked_rmse(solved_err, high_travel),
        "high_acc": masked_rmse(solved_err, high_accel),
        "low_mag": masked_rmse(solved_err, low_mag),
        "high_mag": masked_rmse(solved_err, high_mag),
        "bad_mag": masked_rmse(solved_err, masked_bad_mag),
        "zv": masked_rmse(solved_err, masked_zv),
    }

    condition_ratio_row = {
        "log": log_name,
        "low_trav": masked_ratio_pct(low_travel),
        "high_trav": masked_ratio_pct(high_travel),
        "high_acc": masked_ratio_pct(high_accel),
        "low_mag": masked_ratio_pct(low_mag),
        "high_mag": masked_ratio_pct(high_mag),
        "bad_mag": masked_ratio_pct(masked_bad_mag),
        "zv": masked_ratio_pct(masked_zv),
    }

    solver_delta_row = {
        "log": log_name,
        "all_d": stage_row["solver_delta"],
        "low_trav_d": masked_rmse(solved_err, low_travel) - masked_rmse(mag_adj_err, low_travel),
        "high_trav_d": masked_rmse(solved_err, high_travel) - masked_rmse(mag_adj_err, high_travel),
        "anchor_off_d": masked_rmse(solved_err, anchor_off) - masked_rmse(mag_adj_err, anchor_off),
        "anchor_on_d": masked_rmse(solved_err, anchor_on) - masked_rmse(mag_adj_err, anchor_on),
        "bad_mag_d": masked_rmse(solved_err, masked_bad_mag) - masked_rmse(mag_adj_err, masked_bad_mag),
        "zv_d": masked_rmse(solved_err, masked_zv) - masked_rmse(mag_adj_err, masked_zv),
    }

    pooled = {
        "travel": masked_travel,
        "mag": masked_mag,
        "accel_hp_abs": masked_accel_hp_abs,
        "dmag_abs": np.abs(np.gradient(mag))[mask],
        "dtravel_abs": np.abs(np.gradient(travel))[mask],
        "solved_abs_err": np.abs(solved_err),
    }
    return stage_row, condition_row, condition_ratio_row, solver_delta_row, pooled


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = flatten_1d(x)
    y = flatten_1d(y)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(finite_mask) < 2:
        return float("nan")
    x = x[finite_mask]
    y = y[finite_mask]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def format_value(value: object) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "nan"
        return f"{float(value):.2f}"
    return str(value)


def print_table(title: str, columns: list[tuple[str, str]], rows: Iterable[dict[str, object]], sort_key: str = "n") -> None:
    rows = list(rows)
    if rows and sort_key not in rows[0]:
        sort_key = "log" if "log" in rows[0] else columns[0][0]
    rows.sort(key=lambda row: row[sort_key] if sort_key in row else row.get("log", ""))
    formatted_rows = [{key: format_value(row.get(key, "")) for key, _ in columns} for row in rows]
    widths = []
    for key, label in columns:
        content_width = max((len(row[key]) for row in formatted_rows), default=0)
        widths.append(max(len(label), content_width))

    print(title)
    header = "  ".join(label.ljust(width) if key == "log" else label.rjust(width) for (key, label), width in zip(columns, widths))
    divider = "  ".join("-" * width for width in widths)
    print(header)
    print(divider)
    for row in formatted_rows:
        line = "  ".join(
            row[key].ljust(width) if key == "log" else row[key].rjust(width)
            for (key, _), width in zip(columns, widths)
        )
        print(line)
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize pipeline cache durations and error statistics.")
    parser.add_argument(
        "logs",
        nargs="*",
        default=DEFAULT_LOGS,
        help="Log names to summarize. Defaults to the logs used by refine_mag_proj.py.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("backend/run_artifacts"),
        help="Root containing pipeline cache folders.",
    )
    parser.add_argument(
        "--center-errors",
        action="store_true",
        help="Subtract each masked signal mean before computing error stats.",
    )
    parser.add_argument(
        "--error-threshold",
        type=float,
        help="Minimum GT travel for error",
    )
    parser.add_argument(
        "--sort-key",
        type=str,
        default="log"
    )
    parser.add_argument(
        "--deep-dive",
        action="store_true",
        help="Print stage and feature-conditioned diagnostics for the selected logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary_rows: list[dict[str, object]] = []
    error_rows: dict[str, list[dict[str, object]]] = {pred_key: [] for pred_key, _ in COMPARISONS}
    diagnostic_stage_rows: list[dict[str, object]] = []
    diagnostic_condition_rows: list[dict[str, object]] = []
    diagnostic_condition_ratio_rows: list[dict[str, object]] = []
    diagnostic_delta_rows: list[dict[str, object]] = []
    pooled_features: dict[str, list[np.ndarray]] = {
        "travel": [],
        "mag": [],
        "accel_hp_abs": [],
        "dmag_abs": [],
        "dtravel_abs": [],
        "solved_abs_err": [],
    }
    failures: list[tuple[str, Exception]] = []

    for log_name in args.logs:
        try:
            summary_row, per_comparison_rows = summarize_log(log_name, args.cache_root, center_errors=args.center_errors, error_threshold=args.error_threshold)
        except Exception as exc:  # Keep the report useful even if a cache is missing or malformed.
            failures.append((log_name, exc))
            continue

        summary_rows.append(summary_row)
        for pred_key, _ in COMPARISONS:
            error_rows[pred_key].append(per_comparison_rows[pred_key])
        if args.deep_dive:
            try:
                stage_row, condition_row, condition_ratio_row, delta_row, pooled = diagnostic_rows(
                    log_name, args.cache_root, args.center_errors
                )
            except Exception as exc:  # Keep the main stats usable even if diagnostics fail.
                failures.append((f"{log_name} (diagnostics)", exc))
            else:
                diagnostic_stage_rows.append(stage_row)
                diagnostic_condition_rows.append(condition_row)
                diagnostic_condition_ratio_rows.append(condition_ratio_row)
                diagnostic_delta_rows.append(delta_row)
                for key, values in pooled.items():
                    pooled_features[key].append(values)

    if not summary_rows:
        raise SystemExit("No logs could be summarized.")

    print_table(
        title="Cache duration summary",
        columns=[
            ("log", "log"),
            ("samples", "samples"),
            ("dt_ms", "dt_ms"),
            ("total_s", "total_s"),
            ("boring_s", "boring_s"),
            ("boring_pct", "boring_%"),
        ],
        rows=summary_rows,
    )

    center_label = "centered" if args.center_errors else "raw"
    for pred_key, gt_key in COMPARISONS:
        print_table(
            title=f"Error stats on boring_mask ({center_label}): {pred_key} vs {gt_key}",
            columns=[
                ("log", "log"),
                ("n", "n"),
                ("rmse", "rmse"),
                ("mae", "mae"),
                ("me", "me"),
            ],
            rows=error_rows[pred_key],
            sort_key=args.sort_key
        )

    if failures:
        print("Skipped logs")
        for log_name, exc in failures:
            print(f"{log_name:>12}  {exc}")

    if args.deep_dive and diagnostic_stage_rows:
        print_table(
            title=f"Stage RMSE summary on boring_mask ({center_label})",
            columns=[
                ("log", "log"),
                ("n", "n"),
                ("mag_model_rmse", "mag_rmse"),
                ("mag_adj_rmse", "mag_adj"),
                ("solved_rmse", "solved"),
                ("solver_delta", "solv-mag"),
                ("mag_anchor_pct", "anchor_%"),
                ("mag_abs_anchor_pct", "|mag|_%"),
                ("bad_mag_pct", "badmag_%"),
                ("zvs_per_1k", "zv/1k"),
            ],
            rows=diagnostic_stage_rows,
            sort_key=args.sort_key,
        )

        print_table(
            title=f"Conditioned solved RMSE on boring_mask ({center_label})",
            columns=[
                ("log", "log"),
                ("low_trav", "trav<20"),
                ("high_trav", "trav>p80"),
                ("high_acc", "acc>p80"),
                ("low_mag", "mag<p20"),
                ("high_mag", "mag>p80"),
                ("bad_mag", "bad_mag"),
                ("zv", "zv"),
            ],
            rows=diagnostic_condition_rows,
            sort_key=args.sort_key,
        )

        print_table(
            title=f"Condition occurrence on boring_mask ({center_label}, % of diagnostic samples)",
            columns=[
                ("log", "log"),
                ("low_trav", "trav<20"),
                ("high_trav", "trav>p80"),
                ("high_acc", "acc>p80"),
                ("low_mag", "mag<p20"),
                ("high_mag", "mag>p80"),
                ("bad_mag", "bad_mag"),
                ("zv", "zv"),
            ],
            rows=diagnostic_condition_ratio_rows,
            sort_key=args.sort_key,
        )

        print_table(
            title=f"Solver delta vs mag_adj RMSE on boring_mask ({center_label}, negative is better)",
            columns=[
                ("log", "log"),
                ("all_d", "all"),
                ("low_trav_d", "trav<20"),
                ("high_trav_d", "trav>p80"),
                ("anchor_off_d", "anchor_off"),
                ("anchor_on_d", "anchor_on"),
                ("bad_mag_d", "bad_mag"),
                ("zv_d", "zv"),
            ],
            rows=diagnostic_delta_rows,
            sort_key=args.sort_key,
        )

        pooled_rows = []
        solved_abs_err = np.concatenate(pooled_features["solved_abs_err"])
        for key, label in [
            ("travel", "travel"),
            ("mag", "mag"),
            ("accel_hp_abs", "|accel_hp|"),
            ("dmag_abs", "|dmag|"),
            ("dtravel_abs", "|dtravel|"),
        ]:
            pooled_rows.append(
                {
                    "feature": label,
                    "corr": corrcoef_safe(np.concatenate(pooled_features[key]), solved_abs_err),
                }
            )
        print_table(
            title=f"Pooled correlation with |travel/solved error| on boring_mask ({center_label})",
            columns=[
                ("feature", "feature"),
                ("corr", "corr"),
            ],
            rows=pooled_rows,
            sort_key="feature",
        )


if __name__ == "__main__":
    main()
