from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_LOGS = [
    "log022",
    "log029",
    "log030",
    "log031",
    "log038",
    "log056_ccdh",
    "log078",
    "log079",
    "log080",
    "log085",
    "log086",
    "log088",
    "log091",
]

COMPARISONS = (
    ("travel/mag_model", "travel"),
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


def get_error_stats(x: np.ndarray, gt: np.ndarray, center: bool = False) -> tuple[float, float, float]:
    x = flatten_1d(x)
    gt = flatten_1d(gt)
    if x.shape != gt.shape:
        raise ValueError(f"Error inputs must match in shape, got {x.shape} vs {gt.shape}")
    if center:
        x = x - np.mean(x)
        gt = gt - np.mean(gt)
    err = x - gt
    return (
        float(np.sqrt(np.mean(err**2))),
        float(np.mean(np.abs(err))),
        float(np.mean(err)),
    )


def build_mask(cache: np.lib.npyio.NpzFile, pred_key: str, gt_key: str, error_threshold: float | None) -> np.ndarray:
    boring_mask = np.asarray(cache["boring_mask"]).astype(bool).reshape(-1)
    pred = flatten_1d(cache[f"{pred_key}__x"])
    gt = flatten_1d(cache[f"{gt_key}__x"])
    finite_mask = np.isfinite(pred) & np.isfinite(gt)
    mask = boring_mask & finite_mask
    if error_threshold is not None:
        mask = mask & (abs(gt) > error_threshold)
    if pred.shape != gt.shape or pred.shape != boring_mask.shape:
        raise ValueError(
            f"Cache arrays do not align for {pred_key} vs {gt_key}: "
            f"pred={pred.shape}, gt={gt.shape}, mask={boring_mask.shape}"
        )
    return mask


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
        mask = build_mask(cache, pred_key, gt_key, error_threshold)
        masked_pred = pred[mask]
        masked_gt = gt[mask]
        if len(masked_pred) == 0:
            raise ValueError(f"{log_name}: no finite boring-mask samples for {pred_key} vs {gt_key}")

        rmse, mae, me = get_error_stats(masked_pred, masked_gt, center=center_errors)
        error_rows[pred_key] = {
            "log": log_name,
            "n": int(np.sum(mask)),
            "rmse": rmse,
            "mae": mae,
            "me": me,
        }

    return summary_row, error_rows


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
    rows.sort(key = lambda x: x.get(sort_key, x["log"]))
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary_rows: list[dict[str, object]] = []
    error_rows: dict[str, list[dict[str, object]]] = {pred_key: [] for pred_key, _ in COMPARISONS}
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


if __name__ == "__main__":
    main()
