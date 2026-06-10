#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_SOLVED_KEY = "travel/solved"
DEFAULT_GROUND_TRUTH_KEY = "travel"


@dataclass(frozen=True)
class SeriesExport:
    label: str
    key: str
    time_s: np.ndarray
    value_mm: np.ndarray


@dataclass(frozen=True)
class RearLinkageCurve:
    wheel_mm: np.ndarray
    shock_mm: np.ndarray

    @property
    def max_wheel_mm(self) -> float:
        return float(self.wheel_mm[-1])

    @property
    def max_shock_mm(self) -> float:
        return float(self.shock_mm[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export solved and angle-derived travel from a cached pipeline run into "
            "semicolon-delimited CSV files that SST can import."
        )
    )
    parser.add_argument(
        "logs",
        nargs="+",
        help=(
            "Log names whose cache lives at backend/run_artifacts/<log>/cache/all.npz. "
            "Example: log112"
        ),
    )
    parser.add_argument(
        "--max-travel-mm",
        type=float,
        required=True,
        help=(
            "Physical max wheel travel used to normalize the exported data when "
            "no linkage CSV is provided."
        ),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("backend/run_artifacts"),
        help="Root directory that contains per-log cache folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for CSV output. By default each log writes into "
            "backend/run_artifacts/<log>/sst."
        ),
    )
    parser.add_argument(
        "--channel",
        choices=("shock", "fork"),
        default="shock",
        help="Which SST column to emit.",
    )
    parser.add_argument(
        "--scale",
        choices=("fraction", "percent"),
        default="fraction",
        help="Whether to export normalized values in 0..1 or 0..100.",
    )
    parser.add_argument(
        "--solved-key",
        default=DEFAULT_SOLVED_KEY,
        help="Cache time-series key for the solved travel export.",
    )
    parser.add_argument(
        "--ground-truth-key",
        default=DEFAULT_GROUND_TRUTH_KEY,
        help="Cache time-series key for the angle-derived ground-truth export.",
    )
    parser.add_argument(
        "--linkage-csv",
        type=Path,
        default=None,
        help=(
            "Optional SST linkage CSV (semicolon-delimited Wheel_T plus either "
            "Leverage_R or Shock_T). When set with --channel shock, the script "
            "converts wheel-travel mm into normalized shock stroke before export."
        ),
    )
    return parser.parse_args()


def load_series(cache: np.lib.npyio.NpzFile, key: str) -> tuple[np.ndarray, np.ndarray]:
    t_key = f"{key}__t"
    x_key = f"{key}__x"
    if t_key not in cache or x_key not in cache:
        raise KeyError(f"Missing time-series key '{key}' in cache")

    time_s = np.asarray(cache[t_key], dtype=float).reshape(-1)
    value = np.asarray(cache[x_key], dtype=float)
    if value.ndim == 2:
        if value.shape[1] != 1:
            raise ValueError(f"Expected 1D series for '{key}', got shape {value.shape}")
        value = value[:, 0]
    elif value.ndim != 1:
        raise ValueError(f"Expected 1D series for '{key}', got shape {value.shape}")

    if time_s.shape[0] != value.shape[0]:
        raise ValueError(
            f"Time/value length mismatch for '{key}': {time_s.shape[0]} vs {value.shape[0]}"
        )
    return time_s, value


def load_rear_linkage_curve(path: Path) -> RearLinkageCurve:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        fieldnames = set(reader.fieldnames or [])
        if "Wheel_T" not in fieldnames:
            raise ValueError(f"{path} is missing the required 'Wheel_T' column")

        wheel_mm: list[float] = []
        shock_mm: list[float] = []

        if "Shock_T" in fieldnames:
            for row in reader:
                wheel_mm.append(float(row["Wheel_T"]))
                shock_mm.append(float(row["Shock_T"]))
        elif "Leverage_R" in fieldnames:
            shock = 0.0
            for row in reader:
                wheel_mm.append(float(row["Wheel_T"]))
                shock_mm.append(shock)
                # Mirror SST's own linkage import logic so exported fractions
                # round-trip through the GUI the same way.
                leverage = float(row["Leverage_R"])
                shock += 1.0 / leverage
        else:
            raise ValueError(
                f"{path} must contain either 'Shock_T' or 'Leverage_R' alongside 'Wheel_T'"
            )

    if not wheel_mm or not shock_mm:
        raise ValueError(f"{path} does not contain any linkage rows")

    wheel = np.asarray(wheel_mm, dtype=float)
    shock = np.asarray(shock_mm, dtype=float)
    order = np.argsort(wheel, kind="stable")
    wheel = wheel[order]
    shock = shock[order]

    keep = np.concatenate(([True], np.diff(wheel) > 0))
    wheel = wheel[keep]
    shock = shock[keep]

    if wheel.shape[0] < 2:
        raise ValueError(f"{path} must contain at least two distinct Wheel_T samples")
    if shock[-1] <= 0:
        raise ValueError(f"{path} does not define a positive max shock stroke")

    return RearLinkageCurve(wheel_mm=wheel, shock_mm=shock)


def normalize_direct(travel_mm: np.ndarray, max_travel_mm: float) -> np.ndarray:
    if max_travel_mm <= 0:
        raise ValueError("--max-travel-mm must be positive")
    return travel_mm / max_travel_mm


def normalize_with_rear_linkage(travel_mm: np.ndarray, linkage: RearLinkageCurve) -> np.ndarray:
    wheel_clipped = np.clip(travel_mm, linkage.wheel_mm[0], linkage.max_wheel_mm)
    shock_mm = np.interp(wheel_clipped, linkage.wheel_mm, linkage.shock_mm)
    return shock_mm / linkage.max_shock_mm


def to_sst_normalized(
    travel_mm: np.ndarray,
    *,
    max_travel_mm: float,
    channel: str,
    linkage: RearLinkageCurve | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if linkage is not None and channel == "shock":
        raw_normalized = normalize_with_rear_linkage(travel_mm, linkage)
        mode = (
            "rear shock-stroke fraction reconstructed from linkage CSV "
            f"(wheel max {linkage.max_wheel_mm:.3f} mm, shock max {linkage.max_shock_mm:.3f} mm)"
        )
    else:
        # This matches the SST wiki wording ("travel percentage"), and it is
        # also exact for fork data because fork travel is linear in stroke.
        raw_normalized = normalize_direct(travel_mm, max_travel_mm)
        mode = f"direct wheel-travel fraction using max travel {max_travel_mm:.3f} mm"

    return raw_normalized, np.clip(raw_normalized, 0.0, 1.0), mode


def scale_values(normalized: np.ndarray, scale: str) -> np.ndarray:
    if scale == "percent":
        return normalized * 100.0
    return normalized


def output_dir_for_log(log_name: str, args: argparse.Namespace) -> Path:
    if args.out_dir is not None:
        return args.out_dir
    return args.cache_root / log_name / "sst"


def column_name_for_channel(channel: str) -> str:
    return "Shock" if channel == "shock" else "Fork"


def write_sst_csv(
    path: Path,
    *,
    time_s: np.ndarray,
    values: np.ndarray,
    channel: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    column_name = column_name_for_channel(channel)
    elapsed_s = time_s - float(time_s[0])

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Time", column_name])
        for t, value in zip(elapsed_s, values, strict=True):
            writer.writerow([f"{t:.6f}", f"{value:.6f}"])


def cache_path_for_log(cache_root: Path, log_name: str) -> Path:
    return cache_root / log_name / "cache" / "all.npz"


def export_log(
    log_name: str,
    *,
    args: argparse.Namespace,
    linkage: RearLinkageCurve | None,
) -> None:
    cache_path = cache_path_for_log(args.cache_root, log_name)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found for {log_name}: {cache_path}")

    with np.load(cache_path, allow_pickle=False) as cache:
        exports = []
        for label, key in (
            ("solved", args.solved_key),
            ("ground_truth", args.ground_truth_key),
        ):
            time_s, value_mm = load_series(cache, key)
            exports.append(SeriesExport(label=label, key=key, time_s=time_s, value_mm=value_mm))

    target_dir = output_dir_for_log(log_name, args)
    for export in exports:
        raw_normalized, normalized, mode = to_sst_normalized(
            export.value_mm,
            max_travel_mm=args.max_travel_mm,
            channel=args.channel,
            linkage=linkage,
        )
        scaled = scale_values(normalized, args.scale)
        out_path = target_dir / f"{log_name}_{export.label}_sst.csv"
        write_sst_csv(
            out_path,
            time_s=export.time_s,
            values=scaled,
            channel=args.channel,
        )

        low_count = int(np.sum(raw_normalized < 0.0))
        high_count = int(np.sum(raw_normalized > 1.0))
        print(
            f"{log_name} {export.label}: wrote {out_path} from cache key '{export.key}' "
            f"using {mode}. clipped_low={low_count} clipped_high={high_count}"
        )


def main() -> None:
    args = parse_args()
    linkage = load_rear_linkage_curve(args.linkage_csv) if args.linkage_csv else None

    for log_name in args.logs:
        export_log(log_name, args=args, linkage=linkage)


if __name__ == "__main__":
    main()
