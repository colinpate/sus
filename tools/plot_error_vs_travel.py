#!/usr/bin/env python3
"""Plot recording-balanced travel error as a function of reference travel."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep matplotlib's font cache out of the user's home directory.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sus-matplotlib-cache"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from backend.log_registry import DEFAULT_REGISTRY_PATH, ResolvedLog, load_registry  # noqa: E402
from tools.stats_aggregator import build_mask, flatten_1d, load_cache  # noqa: E402


METRIC_LABELS = {
    "signed-median": "Median signed error (mm)",
    "mae": "Mean absolute error (mm)",
    "rmse": "RMSE (mm)",
    "p95-absolute": "95th-percentile absolute error (mm)",
}


def setup_key(log: ResolvedLog, group_by: str) -> tuple[str, ...]:
    position = str(log.metadata.get("position", log.pipeline)).title()
    bike = str(log.metadata.get("bike_model", "Unknown bike"))
    if group_by == "bike":
        return position, bike
    pod = str(log.metadata.get("pod_version", "?"))
    return position, bike, f"pod v{pod}"


def setup_title(key: tuple[str, ...]) -> str:
    return " · ".join(key)


def independent_unit(log: ResolvedLog) -> str:
    """Collapse derived child logs to their parent recording."""
    return str(log.metadata.get("parent_log", log.log_id))


def aggregation_unit(log: ResolvedLog, *, separate_segments: bool) -> str:
    """Choose whether split segments count separately or as one recording."""
    return log.log_id if separate_segments else independent_unit(log)


def metric_value(error: np.ndarray, metric: str) -> float:
    if metric == "signed-median":
        return float(np.median(error))
    if metric == "mae":
        return float(np.mean(np.abs(error)))
    if metric == "rmse":
        return float(np.sqrt(np.mean(error**2)))
    if metric == "p95-absolute":
        return float(np.percentile(np.abs(error), 95))
    raise ValueError(f"Unknown metric {metric!r}")


def per_log_bins(
    log: ResolvedLog,
    *,
    cache_root: Path,
    prediction: str,
    edges: np.ndarray,
    metric: str,
    centered: bool,
    min_samples: int,
    mag: bool,
) -> np.ndarray:
    with load_cache(log.log_id, cache_root) as cache:
        prediction_key = f"{prediction}__x"
        if prediction_key not in cache:
            raise KeyError(f"{log.log_id}: cache is missing {prediction}")
        predicted = flatten_1d(cache[prediction_key])
        reference = flatten_1d(cache["travel__x"])
        if mag:
            x_axis = flatten_1d(cache["mag/norm/lpf__x"])
        else:
            x_axis = reference
        mask = build_mask(cache, prediction, "travel")

    error = predicted - reference
    if centered:
        # One offset for the whole scored recording, never one offset per bin.
        error = error - float(np.mean(error[mask]))

    values = np.full(len(edges) - 1, np.nan)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = mask & (x_axis >= lower)
        in_bin &= x_axis <= upper if index == len(values) - 1 else x_axis < upper
        if int(np.sum(in_bin)) >= min_samples:
            values[index] = metric_value(error[in_bin], metric)
    return values


def collapse_children(rows: Iterable[tuple[str, np.ndarray]]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {}
    for unit, values in rows:
        grouped.setdefault(unit, []).append(values)
    collapsed: dict[str, np.ndarray] = {}
    for unit, children in grouped.items():
        child_values = np.vstack(children)
        values = np.full(child_values.shape[1], np.nan)
        occupied = np.any(np.isfinite(child_values), axis=0)
        values[occupied] = np.nanmedian(child_values[:, occupied], axis=0)
        collapsed[unit] = values
    return collapsed


def column_percentile(values: np.ndarray, percentile: float) -> np.ndarray:
    result = np.full(values.shape[1], np.nan)
    occupied = np.any(np.isfinite(values), axis=0)
    result[occupied] = np.nanpercentile(values[:, occupied], percentile, axis=0)
    return result


def write_values_csv(
    path: Path,
    setup_units: dict[tuple[str, ...], dict[str, np.ndarray]],
    edges: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("setup", "independent_unit", "bin_start_mm", "bin_stop_mm", "value_mm"))
        for setup, units in setup_units.items():
            for unit, values in units.items():
                for lower, upper, value in zip(edges[:-1], edges[1:], values):
                    if np.isfinite(value):
                        writer.writerow((setup_title(setup), unit, lower, upper, value))


def plot_setups(
    setup_units: dict[tuple[str, ...], dict[str, np.ndarray]],
    *,
    edges: np.ndarray,
    metric: str,
    centered: bool,
    min_units: int,
    show_individuals: bool,
    output: Path,
    mag: bool,
) -> None:
    setups = sorted(setup_units, key=lambda key: (key[0] == "Rear", key))
    centers = (edges[:-1] + edges[1:]) / 2.0
    columns = 2
    rows = math.ceil(len(setups) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12.0, 3.6 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for axis, setup in zip(axes.flat, setups):
        unit_map = setup_units[setup]
        unit_values = np.vstack(list(unit_map.values()))
        counts = np.sum(np.isfinite(unit_values), axis=0)
        median = column_percentile(unit_values, 50)
        q25 = column_percentile(unit_values, 25)
        q75 = column_percentile(unit_values, 75)
        supported = counts >= min_units
        median[~supported] = np.nan
        q25[~supported] = np.nan
        q75[~supported] = np.nan

        if show_individuals:
            for values in unit_values:
                axis.plot(centers, values, color="#8c8c8c", alpha=0.28, linewidth=0.8)
        axis.fill_between(centers, q25, q75, color="#4477aa", alpha=0.20, linewidth=0)
        axis.plot(centers, median, color="#275d91", marker="o", markersize=3.5, linewidth=2.0)
        if metric == "signed-median":
            axis.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.65)
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.set_title(f"{setup_title(setup)}\n{len(unit_map)} independent recordings", fontsize=10)

        # Sample support is placed at the bottom of each panel, independent of y scale.
        for x, count, ok in zip(centers, counts, supported):
            if ok:
                axis.text(
                    x,
                    0.015,
                    str(int(count)),
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    color="#666666",
                    fontsize=6.5,
                )
        axis.text(
            0.006,
            0.015,
            "n:",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            color="#666666",
            fontsize=6.5,
        )

    for axis in axes.flat[len(setups):]:
        axis.remove()
    for axis in axes[-1, :]:
        if axis in fig.axes:
            if mag:
                axis.set_xlabel("Magnetometer norm (mG)")
            else:
                axis.set_xlabel("Reference travel (mm)")
    for axis in axes[:, 0]:
        if axis in fig.axes:
            axis.set_ylabel(METRIC_LABELS[metric])

    alignment = "offset-aligned" if centered else "absolute (uncentered)"
    fig.suptitle(f"Travel-dependent error: {alignment}", fontsize=14)
    legend = [
        Line2D([0], [0], color="#275d91", marker="o", markersize=4, linewidth=2, label="Setup median"),
        Line2D([0], [0], color="#4477aa", linewidth=7, alpha=0.20, label="Recording IQR"),
    ]
    if show_individuals:
        legend.append(Line2D([0], [0], color="#8c8c8c", alpha=0.5, linewidth=1, label="Independent recording"))
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=len(legend), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sets",
        nargs="+",
        default=("front-default",),
        help="Registry sets to include (default: front-default).",
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "backend" / "run_artifacts")
    parser.add_argument("--prediction", default="travel/solved")
    parser.add_argument("--metric", choices=tuple(METRIC_LABELS), default="signed-median")
    parser.add_argument("--centering", choices=("aligned", "absolute"), default="aligned")
    parser.add_argument("--bin-width", type=float, default=20.0)
    parser.add_argument("--min-samples", type=int, default=40)
    parser.add_argument("--min-units", type=int, default=3)
    parser.add_argument(
        "--group-by",
        choices=("bike", "bike-pod"),
        default="bike-pod",
        help="Whether pod generations form separate setup panels (default: bike-pod).",
    )
    parser.add_argument("--no-individuals", action="store_true")
    parser.add_argument(
        "--separate-segments",
        action="store_true",
        help=(
            "Treat split child-log segments as independent units instead of "
            "collapsing them to their parent recording."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reports" / "error_vs_travel" / "error_vs_travel.png",
    )
    parser.add_argument(
        "--mag",
        action="store_true"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bin_width <= 0:
        raise ValueError("--bin-width-mm must be positive")
    if args.min_samples <= 0 or args.min_units <= 0:
        raise ValueError("--min-samples and --min-units must be positive")

    registry = load_registry(args.registry)
    selected: dict[str, ResolvedLog] = {}
    for set_name in args.sets:
        for log in registry.select(set_name=set_name, usable_only=True):
            selected[log.log_id] = log
    if not selected:
        raise ValueError("No usable logs matched the requested registry sets")

    strokes = [
        float(log.metadata.get(f"{log.pipeline}_travel_mm", 0.0))
        for log in selected.values()
    ]
    if args.mag:
        max_stroke = 30000 # mG
    else:
        max_stroke = max(strokes)
    if max_stroke <= 0:
        raise ValueError("Selected logs do not define front_travel_mm or rear_travel_mm")
    stop = math.ceil(max_stroke / args.bin_width) * args.bin_width
    edges = np.arange(0.0, stop + args.bin_width * 0.5, args.bin_width)

    raw_rows: dict[tuple[str, ...], list[tuple[str, np.ndarray]]] = {}
    skipped: list[str] = []
    for log in sorted(selected.values(), key=lambda item: item.log_id):
        try:
            values = per_log_bins(
                log,
                cache_root=args.cache_root,
                prediction=args.prediction,
                edges=edges,
                metric=args.metric,
                centered=args.centering == "aligned",
                min_samples=args.min_samples,
                mag=args.mag
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            skipped.append(f"{log.log_id}: {exc}")
            continue
        raw_rows.setdefault(setup_key(log, args.group_by), []).append(
            (
                aggregation_unit(
                    log, separate_segments=args.separate_segments
                ),
                values,
            )
        )

    setup_units = {setup: collapse_children(rows) for setup, rows in raw_rows.items()}
    setup_units = {setup: units for setup, units in setup_units.items() if units}
    if not setup_units:
        raise ValueError("None of the selected logs had usable cached predictions")

    plot_setups(
        setup_units,
        edges=edges,
        metric=args.metric,
        centered=args.centering == "aligned",
        min_units=args.min_units,
        show_individuals=not args.no_individuals,
        output=args.output,
        mag=args.mag,
    )
    write_values_csv(args.output.with_suffix(".csv"), setup_units, edges)

    units = sum(len(values) for values in setup_units.values())
    print(f"Wrote {args.output}, {args.output.with_suffix('.pdf')}, and {args.output.with_suffix('.csv')}")
    print(f"Included {units} independent recordings in {len(setup_units)} setup panels")
    if skipped:
        print(f"Skipped {len(skipped)} logs:", file=sys.stderr)
        for message in skipped:
            print(f"  {message}", file=sys.stderr)


if __name__ == "__main__":
    main()
