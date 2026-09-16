#!/usr/bin/env python3
"""Render the summary figures for the magnetic reference-point sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#35618d"
ORANGE = "#d17a22"
GREEN = "#4e8067"
GRAY = "#777777"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def balanced(rows: list[dict[str, str]], split: str) -> dict[str, dict[str, str]]:
    return {
        row["config"]: row
        for row in rows
        if row["split"] == split and row.get("scope") == "cohort-balanced"
    }


def value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.22,
        }
    )


def plot_anchor_travel(output_dir: Path) -> None:
    rows = read_rows(output_dir / "oracle_anchor_travel_aggregate.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for split, color, label in (
        ("tuning", BLUE, "front-default (tuning)"),
        ("validation", ORANGE, "slayer-filtered (held out)"),
    ):
        selected = sorted(
            (row for row in rows if row["split"] == split),
            key=lambda row: value(row, "target_travel_mm"),
        )
        x = [value(row, "target_travel_mm") for row in selected]
        axes[0].plot(
            x,
            [value(row, "uncentered_rmse_mm") for row in selected],
            marker="o",
            color=color,
            label=label,
        )
        axes[1].plot(
            x,
            [value(row, "abs_offset_error_mm") for row in selected],
            marker="o",
            color=color,
        )
        axes[2].plot(
            x,
            [value(row, "local_sensitivity_mm_per_mg") for row in selected],
            marker="o",
            color=color,
        )
    axes[0].set_ylabel("Adjusted-travel RMSE (mm)")
    axes[1].set_ylabel("Absolute offset error (mm)")
    axes[2].set_ylabel("Local sensitivity (mm/mG)")
    for axis in axes:
        axis.set_xlabel("Oracle reference travel (mm)")
        axis.set_xticks((5, 25, 50, 75, 100, 125, 150))
        axis.tick_params(axis="x", rotation=45)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Where a correct reference point is most informative")
    fig.tight_layout()
    fig.savefig(output_dir / "anchor_travel_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def plot_key_parameters(output_dir: Path) -> None:
    tuning = balanced(read_rows(output_dir / "screen_aggregate.csv"), "tuning")
    validation = balanced(
        read_rows(output_dir / "diagnostic_validation_screen_aggregate.csv"),
        "validation",
    )
    panels = (
        ("bump_len_s", (0.2, 0.3, 0.4, 0.5, 0.75, 1.0), "Bump window (s)"),
        (
            "bump_mag_min_mg",
            (50.0, 100.0, 200.0, 300.0, 500.0, 750.0, 1000.0, 1500.0),
            "Minimum magnetic bump (mG)",
        ),
        (
            "still_a_max_mm_s2",
            (250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0),
            "Still-acceleration limit (mm/s²)",
        ),
        (
            "bump_dx_min_mm",
            (5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0),
            "Minimum integrated displacement (mm)",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, (field, levels, xlabel) in zip(axes.ravel(), panels):
        names = [f"{field}_{str(level).replace('.', 'p')}" for level in levels]
        axis.plot(
            levels,
            [value(tuning[name], "uncentered_rmse_mm") for name in names],
            marker="o",
            color=BLUE,
            label="tuning",
        )
        axis.plot(
            levels,
            [value(validation[name], "uncentered_rmse_mm") for name in names],
            marker="o",
            color=ORANGE,
            linestyle="--",
            label="held-out diagnostic",
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Adjusted-travel RMSE (mm)")
        axis.axvline(
            {"bump_len_s": 0.3, "bump_mag_min_mg": 1000,
             "still_a_max_mm_s2": 1000, "bump_dx_min_mm": 20}[field],
            color=GRAY,
            linewidth=1,
            alpha=0.7,
        )
    axes[0, 0].legend(frameon=False)
    fig.suptitle("One-parameter sensitivity (vertical line = current value)")
    fig.tight_layout()
    fig.savefig(output_dir / "key_parameter_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def plot_selector_comparison(output_dir: Path) -> None:
    tuning = balanced(read_rows(output_dir / "screen_aggregate.csv"), "tuning")
    validation = balanced(
        read_rows(output_dir / "diagnostic_validation_screen_aggregate.csv"),
        "validation",
    )
    configs = (
        ("current_generic", "mag band\n(current)"),
        ("selector_all", "all points"),
        ("selector_x_gt_20", "x > 20 mm"),
        ("selector_x_gt_60", "x > 60 mm"),
        ("selector_x_60_80", "60 < x < 80"),
        ("selector_mag_and_x_gt_60", "mag band +\nx > 60"),
    )
    x = np.arange(len(configs))
    width = 0.38
    fig, axis = plt.subplots(figsize=(9, 4.1))
    axis.bar(
        x - width / 2,
        [value(tuning[name], "uncentered_rmse_mm") for name, _ in configs],
        width,
        color=BLUE,
        label="tuning",
    )
    axis.bar(
        x + width / 2,
        [value(validation[name], "uncentered_rmse_mm") for name, _ in configs],
        width,
        color=ORANGE,
        label="held-out diagnostic",
    )
    axis.set_xticks(x, [label for _, label in configs])
    axis.set_ylabel("Adjusted-travel RMSE (mm)")
    axis.set_title("Reference-point selector: displacement-only rules do not transfer")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "selector_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_final_performance(output_dir: Path) -> None:
    adjusted_rows = read_rows(output_dir / "finalist_aggregate.csv")
    solver_rows = read_rows(output_dir / "solver_aggregate.csv")
    adjusted = {
        split: balanced(adjusted_rows, split) for split in ("tuning", "validation")
    }
    solver = {
        split: balanced(solver_rows, split) for split in ("tuning", "validation")
    }
    groups = ("Tuning: adjusted", "Tuning: final solver", "Held out: adjusted", "Held out: final solver")
    current = (
        value(adjusted["tuning"]["current_generic"], "uncentered_rmse_mm"),
        value(solver["tuning"]["current_generic"], "uncentered_rmse_mm"),
        value(adjusted["validation"]["current_generic"], "uncentered_rmse_mm"),
        value(solver["validation"]["current_generic"], "uncentered_rmse_mm"),
    )
    short = (
        value(adjusted["tuning"]["bump_len_s_0p2"], "uncentered_rmse_mm"),
        value(solver["tuning"]["bump_len_s_0p2"], "uncentered_rmse_mm"),
        value(adjusted["validation"]["bump_len_s_0p2"], "uncentered_rmse_mm"),
        value(solver["validation"]["bump_len_s_0p2"], "uncentered_rmse_mm"),
    )
    x = np.arange(len(groups))
    width = 0.38
    fig, axis = plt.subplots(figsize=(9, 4.2))
    axis.bar(x - width / 2, current, width, color=GRAY, label="current 0.30 s")
    bars = axis.bar(x + width / 2, short, width, color=GREEN, label="selected 0.20 s")
    for bar, old, new in zip(bars, current, short):
        reduction = 100.0 * (old - new) / old
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.35,
            f"−{reduction:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xticks(x, groups)
    axis.set_ylabel("Uncentered RMSE (mm)")
    axis.set_title("Frozen tuning winner transfers to the held-out cohort")
    axis.legend(frameon=False)
    axis.set_ylim(0, max(current) * 1.18)
    fig.tight_layout()
    fig.savefig(output_dir / "selected_config_performance.png", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    plot_anchor_travel(args.output_dir)
    plot_key_parameters(args.output_dir)
    plot_selector_comparison(args.output_dir)
    plot_final_performance(args.output_dir)


if __name__ == "__main__":
    main()
