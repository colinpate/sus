#!/usr/bin/env python3
"""Plot the magnetic reference-point placement experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = ("#9a6a3a", "#6685a3", "#3f7d68")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    aggregate = read_rows(args.output_dir / "aggregate.csv")
    per_log = read_rows(args.output_dir / "per_log.csv")
    variants = (
        ("pre_uncorrected_automatic_0.2s", "detect raw,\napply before"),
        ("post_uncorrected_automatic_0.2s", "detect raw,\napply after"),
        ("post_corrected_automatic_0.2s", "detect corrected,\napply after (current)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axis = axes[0]
    x = np.arange(2)
    width = 0.24
    for index, ((variant, label), color) in enumerate(zip(variants, COLORS)):
        values = []
        for split in ("tuning", "validation"):
            row = next(
                row
                for row in aggregate
                if row["split"] == split
                and row["stage"] == "final_solver"
                and row["scope"] == "cohort-balanced"
                and row["variant"] == variant
            )
            values.append(float(row["uncentered_rmse_mm"]))
        axis.bar(x + (index - 1) * width, values, width, color=color, label=label)
    axis.set_xticks(x, ("front-default\n(tuning)", "Slayer\n(held out)"))
    axis.set_ylabel("Final-solver RMSE (mm)")
    axis.set_title("Automatic 0.20 s reference")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.2)

    axis = axes[1]
    selected = [
        row
        for row in per_log
        if row["split"] == "validation"
        and row["stage"] == "final_solver"
        and row["variant"]
        in {
            "pre_uncorrected_automatic_0.2s",
            "post_corrected_automatic_0.2s",
        }
    ]
    parents = sorted({row["parent_log"] for row in selected})
    deltas = []
    for parent in parents:
        pre = np.mean(
            [
                float(row["uncentered_rmse_mm"])
                for row in selected
                if row["parent_log"] == parent
                and row["variant"] == "pre_uncorrected_automatic_0.2s"
            ]
        )
        post = np.mean(
            [
                float(row["uncentered_rmse_mm"])
                for row in selected
                if row["parent_log"] == parent
                and row["variant"] == "post_corrected_automatic_0.2s"
            ]
        )
        deltas.append(post - pre)
    y = np.arange(len(parents))
    axis.barh(
        y,
        deltas,
        color=[COLORS[2] if delta <= 0 else "#b34d4d" for delta in deltas],
    )
    axis.axvline(0.0, color="#555555", linewidth=1)
    axis.set_yticks(y, parents)
    axis.set_xlabel("Post minus pre RMSE (mm); lower is better")
    axis.set_title("Held-out effect by independent parent")
    axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Reference-point placement comparison")
    fig.tight_layout()
    fig.savefig(args.output_dir / "placement_comparison.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
