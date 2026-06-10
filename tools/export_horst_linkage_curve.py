#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.horst_linkage_example import (  # noqa: E402
    DEFAULT_ROCKER_ANGLE_DEG,
    axle_position,
    build_horst_linkage,
    rocker_angle_deg_from_mechanism,
)


@dataclass(frozen=True)
class SweepSample:
    sample_index: int
    requested_rocker_angle_deg: float
    rocker_angle_deg: float
    wheel_travel_mm: float
    axle_x_mm: float
    axle_y_mm: float
    axle_dx_mm: float
    axle_dy_mm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample the Horst linkage over a rocker-angle sweep and export a "
            "CSV that the linkage angle-to-travel step can interpolate."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="CSV path to write. The loader expects rocker_angle_deg and wheel_travel_mm columns.",
    )
    parser.add_argument(
        "--branch",
        type=int,
        choices=(0, 1),
        default=1,
        help="Assembly branch for the closure joint. 1 is the default convex layout.",
    )
    parser.add_argument(
        "--rocker-angle-start-deg",
        type=float,
        default=DEFAULT_ROCKER_ANGLE_DEG,
        help="Start of the sampled rocker-angle sweep in degrees.",
    )
    parser.add_argument(
        "--rocker-angle-stop-deg",
        type=float,
        required=True,
        help="End of the sampled rocker-angle sweep in degrees.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=121,
        help="Number of linkage poses to sample across the sweep.",
    )
    parser.add_argument(
        "--axle-offset-x",
        type=float,
        required=True,
        help="Axle x offset from B in the local frame of link 3. +x points from B to C.",
    )
    parser.add_argument(
        "--axle-offset-y",
        type=float,
        required=True,
        help="Axle y offset from B in the local frame of link 3. +y is 90 deg CCW from B->C.",
    )
    return parser.parse_args()


def sweep_linkage(
    *,
    branch: int,
    rocker_angle_start_deg: float,
    rocker_angle_stop_deg: float,
    samples: int,
    axle_offset_x_mm: float,
    axle_offset_y_mm: float,
) -> list[SweepSample]:
    if samples < 2:
        raise ValueError("--samples must be at least 2")

    requested_angles_deg = np.linspace(
        rocker_angle_start_deg,
        rocker_angle_stop_deg,
        samples,
        dtype=float,
    )

    rows: list[SweepSample] = []
    origin_axle: tuple[float, float] | None = None
    prev_axle: tuple[float, float] | None = None
    wheel_travel_mm = 0.0

    for sample_index, requested_angle_deg in enumerate(requested_angles_deg):
        mechanism = build_horst_linkage(
            branch=branch,
            omega_deg_per_step=0.0,
            rocker_angle_deg=float(requested_angle_deg),
            axle_offset_x_mm=axle_offset_x_mm,
            axle_offset_y_mm=axle_offset_y_mm,
        )
        axle = axle_position(mechanism)
        if axle is None:
            raise ValueError("This sweep requires an axle tracker; no axle position was solved")

        rocker_angle_deg = rocker_angle_deg_from_mechanism(mechanism)
        if origin_axle is None:
            origin_axle = axle
        if prev_axle is not None:
            wheel_travel_mm += math.hypot(axle[0] - prev_axle[0], axle[1] - prev_axle[1])

        rows.append(
            SweepSample(
                sample_index=sample_index,
                requested_rocker_angle_deg=float(requested_angle_deg),
                rocker_angle_deg=rocker_angle_deg,
                wheel_travel_mm=wheel_travel_mm,
                axle_x_mm=axle[0],
                axle_y_mm=axle[1],
                axle_dx_mm=axle[0] - origin_axle[0],
                axle_dy_mm=axle[1] - origin_axle[1],
            )
        )
        prev_axle = axle

    return rows


def write_curve_csv(path: Path, rows: list[SweepSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "requested_rocker_angle_deg",
                "rocker_angle_deg",
                "wheel_travel_mm",
                "axle_x_mm",
                "axle_y_mm",
                "axle_dx_mm",
                "axle_dy_mm",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_index": row.sample_index,
                    "requested_rocker_angle_deg": f"{row.requested_rocker_angle_deg:.9f}",
                    "rocker_angle_deg": f"{row.rocker_angle_deg:.9f}",
                    "wheel_travel_mm": f"{row.wheel_travel_mm:.9f}",
                    "axle_x_mm": f"{row.axle_x_mm:.9f}",
                    "axle_y_mm": f"{row.axle_y_mm:.9f}",
                    "axle_dx_mm": f"{row.axle_dx_mm:.9f}",
                    "axle_dy_mm": f"{row.axle_dy_mm:.9f}",
                }
            )


def main() -> None:
    args = parse_args()
    rows = sweep_linkage(
        branch=args.branch,
        rocker_angle_start_deg=args.rocker_angle_start_deg,
        rocker_angle_stop_deg=args.rocker_angle_stop_deg,
        samples=args.samples,
        axle_offset_x_mm=args.axle_offset_x,
        axle_offset_y_mm=args.axle_offset_y,
    )
    write_curve_csv(args.output, rows)

    rocker_angles = np.asarray([row.rocker_angle_deg for row in rows], dtype=float)
    wheel_travel = np.asarray([row.wheel_travel_mm for row in rows], dtype=float)
    monotonic_sign = np.sign(np.nanmean(np.diff(rocker_angles)))
    monotonic = np.all(np.diff(rocker_angles) >= 0.0) or np.all(np.diff(rocker_angles) <= 0.0)

    print(f"Wrote {len(rows)} linkage samples to {args.output}")
    print(
        "Rocker angle range (deg): "
        f"{rocker_angles.min():.3f} .. {rocker_angles.max():.3f}"
    )
    print(
        "Wheel travel range (mm): "
        f"{wheel_travel.min():.3f} .. {wheel_travel.max():.3f}"
    )
    print(
        "Rocker angle monotonic over sweep: "
        f"{'yes' if monotonic else 'no'}"
        f" ({'increasing' if monotonic_sign >= 0 else 'decreasing'})"
    )


if __name__ == "__main__":
    main()
