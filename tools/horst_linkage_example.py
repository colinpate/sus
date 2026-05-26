#!/usr/bin/env python3
"""Build and draw a simple 4-bar Horst linkage with pylinkage.

This example interprets the linkage as a closed A-B-C-D loop:

    A ---- B
    |      |
    D ---- C

With the user-provided link lengths assigned as:
    Link 1 (frame):      D-A = 148 mm
    Link 2 (rocker):     C-D = 98 mm
    Link 3 (seatstay):   B-C = 374 mm
    Link 4 (chainstay):  A-B = 379 mm

Assumptions for the initial pose:
    - The chainstay (A-B) is horizontal.
    - The inside angle between the frame link (D-A) and chainstay (A-B)
      at pivot A is 110 degrees.
    - The default rocker angle reproduces that same pose.
    - We use the convex assembly branch by default so the 4-bar draws as a
      non-crossed loop.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable
import json

from pylinkage.mechanism import Mechanism, MechanismBuilder

LINK_1_FRAME_MM = 148.0
LINK_2_ROCKER_MM = 98.0
LINK_3_SEATSTAY_MM = 374.0
LINK_4_CHAINSTAY_MM = 379.0
INITIAL_INSIDE_ANGLE_DEG = 110.0
DEFAULT_ROCKER_ANGLE_DEG = 3.753006490754633


def build_horst_linkage(
    branch: int = 1,
    omega_deg_per_step: float = 1.0,
    rocker_angle_deg: float = DEFAULT_ROCKER_ANGLE_DEG,
    axle_offset_x_mm: float | None = None,
    axle_offset_y_mm: float | None = None,
) -> Mechanism:
    """Create the 4-bar mechanism for a given rocker angle at D."""
    inside_angle_rad = math.radians(INITIAL_INSIDE_ANGLE_DEG)
    main_pivot = (0.0, 0.0)

    # Place the second frame pivot so the frame-chainstay inside angle at A
    # is 110 degrees, measured counterclockwise from the horizontal chainstay.
    frame_pivot = (
        LINK_1_FRAME_MM * math.cos(inside_angle_rad),
        LINK_1_FRAME_MM * math.sin(inside_angle_rad),
    )

    builder = (
        MechanismBuilder("horst-4bar")
        .add_ground_link("frame", ports={"A": main_pivot, "D": frame_pivot})
        .add_driver_link(
            "rocker",
            length=LINK_2_ROCKER_MM,
            motor_port="D",
            omega=math.radians(omega_deg_per_step),
            initial_angle=math.radians(rocker_angle_deg),
        )
        .add_link("seatstay", length=LINK_3_SEATSTAY_MM)
        .add_link("chainstay", length=LINK_4_CHAINSTAY_MM)
        .connect("rocker.tip", "seatstay.1")
        .connect("seatstay.0", "chainstay.1")
        .connect("chainstay.0", "frame.A")
        .set_branch("seatstay.0", 1 - branch)
    )

    if axle_offset_x_mm is not None and axle_offset_y_mm is not None:
        builder.add_point_tracker(
            "axle",
            ref_port1="seatstay.0",
            ref_port2="seatstay.1",
            distance=math.hypot(axle_offset_x_mm, axle_offset_y_mm),
            angle=math.atan2(axle_offset_y_mm, axle_offset_x_mm),
        )

    return builder.build()


def find_joint_position(mechanism: Mechanism, *tokens: str) -> tuple[float, float]:
    """Find a joint position by matching identifying substrings in its ID."""
    for joint in mechanism.joints:
        if all(token in joint.id for token in tokens):
            x, y = joint.position
            if x is None or y is None:
                raise ValueError(f"Joint {joint.id} does not have a solved position")
            return (x, y)
    raise KeyError(f"Could not find a joint containing tokens: {tokens}")


def named_points(mechanism: Mechanism) -> dict[str, tuple[float, float]]:
    """Return the four bar pivots as A, B, C, D."""
    return {
        "A": find_joint_position(mechanism, "frame.A", "chainstay.0"),
        "B": find_joint_position(mechanism, "chainstay.1", "seatstay.0"),
        "C": find_joint_position(mechanism, "rocker.tip", "seatstay.1"),
        "D": find_joint_position(mechanism, "frame.D", "rocker.motor"),
    }


def rocker_angle_deg_from_mechanism(mechanism: Mechanism) -> float:
    """Return the rocker angle at D, measured from the global +x axis."""
    points = named_points(mechanism)
    dx = points["C"][0] - points["D"][0]
    dy = points["C"][1] - points["D"][1]
    return math.degrees(math.atan2(dy, dx))


def find_joint_index(mechanism: Mechanism, joint_id: str) -> int:
    """Return the index of a joint by exact ID."""
    for index, joint in enumerate(mechanism.joints):
        if joint.id == joint_id:
            return index
    raise KeyError(f"Could not find joint with id {joint_id!r}")


def axle_position(mechanism: Mechanism) -> tuple[float, float] | None:
    """Return the axle tracker position if present."""
    for joint in mechanism.joints:
        if joint.id == "axle":
            x, y = joint.position
            if x is None or y is None:
                raise ValueError("Axle tracker exists but does not have a solved position")
            return (x, y)
    return None


def axle_path(
    branch: int,
    omega_deg_per_step: float,
    rocker_angle_deg: float,
    axle_offset_x_mm: float,
    axle_offset_y_mm: float,
    steps: int,
) -> list[tuple[float, float]]:
    """Simulate and return the axle path for a number of driver steps."""
    mechanism = build_horst_linkage(
        branch=branch,
        omega_deg_per_step=omega_deg_per_step,
        rocker_angle_deg=rocker_angle_deg,
        axle_offset_x_mm=axle_offset_x_mm,
        axle_offset_y_mm=axle_offset_y_mm,
    )
    axle_idx = find_joint_index(mechanism, "axle")
    path: list[tuple[float, float]] = []

    initial = mechanism.joints[axle_idx].position
    if initial[0] is None or initial[1] is None:
        raise ValueError("Axle tracker exists but its initial position is undefined")
    path.append((initial[0], initial[1]))

    angle = 0
    angles = [0]
    for _ in range(steps):
        frame = next(mechanism.step(dt=1.0))
        x, y = frame[axle_idx]
        if x is not None and y is not None:
            path.append((x, y))
            angle += omega_deg_per_step
            angles.append(angle)

    return path, angles


def draw_horst_linkage(
    mechanism: Mechanism,
    output_path: Path | None = None,
    show_plot: bool = True,
    axle_path_points: Iterable[tuple[float, float]] | None = None,
) -> None:
    """Draw the initial linkage geometry."""
    import matplotlib.pyplot as plt
    from pylinkage.visualizer import plot_static_linkage

    fig, ax = plt.subplots(figsize=(9, 6))
    loci = [tuple(joint.position for joint in mechanism.joints)]
    plot_static_linkage(
        mechanism,
        axis=ax,
        loci=loci,
        show_labels=False,
        show_loci=False,
        title="4-Bar Horst Linkage Initial Pose",
    )

    points = named_points(mechanism)
    for label, (x, y) in points.items():
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=10,
            fontweight="bold",
        )

    axle = axle_position(mechanism)
    if axle is not None:
        ax.plot(axle[0], axle[1], "o", color="tab:red", markersize=8, zorder=6)
        ax.annotate(
            "axle",
            axle,
            textcoords="offset points",
            xytext=(6, -14),
            fontsize=10,
            fontweight="bold",
            color="tab:red",
        )

    if axle_path_points:
        axle_path_points = list(axle_path_points)
        ax.plot(
            [point[0] for point in axle_path_points],
            [point[1] for point in axle_path_points],
            "--",
            color="tab:red",
            linewidth=1.5,
            alpha=0.8,
            label="Axle path",
        )

    ax.text(
        0.02,
        0.98,
        "\n".join(
            [
                f"Link 1 (frame): {LINK_1_FRAME_MM:.0f} mm",
                f"Link 2 (rocker): {LINK_2_ROCKER_MM:.0f} mm",
                f"Link 3 (seatstay): {LINK_3_SEATSTAY_MM:.0f} mm",
                f"Link 4 (chainstay): {LINK_4_CHAINSTAY_MM:.0f} mm",
                f"Frame angle at A: {INITIAL_INSIDE_ANGLE_DEG:.0f} deg",
                f"Rocker angle at D: {rocker_angle_deg_from_mechanism(mechanism):.3f} deg",
            ]
        ),
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save a PNG/SVG/PDF of the linkage plot.",
    )
    parser.add_argument(
        "--branch",
        type=int,
        choices=(0, 1),
        default=1,
        help="Assembly branch for the closure joint. 1 is the default convex layout.",
    )
    parser.add_argument(
        "--omega-deg-per-step",
        type=float,
        default=0.5,
        help="Rocker angular step in degrees per simulation step.",
    )
    parser.add_argument(
        "--rocker-angle-deg",
        type=float,
        default=DEFAULT_ROCKER_ANGLE_DEG,
        help=(
            "Initial rocker angle at D, measured from the global +x axis. "
            "The default reproduces the original horizontal-chainstay pose."
        ),
    )
    parser.add_argument(
        "--axle-offset-x",
        type=float,
        default=-20,
        help="Axle x offset from B in the local frame of link 3. +x points from B to C.",
    )
    parser.add_argument(
        "--axle-offset-y",
        type=float,
        default=-10,
        help="Axle y offset from B in the local frame of link 3. +y is 90 deg CCW from B->C.",
    )
    parser.add_argument(
        "--plot-axle-path",
        action="store_true",
        help="Also simulate and draw the axle path over a number of steps.",
    )
    parser.add_argument(
        "--path-steps",
        type=int,
        default=120,
        help="Number of driver steps to use for the optional axle path plot.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window.",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="File to export axle pts to"
    )
    args = parser.parse_args()

    if (args.axle_offset_x is None) != (args.axle_offset_y is None):
        parser.error("Provide both --axle-offset-x and --axle-offset-y, or neither.")
    if args.path_steps < 1:
        parser.error("--path-steps must be at least 1.")

    mechanism = build_horst_linkage(
        branch=args.branch,
        omega_deg_per_step=args.omega_deg_per_step,
        rocker_angle_deg=args.rocker_angle_deg,
        axle_offset_x_mm=args.axle_offset_x,
        axle_offset_y_mm=args.axle_offset_y,
    )
    points = named_points(mechanism)
    axle = axle_position(mechanism)

    print("Initial pivot coordinates (mm):")
    for label in ("A", "B", "C", "D"):
        x, y = points[label]
        print(f"  {label}: ({x:9.3f}, {y:9.3f})")
    print(f"  rocker angle at D: {rocker_angle_deg_from_mechanism(mechanism):9.3f} deg")

    if axle is not None:
        print(f"  axle: ({axle[0]:9.3f}, {axle[1]:9.3f})")

    axle_path_points = None
    if args.plot_axle_path or args.export:
        if args.axle_offset_x is None or args.axle_offset_y is None:
            parser.error("--plot-axle-path requires --axle-offset-x and --axle-offset-y.")
        axle_path_points, angles = axle_path(
            branch=args.branch,
            omega_deg_per_step=args.omega_deg_per_step,
            rocker_angle_deg=args.rocker_angle_deg,
            axle_offset_x_mm=args.axle_offset_x,
            axle_offset_y_mm=args.axle_offset_y,
            steps=args.path_steps,
        )
    
    draw_horst_linkage(
        mechanism,
        output_path=args.output,
        show_plot=not args.no_show,
        axle_path_points=axle_path_points,
    )
    if args.export:
        with open(args.export, "w") as fo:
            json.dump([angles, axle_path_points], fo, indent=4)


if __name__ == "__main__":
    main()
