#!/usr/bin/env python3
"""Test whether magnetometer residuals look like a world-fixed ambient field.

This is intentionally a small diagnostic, not a production estimator. It:

1. Learns encoder-binned primary and LIS3MDL XYZ field curves versus travel.
2. Subtracts the curves to obtain magnetic residuals.
3. Integrates gyro1 and rotates the residuals into the initial gyro1 frame.
4. Saves plots that show whether the rotated residual is more stationary.

The script reads raw samples from logs/<name>.csv and travel plus the static
LIS2-to-LIS1 alignment from the existing pipeline cache.

Example:
    ./venv/bin/python tools/front/test_ambient_field.py log085
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt


REPO_ROOT = Path(__file__).resolve().parents[2]

# Pod-v1 constraints supplied by the hardware layout:
#   magnetometer +X -> gyro +Y
#   magnetometer +Y -> gyro +Z
# Right-handed coordinates then require magnetometer +Z -> gyro +X.
PRIMARY_MAG_ORIENTATIONS = {
    "pod_v1": np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    "identity": np.eye(3),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log",
        help="Log name (for example log085 or 085) or path to a log CSV.",
    )
    parser.add_argument("--bin-mm", type=float, default=5.0, help="Travel-bin width.")
    parser.add_argument(
        "--min-bin-samples",
        type=int,
        default=50,
        help="Minimum samples required to retain an XYZ travel bin.",
    )
    parser.add_argument(
        "--primary-mag-orientation",
        choices=tuple(PRIMARY_MAG_ORIENTATIONS),
        default="pod_v1",
        help=(
            "Primary-mag to its local gyro frame. pod_v1 means "
            "gyro=(mag_z, mag_x, mag_y)."
        ),
    )
    parser.add_argument(
        "--gyro-bias-mode",
        choices=("median", "start", "none"),
        default="median",
        help="Simple gyro-bias estimate used before integration.",
    )
    parser.add_argument(
        "--gyro-bias-seconds",
        type=float,
        default=2.0,
        help="Initial interval used when --gyro-bias-mode=start.",
    )
    parser.add_argument(
        "--gyro-smooth-ms",
        type=float,
        default=25.0,
        help="Moving-average gyro smoothing before integration.",
    )
    parser.add_argument(
        "--plot-smooth-seconds",
        type=float,
        default=0.5,
        help="Residual smoothing used only in the plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: reports/ambient_field_test/<log>).",
    )
    parser.add_argument(
        "--window-s",
        type=float,
        help="Window length in seconds",
        default=5
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't show the plots"
    )
    parser.add_argument(
        "--mag-max",
        type=float,
        help="Maximum mag magnitude to measure stationarity",
        default=2000
    )
    return parser.parse_args()


def resolve_log_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate.resolve()

    name = value if value.startswith("log") else f"log{value}"
    candidate = REPO_ROOT / "logs" / f"{name}.csv"
    if not candidate.exists():
        raise FileNotFoundError(f"Could not find log CSV: {candidate}")
    return candidate


def load_pipeline_cache(log_name: str) -> np.lib.npyio.NpzFile:
    cache_path = REPO_ROOT / "backend" / "run_artifacts" / log_name / "cache" / "all.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing pipeline cache: {cache_path}\n"
            "Run the normal front pipeline for this log first. The cache supplies "
            "clean encoder travel and the static board-to-board alignment."
        )
    return np.load(cache_path)


def moving_average(values: np.ndarray, samples: int) -> np.ndarray:
    """Centered moving average with edge padding; works for 1D or XYZ arrays."""
    if samples <= 1:
        return values.copy()
    if samples % 2 == 0:
        samples += 1
    pad = samples // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    sums = np.cumsum(padded, axis=0, dtype=float)
    sums = np.vstack([np.zeros((1, values.shape[1])), sums])
    return (sums[samples:] - sums[:-samples]) / samples


def accel_loader_flip(accel_xyz: np.ndarray) -> np.ndarray:
    """Reproduce the simple frame normalization used by AccelLoader."""
    if np.mean(accel_xyz[:, 0]) > 0:
        return np.diag([-1.0, -1.0, 1.0])
    return np.eye(3)


def learn_binned_curve(
    travel: np.ndarray,
    field_xyz: np.ndarray,
    valid: np.ndarray,
    bin_mm: float,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return travel-bin centers, median XYZ per bin, and sample counts."""
    finite = valid & np.isfinite(travel) & np.all(np.isfinite(field_xyz), axis=1)
    bin_id = np.zeros(len(travel), dtype=int)
    bin_id[finite] = np.floor(travel[finite] / bin_mm).astype(int)
    centers: list[float] = []
    medians: list[np.ndarray] = []
    counts: list[int] = []

    for value in np.unique(bin_id[finite]):
        in_bin = finite & (bin_id == value)
        count = int(np.sum(in_bin))
        if count < min_samples:
            continue
        centers.append((value + 0.5) * bin_mm)
        medians.append(np.median(field_xyz[in_bin], axis=0))
        counts.append(count)

    if len(centers) < 3:
        raise RuntimeError("Fewer than three populated travel bins; reduce --min-bin-samples.")
    return np.asarray(centers), np.asarray(medians), np.asarray(counts)


def evaluate_curve(
    travel: np.ndarray,
    centers: np.ndarray,
    curve_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate an XYZ curve without extrapolating past fitted bins."""
    prediction = np.column_stack(
        [np.interp(travel, centers, curve_xyz[:, axis]) for axis in range(3)]
    )
    covered = (travel >= centers[0]) & (travel <= centers[-1])
    prediction[~covered] = np.nan
    return prediction, covered


def gyro_bias(gyro_rad_s: np.ndarray, t: np.ndarray, mode: str, seconds: float) -> np.ndarray:
    if mode == "none":
        return np.zeros(3)
    if mode == "start":
        selected = t <= t[0] + seconds
        if np.sum(selected) < 2:
            raise RuntimeError("Not enough samples in the requested initial gyro-bias interval.")
        return np.mean(gyro_rad_s[selected], axis=0)
    return np.median(gyro_rad_s, axis=0)


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def integrate_body_gyro(t: np.ndarray, gyro_rad_s: np.ndarray) -> np.ndarray:
    """Integrate body-frame angular rate; return body-to-initial-frame rotations."""
    rotations = np.empty((len(t), 3, 3), dtype=float)
    rotations[0] = np.eye(3)
    for index in range(1, len(t)):
        dt = float(t[index] - t[index - 1])
        omega_dt = 0.5 * (gyro_rad_s[index - 1] + gyro_rad_s[index]) * dt
        angle = float(np.linalg.norm(omega_dt))
        if angle < 1e-12:
            delta = np.eye(3) + skew(omega_dt)
        else:
            axis_skew = skew(omega_dt / angle)
            delta = (
                np.eye(3)
                + np.sin(angle) * axis_skew
                + (1.0 - np.cos(angle)) * (axis_skew @ axis_skew)
            )
        rotations[index] = rotations[index - 1] @ delta
    return rotations


def rotate_vectors(rotations: np.ndarray, body_vectors: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nj->ni", rotations, body_vectors)


def rotation_angle(rotation):
    """Rotation-matrix angle in radians."""
    cosine = (np.trace(rotation, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def distances_from(reference, rotations):
    relative = np.einsum("ij,njk->nik", reference.T, rotations)
    return rotation_angle(relative)


def orientation_span(rotations):
    """Approximate largest angular separation within a window."""
    # Find an orientation far from the first sample.
    first_far = np.argmax(distances_from(rotations[0], rotations))

    # Then find the orientation farthest from that one.
    distances = distances_from(rotations[first_far], rotations)
    second_far = np.argmax(distances)

    return distances[second_far]


def find_angular_windows(t, rotations, window_seconds=10, step_seconds=1):
    sample_hz = 1.0 / np.median(np.diff(t))
    window_samples = round(window_seconds * sample_hz)
    step_samples = round(step_seconds * sample_hz)

    windows = []

    for start in range(0, len(t) - window_samples + 1, step_samples):
        end = start + window_samples
        window_rotations = rotations[start:end]

        span_deg = np.degrees(orientation_span(window_rotations))

        windows.append({
            "start": start,
            "end": end,
            "start_s": t[start],
            "end_s": t[end - 1],
            "span_deg": span_deg,
        })

    return sorted(windows, key=lambda window: window["span_deg"], reverse=True)


def save_curve_csv(
    path: Path,
    primary: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    rows = []
    for sensor, (centers, xyz, counts) in (("primary", primary),):
        for center, vector, count in zip(centers, xyz, counts):
            rows.append(
                {
                    "sensor": sensor,
                    "travel_mm": center,
                    "count": count,
                    "field_x_mG": vector[0],
                    "field_y_mG": vector[1],
                    "field_z_mG": vector[2],
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def component_std(values: np.ndarray, valid: np.ndarray) -> list[float]:
    return np.std(values[valid], axis=0).round(3).tolist()


def lpf(x, fc_hz=40, fs_hz=200):
    sos = butter(N=2, Wn=fc_hz, btype="low", fs=fs_hz, output="sos")

    return sosfiltfilt(sos, x, axis=0)


def stationarity_rms(values: np.ndarray, valid: np.ndarray) -> float:
    """RMS distance from the best constant vector over the selected samples."""
    selected = values[valid]
    centered = selected - np.mean(selected, axis=0)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def plot_results(
    output_path: Path,
    t: np.ndarray,
    primary_curve: tuple[np.ndarray, np.ndarray, np.ndarray],
    primary_body_residual: np.ndarray,
    primary_world_residual: np.ndarray,
    primary_valid: np.ndarray,
    plot_smooth_samples: int,
) -> None:
    colors = ("tab:red", "tab:green", "tab:blue")
    labels = ("X", "Y", "Z")
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)

    for axis, color, label in zip(range(3), colors, labels):
        axes[0, 0].plot(primary_curve[0], primary_curve[1][:, axis], color=color, label=label)
    axes[0, 0].set_title("Primary encoder-binned field curve")
    axes[0, 1].set_title("LIS3MDL encoder-binned field curve")
    for axis in axes[0]:
        axis.set_xlabel("Travel (mm)")
        axis.set_ylabel("Field (mG)")
        axis.grid(alpha=0.25)
        axis.legend(ncol=3)

    def fill_for_plot(values: np.ndarray) -> np.ndarray:
        filled = values.copy()
        sample_index = np.arange(len(filled))
        for component in range(3):
            finite = np.isfinite(filled[:, component])
            if np.any(finite):
                filled[:, component] = np.interp(
                    sample_index, sample_index[finite], filled[finite, component]
                )
        return filled

    body_plot = moving_average(fill_for_plot(primary_body_residual), plot_smooth_samples)
    world_plot = moving_average(fill_for_plot(primary_world_residual), plot_smooth_samples)
    elapsed = t - t[0]

    for axis, color, label in zip(range(3), colors, labels):
        axes[1, 0].plot(elapsed[primary_valid], body_plot[primary_valid, axis], color=color, label=label)
        axes[1, 1].plot(
            elapsed[primary_valid], world_plot[primary_valid, axis], color=color, label=f"Primary {label}"
        )

    axes[1, 0].set_title("Primary residual in gyro1/body frame")
    axes[1, 1].set_title("Residual rotated into initial-world frame")
    for axis in axes[1]:
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Residual field (mG)")
        axis.grid(alpha=0.25)
    axes[1, 0].legend(ncol=3)
    axes[1, 1].legend(ncol=2, fontsize=8)

    figure.suptitle(
        "Ambient-field hypothesis test\n"
        "A world-fixed ambient field should look flatter in the lower-right plot",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    log_path = resolve_log_path(args.log)
    log_name = log_path.stem
    output_dir = args.output_dir or REPO_ROOT / "reports" / "ambient_field_test" / log_name
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(log_path)
    cache = load_pipeline_cache(log_name)
    required = {
        "t_s",
        "lis1_x", "lis1_y", "lis1_z",
        "lis2_x", "lis2_y", "lis2_z",
        "gyro1_dps10_x", "gyro1_dps10_y", "gyro1_dps10_z",
        "mmc_mG_x", "mmc_mG_y", "mmc_mG_z",
        "lis3mdl_mG_x", "lis3mdl_mG_y", "lis3mdl_mG_z",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Log is missing required columns: {missing}")

    t = df["t_s"].to_numpy(dtype=float)
    sample_hz = 1.0 / np.median(np.diff(t))
    travel = np.interp(t, cache["travel__t"], cache["travel__x"][:, 0])

    primary_raw = df[["mmc_mG_x", "mmc_mG_y", "mmc_mG_z"]].to_numpy(dtype=float)
    primary_filt = lpf(primary_raw)
    primary_norm = np.linalg.norm(primary_filt, axis=1)
    flip1 = accel_loader_flip(df[["lis1_x", "lis1_y", "lis1_z"]].to_numpy(dtype=float))
    primary_to_local_gyro = PRIMARY_MAG_ORIENTATIONS[args.primary_mag_orientation]
    primary_to_body = flip1 @ primary_to_local_gyro
    primary_body = primary_filt @ primary_to_body.T

    primary_valid = np.ones(len(t), dtype=bool)

    primary_curve = learn_binned_curve(
        travel, primary_body, primary_valid, args.bin_mm, args.min_bin_samples
    )
    primary_expected, primary_covered = evaluate_curve(travel, primary_curve[0], primary_curve[1])
    primary_valid &= primary_covered

    primary_body_residual = primary_body - primary_expected

    gyro_dps = 0.1 * df[
        ["gyro1_dps10_x", "gyro1_dps10_y", "gyro1_dps10_z"]
    ].to_numpy(dtype=float)
    gyro_rad_s = np.deg2rad(gyro_dps)
    gyro_rad_s = lpf(gyro_rad_s)
    bias = gyro_bias(gyro_rad_s, t, args.gyro_bias_mode, args.gyro_bias_seconds)
    rotations = integrate_body_gyro(t, gyro_rad_s - bias)
    primary_world_residual = rotate_vectors(rotations, primary_body_residual)

    # Find chunks with large amounts of angular movement
    windows = find_angular_windows(t, rotations, window_seconds=args.window_s, step_seconds=args.window_s)

    stat_valid = primary_valid & (primary_norm <= args.mag_max)

    world_stats = []
    body_stats = []
    for window in windows:
        slice_i = slice(window['start'], window['end'])
        w_valid = stat_valid[slice_i]
        if np.sum(w_valid) < 2:
            continue
        world_res = primary_world_residual[slice_i]
        body_res = primary_body_residual[slice_i]
        world_stats.append(stationarity_rms(world_res, w_valid))
        body_stats.append(stationarity_rms(body_res, w_valid))

    world_stats = np.asarray(world_stats)
    body_stats = np.asarray(body_stats)
    stat_delta = world_stats - body_stats
    print(f"World stationarity mean {np.mean(world_stats):.2f}, body mean {np.mean(body_stats):.2f}")
    print(f"Mean difference {np.mean(stat_delta):.2f}")
    
    n_w = 5
    plt.figure(figsize=(14,10))
    for i, window in enumerate(windows[:n_w]):
        print(
            f"{window['start_s']:.1f}–{window['end_s']:.1f} s: "
            f"{window['span_deg']:.1f}° span"
        )
        slice_i = slice(window['start'], window['end'])
        plt.subplot(4, n_w, i+1)
        plt.plot(primary_norm[slice_i])
        plt.subplot(4, n_w, (n_w)+i+1)
        plt.plot(primary_body_residual[slice_i])
        plt.subplot(4, n_w, (2 * n_w)+i+1)
        plt.plot(primary_world_residual[slice_i])
        plt.subplot(4, n_w, (3 * n_w)+i+1)
        plt.plot(gyro_rad_s[slice_i])
    plt.tight_layout()
    if not args.no_show:
        plt.show()

    plot_samples = max(1, int(round(args.plot_smooth_seconds * sample_hz)))
    plot_path = output_dir / "ambient_residuals.png"
    plot_results(
        plot_path,
        t,
        primary_curve,
        primary_body_residual,
        primary_world_residual,
        primary_valid,
        plot_samples,
    )
    save_curve_csv(output_dir / "field_curves.csv", primary_curve)

    summary = {
        "log": log_name,
        "samples": len(t),
        "sample_hz": sample_hz,
        "primary_mag_orientation": args.primary_mag_orientation,
        "gyro_bias_dps": np.rad2deg(bias).round(6).tolist(),
        "primary_to_gyro1_matrix": primary_to_body.round(8).tolist(),
        "primary_body_residual_std_mG": component_std(primary_body_residual, primary_valid),
        "primary_world_residual_std_mG": component_std(primary_world_residual, primary_valid),
        "primary_body_stationarity_rms_mG": stationarity_rms(primary_body_residual, primary_valid),
        "primary_world_stationarity_rms_mG": stationarity_rms(primary_world_residual, primary_valid),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Log: {log_name}")
    print(f"Primary mag orientation: {args.primary_mag_orientation}")
    print(f"Gyro bias estimate (deg/s): {summary['gyro_bias_dps']}")
    print(f"Primary residual std in body frame (mG): {summary['primary_body_residual_std_mG']}")
    print(f"Primary residual std in initial-world frame (mG): {summary['primary_world_residual_std_mG']}")
    print(
        "Primary constant-vector RMS, body -> world (mG): "
        f"{summary['primary_body_stationarity_rms_mG']:.2f} -> "
        f"{summary['primary_world_stationarity_rms_mG']:.2f}"
    )
    print(f"Plot: {plot_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
