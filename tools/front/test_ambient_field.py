#!/usr/bin/env python3
"""Test whether magnetometer residuals look like a world-fixed ambient field.

This is intentionally a small diagnostic, not a production estimator. It:

1. Learns an encoder-binned primary-magnetometer XYZ curve versus travel.
2. Subtracts the curve to obtain the magnetic residual.
3. Integrates gyro1 and rotates the residual into a local world frame.
4. Cross-validates world-only and body-offset + world-field models in local windows.

This uses encoder travel to build and score the models. It is an offline test of
what an anchored or joint estimator might recover, not an online estimator.

The script reads raw samples from logs/<name>.csv and cleaned encoder travel
from the existing pipeline cache.

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
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[2]

# Empirical pod-v1 mapping between the recorded MMC and gyro1 channels:
#   magnetometer +X -> gyro +Z
#   magnetometer +Y -> gyro -Y
#   magnetometer +Z -> gyro +X
PRIMARY_MAG_ORIENTATIONS = {
    "pod_v1": np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]),
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
            "gyro=(mag_z, -mag_y, mag_x)."
        ),
    )
    parser.add_argument(
        "--gyro-bias-mode",
        choices=("median", "start", "none"),
        default="none",
        help="Simple gyro-bias estimate used before integration.",
    )
    parser.add_argument(
        "--gyro-bias-seconds",
        type=float,
        default=2.0,
        help="Initial interval used when --gyro-bias-mode=start.",
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
        default=20,
    )
    parser.add_argument(
        "--window-step-s",
        type=float,
        help="Spacing between candidate window starts in seconds",
        default=1,
    )
    parser.add_argument(
        "--min-window-span-deg",
        type=float,
        help="Ignore windows with less angular excitation than this",
        default=3,
    )
    parser.add_argument(
        "--min-window-valid-fraction",
        type=float,
        help="Minimum fraction of low-magnitude, curve-covered samples per window",
        default=0.2,
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
        default=1500,
    )
    parser.add_argument(
        "--mag-gate-source",
        choices=("expected", "measured"),
        default="expected",
        help=(
            "Choose whether --mag-max applies to the learned travel curve or the "
            "measured field. Expected is less contaminated by ambient field."
        ),
    )
    parser.add_argument(
        "--mag-filter-hz",
        type=float,
        default=20.0,
        help="Primary-magnetometer low-pass cutoff.",
    )
    parser.add_argument(
        "--gyro-filter-hz",
        type=float,
        default=40.0,
        help="Gyro low-pass cutoff.",
    )
    parser.add_argument(
        "--cv-block-s",
        type=float,
        default=1.0,
        help="Alternating train/test block length for held-out model scoring.",
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
            "clean encoder travel."
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
    window_samples = max(2, round(window_seconds * sample_hz))
    step_samples = max(1, round(step_seconds * sample_hz))

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


def select_non_overlapping_windows(windows: list[dict], count: int) -> list[dict]:
    """Select the highest-span windows without showing the same event repeatedly."""
    selected = []
    for candidate in windows:
        overlaps = any(
            candidate["start"] < existing["end"]
            and existing["start"] < candidate["end"]
            for existing in selected
        )
        if not overlaps:
            selected.append(candidate)
        if len(selected) == count:
            break
    return selected


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
    if fc_hz >= fs_hz / 2:
        raise ValueError(f"Low-pass cutoff {fc_hz} Hz must be below Nyquist ({fs_hz / 2} Hz)")
    sos = butter(N=2, Wn=fc_hz, btype="low", fs=fs_hz, output="sos")

    return sosfiltfilt(sos, x, axis=0)


def stationarity_rms(values: np.ndarray, valid: np.ndarray) -> float:
    """RMS distance from the best constant vector over the selected samples."""
    selected = values[valid]
    centered = selected - np.mean(selected, axis=0)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def vector_model_design(rotations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return designs for a body-fixed vector and a world-fixed vector."""
    sample_count = len(rotations)
    body_design = np.tile(np.eye(3), (sample_count, 1))
    # rotations map body -> local world, so R.T maps a world vector into body.
    world_design = rotations.transpose(0, 2, 1).reshape(-1, 3)
    return body_design, world_design


def vector_prediction(design: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return (design @ coefficients).reshape(-1, 3)


def prediction_rms(
    values: np.ndarray,
    selected: np.ndarray,
    design: np.ndarray,
    coefficients: np.ndarray,
) -> float:
    error = values[selected] - vector_prediction(design[selected.repeat(3)], coefficients)
    return float(np.sqrt(np.mean(np.sum(error * error, axis=1))))


def cross_validated_vector_models(
    t: np.ndarray,
    rotations: np.ndarray,
    body_values: np.ndarray,
    measured_body: np.ndarray,
    true_travel: np.ndarray,
    valid: np.ndarray,
    block_seconds: float,
    curve_tree: cKDTree,
    curve_travel: np.ndarray,
) -> dict:
    """Compare body-only, world-only, and body+world models on held-out blocks."""
    if block_seconds <= 0:
        raise ValueError("--cv-block-s must be positive")

    block_id = np.floor((t - t[0]) / block_seconds).astype(int)
    test = valid & (block_id % 2 == 1)
    train = valid & ~test
    if np.sum(train) < 6 or np.sum(test) < 6:
        raise RuntimeError("Not enough valid samples in both CV halves")

    body_design, world_design = vector_model_design(rotations)
    combined_design = np.column_stack((body_design, world_design))
    flattened_values = body_values.reshape(-1)
    train_rows = train.repeat(3)

    def fit(design: np.ndarray) -> np.ndarray:
        return np.linalg.lstsq(
            design[train_rows], flattened_values[train_rows], rcond=None
        )[0]

    body_coefficients = fit(body_design)
    world_coefficients = fit(world_design)
    combined_coefficients = fit(combined_design)
    body_rms = prediction_rms(body_values, test, body_design, body_coefficients)
    world_rms = prediction_rms(body_values, test, world_design, world_coefficients)
    combined_rms = prediction_rms(
        body_values, test, combined_design, combined_coefficients
    )

    def inferred_travel(field: np.ndarray) -> np.ndarray:
        _, curve_index = curve_tree.query(field)
        return curve_travel[curve_index]

    def travel_rmse(field: np.ndarray) -> float:
        error = inferred_travel(field) - true_travel[test]
        return float(np.sqrt(np.mean(error * error)))

    world_ambient_body = vector_prediction(
        world_design[test.repeat(3)], world_coefficients
    )
    combined_ambient_body = vector_prediction(
        world_design[test.repeat(3)], combined_coefficients[3:]
    )
    measured_test = measured_body[test]
    travel_baseline_rms = travel_rmse(measured_test)
    travel_world_rms = travel_rmse(measured_test - world_ambient_body)
    travel_combined_ambient_rms = travel_rmse(
        measured_test - combined_ambient_body
    )
    travel_combined_all_rms = travel_rmse(
        measured_test
        - combined_ambient_body
        - combined_coefficients[:3][None, :]
    )

    # Fit all valid samples only to visualize the inferred ambient subtraction.
    valid_rows = valid.repeat(3)
    full_coefficients = np.linalg.lstsq(
        combined_design[valid_rows], flattened_values[valid_rows], rcond=None
    )[0]
    ambient_world = full_coefficients[3:]
    ambient_in_body = vector_prediction(world_design, ambient_world)

    return {
        "cv_body_rms_mG": body_rms,
        "cv_world_rms_mG": world_rms,
        "cv_body_world_rms_mG": combined_rms,
        "cv_world_improvement_fraction": (body_rms - world_rms) / body_rms,
        "cv_body_world_improvement_fraction": (body_rms - combined_rms) / body_rms,
        "travel_baseline_rmse_mm": travel_baseline_rms,
        "travel_world_rmse_mm": travel_world_rms,
        "travel_combined_ambient_rmse_mm": travel_combined_ambient_rms,
        "travel_combined_all_rmse_mm": travel_combined_all_rms,
        "travel_world_improvement_fraction": (
            travel_baseline_rms - travel_world_rms
        )
        / travel_baseline_rms,
        "travel_combined_ambient_improvement_fraction": (
            travel_baseline_rms - travel_combined_ambient_rms
        )
        / travel_baseline_rms,
        "travel_combined_all_improvement_fraction": (
            travel_baseline_rms - travel_combined_all_rms
        )
        / travel_baseline_rms,
        "body_offset_mG": full_coefficients[:3],
        "ambient_world_mG": ambient_world,
        "ambient_corrected_residual": body_values - ambient_in_body,
        "combined_design_condition": float(
            np.linalg.cond(combined_design[train_rows])
        ),
    }


def analyze_stationarity_windows(
    windows: list[dict],
    t: np.ndarray,
    rotations: np.ndarray,
    measured_body: np.ndarray,
    body_residual: np.ndarray,
    travel: np.ndarray,
    valid: np.ndarray,
    min_span_deg: float,
    min_valid_fraction: float,
    cv_block_seconds: float,
    curve_tree: cKDTree,
    curve_travel: np.ndarray,
) -> list[dict]:
    """Score each window in its own local frame and with held-out prediction."""
    results = []
    for window in windows:
        if window["span_deg"] < min_span_deg:
            continue

        start, end = window["start"], window["end"]
        window_valid = valid[start:end]
        valid_fraction = float(np.mean(window_valid))
        if valid_fraction < min_valid_fraction or np.sum(window_valid) < 2:
            continue

        # Removing the window-start orientation is exactly equivalent to
        # re-integrating the same gyro deltas from identity, but much faster.
        local_rotations = np.einsum(
            "ij,njk->nik", rotations[start].T, rotations[start:end]
        )
        local_world_residual = rotate_vectors(local_rotations, body_residual[start:end])
        body_rms = stationarity_rms(body_residual[start:end], window_valid)
        world_rms = stationarity_rms(local_world_residual, window_valid)
        try:
            model_scores = cross_validated_vector_models(
                t[start:end],
                local_rotations,
                body_residual[start:end],
                measured_body[start:end],
                travel[start:end],
                window_valid,
                cv_block_seconds,
                curve_tree,
                curve_travel,
            )
        except RuntimeError:
            continue
        results.append(
            {
                **window,
                "valid_fraction": valid_fraction,
                "body_rms_mG": body_rms,
                "world_rms_mG": world_rms,
                "delta_mG": world_rms - body_rms,
                "local_world_residual": local_world_residual,
                **model_scores,
            }
        )
    return results


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
    axes[0, 1].plot(
        primary_curve[0],
        np.linalg.norm(primary_curve[1], axis=1),
        color="black",
        label="Magnitude",
    )
    axes[0, 1].set_title("Primary encoder-binned field magnitude")
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
        axes[1, 0].plot(
            elapsed[primary_valid],
            body_plot[primary_valid, axis],
            color=color,
            label=label,
        )
        axes[1, 1].plot(
            elapsed[primary_valid],
            world_plot[primary_valid, axis],
            color=color,
            label=f"Primary {label}",
        )

    axes[1, 0].set_title("Primary residual in gyro1/body frame")
    axes[1, 1].set_title("Whole-record world frame (gyro-drift sensitive)")
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
        "gyro1_dps10_x", "gyro1_dps10_y", "gyro1_dps10_z",
        "mmc_mG_x", "mmc_mG_y", "mmc_mG_z",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Log is missing required columns: {missing}")

    t = df["t_s"].to_numpy(dtype=float)
    sample_hz = 1.0 / np.median(np.diff(t))
    travel = np.interp(t, cache["travel__t"], cache["travel__x"][:, 0])

    primary_raw = df[["mmc_mG_x", "mmc_mG_y", "mmc_mG_z"]].to_numpy(dtype=float)
    primary_filt = lpf(
        primary_raw, fc_hz=args.mag_filter_hz, fs_hz=sample_hz
    )
    primary_norm = np.linalg.norm(primary_filt, axis=1)
    # This matrix was measured directly between the recorded MMC and gyro1
    # channels, so no accelerometer frame normalization belongs here.
    primary_to_body = PRIMARY_MAG_ORIENTATIONS[args.primary_mag_orientation]
    primary_body = primary_filt @ primary_to_body.T

    primary_valid = np.ones(len(t), dtype=bool)

    primary_curve = learn_binned_curve(
        travel, primary_body, primary_valid, args.bin_mm, args.min_bin_samples
    )
    primary_expected, primary_covered = evaluate_curve(travel, primary_curve[0], primary_curve[1])
    primary_expected_norm = np.linalg.norm(primary_expected, axis=1)
    curve_travel = np.arange(
        primary_curve[0][0], primary_curve[0][-1] + 1e-9, 0.25
    )
    curve_field, _ = evaluate_curve(
        curve_travel, primary_curve[0], primary_curve[1]
    )
    curve_tree = cKDTree(curve_field)
    primary_valid &= primary_covered

    primary_body_residual = primary_body - primary_expected

    gyro_dps = 0.1 * df[
        ["gyro1_dps10_x", "gyro1_dps10_y", "gyro1_dps10_z"]
    ].to_numpy(dtype=float)
    gyro_rad_s = np.deg2rad(gyro_dps)
    gyro_rad_s = lpf(
        gyro_rad_s, fc_hz=args.gyro_filter_hz, fs_hz=sample_hz
    )
    bias = gyro_bias(gyro_rad_s, t, args.gyro_bias_mode, args.gyro_bias_seconds)
    gyro_corrected = gyro_rad_s - bias
    rotations = integrate_body_gyro(t, gyro_corrected)
    primary_world_residual = rotate_vectors(rotations, primary_body_residual)

    # Find angularly exciting windows, then restart gyro integration independently
    # inside each one so drift accumulated earlier in the log cannot affect it.
    windows = find_angular_windows(
        t,
        rotations,
        window_seconds=args.window_s,
        step_seconds=args.window_step_s,
    )
    gate_norm = (
        primary_expected_norm
        if args.mag_gate_source == "expected"
        else primary_norm
    )
    stat_valid = primary_valid & (gate_norm <= args.mag_max)
    window_results = analyze_stationarity_windows(
        windows,
        t,
        rotations,
        primary_body,
        primary_body_residual,
        travel,
        stat_valid,
        min_span_deg=args.min_window_span_deg,
        min_valid_fraction=args.min_window_valid_fraction,
        cv_block_seconds=args.cv_block_s,
        curve_tree=curve_tree,
        curve_travel=curve_travel,
    )
    if not window_results:
        raise RuntimeError(
            "No windows passed the angular-span and low-magnitude coverage filters. "
            "Try reducing --min-window-span-deg, increasing --mag-max, or reducing "
            "--min-window-valid-fraction."
        )

    world_stats = np.asarray([result["world_rms_mG"] for result in window_results])
    body_stats = np.asarray([result["body_rms_mG"] for result in window_results])
    stat_delta = world_stats - body_stats
    print(
        f"Window stationarity paired median improvement "
        f"{np.median(-stat_delta):+.2f} mG (body RMS - world RMS); "
        f"world improved {100 * np.mean(stat_delta < 0):.1f}% of windows"
    )

    cv_world_improvements = np.asarray(
        [result["cv_world_improvement_fraction"] for result in window_results]
    )
    cv_combined_improvements = np.asarray(
        [result["cv_body_world_improvement_fraction"] for result in window_results]
    )
    cv_body_stats = np.asarray(
        [result["cv_body_rms_mG"] for result in window_results]
    )
    cv_world_stats = np.asarray(
        [result["cv_world_rms_mG"] for result in window_results]
    )
    cv_combined_stats = np.asarray(
        [result["cv_body_world_rms_mG"] for result in window_results]
    )
    travel_baseline_stats = np.asarray(
        [result["travel_baseline_rmse_mm"] for result in window_results]
    )
    travel_world_stats = np.asarray(
        [result["travel_world_rmse_mm"] for result in window_results]
    )
    travel_combined_ambient_stats = np.asarray(
        [result["travel_combined_ambient_rmse_mm"] for result in window_results]
    )
    travel_combined_all_stats = np.asarray(
        [result["travel_combined_all_rmse_mm"] for result in window_results]
    )
    travel_world_improvements = np.asarray(
        [result["travel_world_improvement_fraction"] for result in window_results]
    )
    travel_combined_ambient_improvements = np.asarray(
        [
            result["travel_combined_ambient_improvement_fraction"]
            for result in window_results
        ]
    )
    travel_combined_all_improvements = np.asarray(
        [
            result["travel_combined_all_improvement_fraction"]
            for result in window_results
        ]
    )
    ambient_magnitudes = np.asarray(
        [np.linalg.norm(result["ambient_world_mG"]) for result in window_results]
    )
    design_conditions = np.asarray(
        [result["combined_design_condition"] for result in window_results]
    )
    print(
        "Held-out median RMS improvement: "
        f"world-only {100 * np.median(cv_world_improvements):+.1f}%, "
        f"body+world {100 * np.median(cv_combined_improvements):+.1f}%"
    )
    print(
        "Encoder-anchored held-out travel RMSE improvement: "
        f"world-only {100 * np.median(travel_world_improvements):+.1f}%, "
        f"body+world {100 * np.median(travel_combined_all_improvements):+.1f}%"
    )

    plot_windows = select_non_overlapping_windows(window_results, 5)
    n_w = len(plot_windows)
    plt.figure(figsize=(max(6, 3 * n_w), 12))
    for i, window in enumerate(plot_windows):
        print(
            f"{window['start_s']:.1f}–{window['end_s']:.1f} s: "
            f"{window['span_deg']:.1f}° span, "
            f"body {window['body_rms_mG']:.1f} -> world {window['world_rms_mG']:.1f} mG, "
            f"body+world CV {100 * window['cv_body_world_improvement_fraction']:+.1f}%"
        )
        slice_i = slice(window['start'], window['end'])
        window_t = t[slice_i] - t[window['start']]
        window_valid = stat_valid[slice_i]
        body_window = primary_body_residual[slice_i].copy()
        world_window = window["local_world_residual"].copy()
        corrected_window = window["ambient_corrected_residual"].copy()
        body_window -= np.mean(body_window[window_valid], axis=0)
        world_window -= np.mean(world_window[window_valid], axis=0)
        corrected_window -= np.mean(corrected_window[window_valid], axis=0)
        body_window[~window_valid] = np.nan
        world_window[~window_valid] = np.nan
        corrected_window[~window_valid] = np.nan

        axis = plt.subplot(5, n_w, i+1)
        axis.plot(window_t, gate_norm[slice_i])
        axis.axhline(args.mag_max, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(
            f"{window['span_deg']:.0f}°; B+W CV "
            f"{100 * window['cv_body_world_improvement_fraction']:+.0f}%"
        )
        axis = plt.subplot(5, n_w, n_w+i+1)
        axis.plot(window_t, body_window)
        axis = plt.subplot(5, n_w, 2*n_w+i+1)
        axis.plot(window_t, world_window)
        axis = plt.subplot(5, n_w, 3*n_w+i+1)
        axis.plot(window_t, corrected_window)
        axis = plt.subplot(5, n_w, 4*n_w+i+1)
        axis.plot(window_t, np.degrees(gyro_corrected[slice_i]))
        axis.set_xlabel("Window time (s)")
    plt.subplot(5, n_w, 1).set_ylabel(f"{args.mag_gate_source} |B| (mG)")
    plt.subplot(5, n_w, n_w+1).set_ylabel("Body residual (mG)")
    plt.subplot(5, n_w, 2*n_w+1).set_ylabel("World residual (mG)")
    plt.subplot(5, n_w, 3*n_w+1).set_ylabel("Ambient-subtracted (mG)")
    plt.subplot(5, n_w, 4*n_w+1).set_ylabel("Gyro (deg/s)")
    plt.suptitle(
        "Highest-angular-span non-overlapping windows\n"
        "Ambient subtraction uses the body-offset + world-field fit"
    )
    plt.tight_layout()
    window_plot_path = output_dir / "windowed_stationarity.png"
    plt.savefig(window_plot_path, dpi=150)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    spans = np.asarray([result["span_deg"] for result in window_results])
    valid_fractions = np.asarray([result["valid_fraction"] for result in window_results])
    scatter = axes[0].scatter(
        spans, -stat_delta, c=valid_fractions, cmap="viridis", alpha=0.75
    )
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Angular span in window (deg)")
    axes[0].set_ylabel("Body RMS - world RMS (mG)")
    axes[0].set_title("Strict world-fixed residual test")
    figure.colorbar(scatter, ax=axes[0], label="Valid low-field fraction")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(
        spans,
        100 * cv_world_improvements,
        alpha=0.45,
        label="World only",
    )
    axes[1].scatter(
        spans,
        100 * cv_combined_improvements,
        alpha=0.45,
        label="Body offset + world field",
    )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Angular span in window (deg)")
    axes[1].set_ylabel("Held-out RMS improvement (%)")
    axes[1].set_title("Cross-validated model comparison")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].scatter(
        spans,
        100 * travel_world_improvements,
        alpha=0.45,
        label="World only",
    )
    axes[2].scatter(
        spans,
        100 * travel_combined_ambient_improvements,
        alpha=0.45,
        label="Combined: ambient term only",
    )
    axes[2].scatter(
        spans,
        100 * travel_combined_all_improvements,
        alpha=0.45,
        label="Combined: both terms",
    )
    axes[2].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[2].set_xlabel("Angular span in window (deg)")
    axes[2].set_ylabel("Held-out travel RMSE improvement (%)")
    axes[2].set_title("Encoder-anchored nearest-curve travel")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    stationarity_scatter_path = output_dir / "windowed_stationarity_scatter.png"
    figure.savefig(stationarity_scatter_path, dpi=150)
    if not args.no_show:
        plt.show()
    plt.close("all")

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
        "mag_filter_hz": args.mag_filter_hz,
        "gyro_filter_hz": args.gyro_filter_hz,
        "gyro_bias_mode": args.gyro_bias_mode,
        "gyro_bias_dps": np.rad2deg(bias).round(6).tolist(),
        "primary_to_gyro1_matrix": primary_to_body.round(8).tolist(),
        "primary_body_residual_std_mG": component_std(primary_body_residual, primary_valid),
        "primary_world_residual_std_mG": component_std(primary_world_residual, primary_valid),
        "primary_body_stationarity_rms_mG": stationarity_rms(primary_body_residual, primary_valid),
        "primary_world_stationarity_rms_mG": stationarity_rms(
            primary_world_residual, primary_valid
        ),
        "windowed_stationarity": {
            "window_seconds": args.window_s,
            "step_seconds": args.window_step_s,
            "mag_max_mG": args.mag_max,
            "mag_gate_source": args.mag_gate_source,
            "min_span_deg": args.min_window_span_deg,
            "min_valid_fraction": args.min_window_valid_fraction,
            "cv_block_seconds": args.cv_block_s,
            "eligible_windows": len(window_results),
            "body_median_rms_mG": float(np.median(body_stats)),
            "world_median_rms_mG": float(np.median(world_stats)),
            "median_world_minus_body_mG": float(np.median(stat_delta)),
            "world_improved_fraction": float(np.mean(stat_delta < 0)),
            "cv_body_median_rms_mG": float(np.median(cv_body_stats)),
            "cv_world_median_rms_mG": float(np.median(cv_world_stats)),
            "cv_body_world_median_rms_mG": float(np.median(cv_combined_stats)),
            "cv_world_median_improvement_fraction": float(
                np.median(cv_world_improvements)
            ),
            "cv_world_improved_fraction": float(np.mean(cv_world_improvements > 0)),
            "cv_body_world_median_improvement_fraction": float(
                np.median(cv_combined_improvements)
            ),
            "cv_body_world_improved_fraction": float(
                np.mean(cv_combined_improvements > 0)
            ),
            "encoder_anchored_travel_baseline_median_rmse_mm": float(
                np.median(travel_baseline_stats)
            ),
            "encoder_anchored_travel_world_median_rmse_mm": float(
                np.median(travel_world_stats)
            ),
            "encoder_anchored_travel_combined_ambient_median_rmse_mm": float(
                np.median(travel_combined_ambient_stats)
            ),
            "encoder_anchored_travel_combined_all_median_rmse_mm": float(
                np.median(travel_combined_all_stats)
            ),
            "encoder_anchored_travel_world_median_improvement_fraction": float(
                np.median(travel_world_improvements)
            ),
            "encoder_anchored_travel_combined_ambient_median_improvement_fraction": float(
                np.median(travel_combined_ambient_improvements)
            ),
            "encoder_anchored_travel_combined_all_median_improvement_fraction": float(
                np.median(travel_combined_all_improvements)
            ),
            "median_fitted_ambient_magnitude_mG": float(
                np.median(ambient_magnitudes)
            ),
            "median_combined_design_condition": float(
                np.median(design_conditions)
            ),
            "top_non_overlapping_windows": [
                {
                    **{
                        key: window[key]
                        for key in (
                            "start_s",
                            "end_s",
                            "span_deg",
                            "valid_fraction",
                            "body_rms_mG",
                            "world_rms_mG",
                            "delta_mG",
                            "cv_body_rms_mG",
                            "cv_world_rms_mG",
                            "cv_body_world_rms_mG",
                            "cv_world_improvement_fraction",
                            "cv_body_world_improvement_fraction",
                            "travel_baseline_rmse_mm",
                            "travel_world_rmse_mm",
                            "travel_combined_ambient_rmse_mm",
                            "travel_combined_all_rmse_mm",
                            "travel_world_improvement_fraction",
                            "travel_combined_ambient_improvement_fraction",
                            "travel_combined_all_improvement_fraction",
                            "combined_design_condition",
                        )
                    },
                    "body_offset_mG": window["body_offset_mG"].round(3).tolist(),
                    "ambient_world_mG": window["ambient_world_mG"].round(3).tolist(),
                }
                for window in plot_windows
            ],
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Log: {log_name}")
    print(f"Primary mag orientation: {args.primary_mag_orientation}")
    print(f"Gyro bias estimate (deg/s): {summary['gyro_bias_dps']}")
    print(f"Primary residual std in body frame (mG): {summary['primary_body_residual_std_mG']}")
    print(
        "Primary residual std in initial-world frame (mG): "
        f"{summary['primary_world_residual_std_mG']}"
    )
    print(
        "Primary constant-vector RMS, body -> world (mG): "
        f"{summary['primary_body_stationarity_rms_mG']:.2f} -> "
        f"{summary['primary_world_stationarity_rms_mG']:.2f}"
    )
    print(
        "Window held-out median improvement, world / body+world: "
        f"{100 * np.median(cv_world_improvements):+.1f}% / "
        f"{100 * np.median(cv_combined_improvements):+.1f}%"
    )
    print(f"Plot: {plot_path}")
    print(f"Window plot: {window_plot_path}")
    print(f"Window scatter: {stationarity_scatter_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
