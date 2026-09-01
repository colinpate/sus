"""Pipeline steps for front-pod magnetic nuisance-field correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from classes.sensor_loader import Workspace
from classes.step import Step
from classes.time_series import TimeSeries
from mag_nuisance_core import (
    PRIMARY_MAG_TO_GYRO,
    MagSolverWeights,
    fit_scalar_parameterized_xyz,
    integrate_gyro,
    interpolate_nuisance_fields,
    solve_iterative_correction,
)


MAG_NUISANCE_SUMMARY_FIELDS = (
    "effective_state_hz",
    "source_stride",
    "xyz_bin_count",
    "update_fraction",
    "proposed_update_rms_mm",
    "applied_update_rms_mm",
    "xyz_scalar_center_mg",
    "xyz_scalar_scale_mg",
)


@dataclass
class MagNuisanceTravelCorrection(Step):
    """Reproduce the validated four-iteration correction on a 10 Hz grid.

    This stage intentionally emits a separate low-rate travel result. It does
    not replace the original solved travel or interpolate the correction onto
    the full-rate pipeline timeline. Keeping that boundary explicit makes this
    a parity checkpoint for the experiment before a second fusion pass is
    introduced.
    """

    state_hz: float = 10.0
    iterations: int = 4
    output_alpha: float = 0.75
    xyz_degree: int = 2
    scalar_bin_mg: float = 100.0
    travel_max_mm: float = 210.0
    travel_step_mm: float = 0.25
    min_bin_samples: int = 5
    mag_update_threshold: float = 1500.0
    world_rw: float = 1.5
    body_rw: float = 1.0
    mag_sigma: float = 40.0
    body_initial_sigma: float = 300.0
    world_initial_sigma: float = 500.0
    integrate_gyro_at_source_rate: bool = True
    mag_to_gyro_matrix: tuple[tuple[float, float, float], ...] = tuple(
        tuple(float(value) for value in row) for row in PRIMARY_MAG_TO_GYRO
    )

    def run(self, ws: Workspace) -> None:
        if len(self.inputs) != 6:
            raise ValueError(
                "MagNuisanceTravelCorrection expects mag XYZ, gyro1, scalar "
                "mag, scalar coefficients, scalar offset, and initial travel"
            )
        if len(self.outputs) != 10:
            raise ValueError(
                "MagNuisanceTravelCorrection requires ten diagnostic outputs"
            )

        mag_ts: TimeSeries = ws[self.inputs[0]]
        gyro_ts: TimeSeries = ws[self.inputs[1]]
        scalar_ts: TimeSeries = ws[self.inputs[2]]
        initial_travel_ts: TimeSeries = ws[self.inputs[5]]

        lengths = {
            len(mag_ts.t),
            len(gyro_ts.t),
            len(scalar_ts.t),
            len(initial_travel_ts.t),
        }
        if len(lengths) != 1:
            raise ValueError("Nuisance-correction inputs must be index-aligned")
        if len(mag_ts.t) < 2 or not np.all(np.diff(mag_ts.t) > 0):
            raise ValueError("Magnetometer time must be strictly increasing")

        state_hz = float(self.param(ws, "state_hz"))
        iterations = int(self.param(ws, "iterations"))
        output_alpha = float(self.param(ws, "output_alpha"))
        xyz_degree = int(self.param(ws, "xyz_degree"))
        scalar_bin_mg = float(self.param(ws, "scalar_bin_mg"))
        travel_max_mm = float(self.param(ws, "travel_max_mm"))
        travel_step_mm = float(self.param(ws, "travel_step_mm"))
        min_bin_samples = int(self.param(ws, "min_bin_samples"))
        if state_hz <= 0:
            raise ValueError("state_hz must be positive")
        if iterations < 1:
            raise ValueError("iterations must be at least one")
        if not 0.0 <= output_alpha <= 1.0:
            raise ValueError("output_alpha must be between zero and one")

        source_hz = 1.0 / float(np.median(np.diff(mag_ts.t)))
        stride = max(1, round(source_hz / state_hz))
        index = np.arange(0, len(mag_ts.t), stride)
        time_s = np.asarray(mag_ts.t, dtype=float)[index]

        mag_to_gyro = np.asarray(
            self.param(ws, "mag_to_gyro_matrix"), dtype=float
        )
        if mag_to_gyro.shape != (3, 3):
            raise ValueError("mag_to_gyro_matrix must be 3x3")
        if not np.allclose(mag_to_gyro @ mag_to_gyro.T, np.eye(3), atol=1e-6):
            raise ValueError("mag_to_gyro_matrix must be orthonormal")

        mag_xyz = np.asarray(mag_ts.x, dtype=float)[index] @ mag_to_gyro.T
        gyro_dps = np.asarray(gyro_ts.x, dtype=float)[index]
        scalar_mag = np.asarray(scalar_ts.x, dtype=float).reshape(-1)[index]
        initial_travel = np.asarray(
            initial_travel_ts.x, dtype=float
        ).reshape(-1)[index]
        scalar_coefficients = np.asarray(
            ws[self.inputs[3]], dtype=float
        ).reshape(-1)
        scalar_offset_mm = float(
            np.asarray(ws[self.inputs[4]], dtype=float).reshape(-1)[0]
        )

        xyz_model = fit_scalar_parameterized_xyz(
            scalar_mag,
            mag_xyz,
            scalar_coefficients,
            scalar_offset_mm,
            scalar_bin_mg=scalar_bin_mg,
            degree=xyz_degree,
            travel_max_mm=travel_max_mm,
            travel_step_mm=travel_step_mm,
            min_bin_samples=min_bin_samples,
        )
        weights = MagSolverWeights(
            mag_update_threshold=float(
                self.param(ws, "mag_update_threshold")
            ),
            world_rw=float(self.param(ws, "world_rw")),
            body_rw=float(self.param(ws, "body_rw")),
            mag_sigma=float(self.param(ws, "mag_sigma")),
            body_initial_sigma=float(self.param(ws, "body_initial_sigma")),
            world_initial_sigma=float(self.param(ws, "world_initial_sigma")),
        )
        integrate_gyro_at_source_rate = bool(
            self.param(ws, "integrate_gyro_at_source_rate")
        )
        state_rotations = None
        if integrate_gyro_at_source_rate:
            state_rotations = integrate_gyro(
                np.asarray(mag_ts.t, dtype=float),
                np.asarray(gyro_ts.x, dtype=float),
            )[index]
        correction = solve_iterative_correction(
            time_s,
            gyro_dps,
            mag_xyz,
            initial_travel,
            xyz_model,
            weights,
            iterations=iterations,
            body_to_reference_rotations=state_rotations,
        )
        blended_travel = initial_travel + output_alpha * (
            correction.travel - initial_travel
        )

        state_meta = {
            **mag_ts.meta,
            "fs_hz": source_hz / stride,
            "state_hz_requested": state_hz,
            "source_stride": stride,
            "iterations": iterations,
            "output_alpha": output_alpha,
            "mag_update_threshold_mg": weights.mag_update_threshold,
            "gyro_integration_hz": (
                source_hz if integrate_gyro_at_source_rate else source_hz / stride
            ),
        }

        def state_series(values: np.ndarray, units: str, frame: str) -> TimeSeries:
            return TimeSeries(
                t=time_s,
                x=np.asarray(values),
                units=units,
                frame=frame,
                meta=state_meta,
            )

        ws[self.outputs[0]] = state_series(blended_travel, "mm", "travel")
        ws[self.outputs[1]] = state_series(correction.travel, "mm", "travel")
        ws[self.outputs[2]] = state_series(
            correction.body_field, "milli-Gauss", "gyro1"
        )
        ws[self.outputs[3]] = state_series(
            correction.world_field, "milli-Gauss", "gyro1"
        )
        ws[self.outputs[4]] = state_series(
            correction.correction, "milli-Gauss", "gyro1"
        )
        ws[self.outputs[5]] = state_series(
            correction.corrected_mag_weak, "milli-Gauss", "gyro1"
        )
        ws[self.outputs[6]] = state_series(
            correction.update_mask.astype(float), "boolean", ""
        )
        ws[self.outputs[7]] = np.column_stack(
            (xyz_model.travel_grid, xyz_model.xyz_grid)
        )
        ws[self.outputs[8]] = np.asarray(correction.iteration_change_mm)

        proposed_change = correction.travel - initial_travel
        applied_change = blended_travel - initial_travel
        updated = correction.update_mask
        ws[self.outputs[9]] = np.array(
            [
                source_hz / stride,
                float(stride),
                float(xyz_model.bin_count),
                float(np.mean(updated)),
                float(np.sqrt(np.mean(proposed_change[updated] ** 2)))
                if np.any(updated)
                else 0.0,
                float(np.sqrt(np.mean(applied_change[updated] ** 2)))
                if np.any(updated)
                else 0.0,
                xyz_model.scalar_center,
                xyz_model.scalar_scale,
            ]
        )

        print(
            "Mag nuisance correction:",
            f"{len(time_s)} states at {source_hz / stride:.2f} Hz,",
            f"{xyz_model.bin_count} XYZ bins,",
            f"{np.mean(updated) * 100:.1f}% updated,",
            "iteration changes",
            np.round(correction.iteration_change_mm, 3).tolist(),
        )


@dataclass
class MagNuisanceFullRateCorrection(Step):
    """Lift slow nuisance estimates onto the full pipeline timeline.

    Two travel signals are emitted for different purposes:

    * ``delta_lifted`` adds only the interpolated 10 Hz correction delta to
      the original solved travel without resampling or low-pass filtering the
      baseline signal.
    * ``corrected_mag_travel`` is a full-rate magnetometer-only observation
      made by subtracting the interpolated XYZ nuisance field and projecting
      onto the learned XYZ path. It is suitable for a second fusion pass.
    """

    output_alpha: float = 0.75
    mag_update_threshold: float = 1500.0
    transition_width_mg: float = 200.0
    mag_to_gyro_matrix: tuple[tuple[float, float, float], ...] = tuple(
        tuple(float(value) for value in row) for row in PRIMARY_MAG_TO_GYRO
    )

    @staticmethod
    def _low_field_weight(
        magnitude: np.ndarray,
        threshold: float,
        transition_width: float,
    ) -> np.ndarray:
        magnitude = np.asarray(magnitude, dtype=float)
        if transition_width == 0.0:
            return (magnitude <= threshold).astype(float)
        lower = threshold - 0.5 * transition_width
        return np.clip((lower + transition_width - magnitude) / transition_width, 0.0, 1.0)

    def run(self, ws: Workspace) -> None:
        if len(self.inputs) != 8:
            raise ValueError(
                "MagNuisanceFullRateCorrection expects full-rate mag, gyro, "
                "initial solved travel, scalar-mag travel, low-rate corrected "
                "travel, body field, world field, and the learned XYZ path"
            )
        if len(self.outputs) != 6:
            raise ValueError(
                "MagNuisanceFullRateCorrection requires six outputs"
            )

        mag_ts: TimeSeries = ws[self.inputs[0]]
        gyro_ts: TimeSeries = ws[self.inputs[1]]
        initial_ts: TimeSeries = ws[self.inputs[2]]
        scalar_travel_ts: TimeSeries = ws[self.inputs[3]]
        low_travel_ts: TimeSeries = ws[self.inputs[4]]
        body_ts: TimeSeries = ws[self.inputs[5]]
        world_ts: TimeSeries = ws[self.inputs[6]]
        xyz_path = np.asarray(ws[self.inputs[7]], dtype=float)

        full_series = (gyro_ts, initial_ts, scalar_travel_ts)
        for series in full_series:
            if len(series.t) != len(mag_ts.t):
                raise ValueError("Full-rate nuisance inputs must be index-aligned")
        for series in (body_ts, world_ts):
            if len(series.t) != len(low_travel_ts.t) or not np.allclose(
                series.t, low_travel_ts.t, rtol=0.0, atol=1e-9
            ):
                raise ValueError("Low-rate nuisance states must be time-aligned")
        if xyz_path.ndim != 2 or xyz_path.shape[1] != 4 or len(xyz_path) < 2:
            raise ValueError("XYZ path must have columns [travel, x, y, z]")
        if not np.all(np.diff(xyz_path[:, 0]) > 0):
            raise ValueError("XYZ-path travel must be strictly increasing")

        output_alpha = float(self.param(ws, "output_alpha"))
        threshold = float(self.param(ws, "mag_update_threshold"))
        transition_width = float(self.param(ws, "transition_width_mg"))
        if not 0.0 <= output_alpha <= 1.0:
            raise ValueError("output_alpha must be between zero and one")
        if threshold <= 0.0 or transition_width < 0.0:
            raise ValueError("threshold must be positive and transition width nonnegative")

        full_time = np.asarray(mag_ts.t, dtype=float)
        state_time = np.asarray(low_travel_ts.t, dtype=float)
        if len(full_time) < 2 or not np.all(np.diff(full_time) > 0):
            raise ValueError("Full-rate timeline must be strictly increasing")
        state_index = np.searchsorted(full_time, state_time)
        state_index = np.clip(state_index, 0, len(full_time) - 1)
        if not np.allclose(full_time[state_index], state_time, rtol=0.0, atol=1e-8):
            raise ValueError("Low-rate state times must be samples of the full timeline")

        mag_to_gyro = np.asarray(
            self.param(ws, "mag_to_gyro_matrix"), dtype=float
        )
        if mag_to_gyro.shape != (3, 3) or not np.allclose(
            mag_to_gyro @ mag_to_gyro.T, np.eye(3), atol=1e-6
        ):
            raise ValueError("mag_to_gyro_matrix must be orthonormal and 3x3")
        mag_xyz = np.asarray(mag_ts.x, dtype=float) @ mag_to_gyro.T
        gyro_dps = np.asarray(gyro_ts.x, dtype=float)
        full_rotations = integrate_gyro(full_time, gyro_dps)
        body_full, world_full = interpolate_nuisance_fields(
            full_time,
            state_time,
            full_rotations,
            full_rotations[state_index],
            np.asarray(body_ts.x, dtype=float),
            np.asarray(world_ts.x, dtype=float),
        )
        field_correction = body_full + world_full
        corrected_xyz = mag_xyz - field_correction

        travel_grid = xyz_path[:, 0]
        xyz_grid = xyz_path[:, 1:]
        inferred_travel = travel_grid[cKDTree(xyz_grid).query(corrected_xyz)[1]]
        initial_travel = np.asarray(initial_ts.x, dtype=float).reshape(-1)
        scalar_travel = np.asarray(scalar_travel_ts.x, dtype=float).reshape(-1)
        expected_xyz = np.column_stack(
            [
                np.interp(initial_travel, travel_grid, xyz_grid[:, axis])
                for axis in range(3)
            ]
        )
        expected_weight = self._low_field_weight(
            np.linalg.norm(expected_xyz, axis=1), threshold, transition_width
        )
        measured_weight = self._low_field_weight(
            np.linalg.norm(mag_xyz, axis=1), threshold, transition_width
        )
        covered = (initial_travel >= travel_grid[0]) & (
            initial_travel <= travel_grid[-1]
        )
        confidence = np.minimum(expected_weight, measured_weight) * covered
        corrected_mag_travel = scalar_travel + output_alpha * confidence * (
            inferred_travel - scalar_travel
        )

        sampled_initial = initial_travel[state_index]
        low_delta = np.asarray(low_travel_ts.x, dtype=float).reshape(-1) - sampled_initial
        full_delta = np.interp(full_time, state_time, low_delta)
        delta_lifted = initial_travel + full_delta

        meta = {
            **mag_ts.meta,
            "fs_hz": 1.0 / float(np.median(np.diff(full_time))),
            "nuisance_state_hz": low_travel_ts.meta.get("fs_hz"),
            "output_alpha": output_alpha,
            "mag_update_threshold_mg": threshold,
            "transition_width_mg": transition_width,
        }

        def full_series_out(values: np.ndarray, units: str, frame: str) -> TimeSeries:
            return TimeSeries(full_time, np.asarray(values), units, frame, meta)

        ws[self.outputs[0]] = full_series_out(delta_lifted, "mm", "travel")
        ws[self.outputs[1]] = full_series_out(
            corrected_mag_travel, "mm", "travel"
        )
        ws[self.outputs[2]] = full_series_out(
            corrected_xyz, "milli-Gauss", "gyro1"
        )
        ws[self.outputs[3]] = full_series_out(
            field_correction, "milli-Gauss", "gyro1"
        )
        ws[self.outputs[4]] = full_series_out(confidence, "ratio", "")
        ws[self.outputs[5]] = np.array(
            [
                float(np.mean(confidence > 0.0)),
                float(np.mean(confidence)),
                float(np.sqrt(np.mean(full_delta**2))),
                float(np.sqrt(np.mean((corrected_mag_travel - scalar_travel) ** 2))),
            ]
        )
