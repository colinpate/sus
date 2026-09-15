from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from classes.time_series import TimeSeries
from classes.runner import Runner
from mag_nuisance import (
    MAG_NUISANCE_SUMMARY_FIELDS,
    MagNuisanceFullRateCorrection,
    MagNuisanceTravelCorrection,
)
from mag_nuisance_core import (
    fit_scalar_parameterized_xyz,
    integrate_gyro,
    interpolate_nuisance_fields,
    invert_scalar_travel_model,
    predict_relative_scalar_travel,
)


OUTPUTS = (
    "travel/corrected",
    "mag/body",
    "mag/world",
    "mag/xyz_path",
    "mag/summary",
)

FULL_RATE_OUTPUTS = (
    "travel/delta_lifted",
    "travel/mag_corrected",
    "mag/corrected_norm",
)


class MagNuisanceTravelCorrectionTests(unittest.TestCase):
    def test_emits_low_rate_non_destructive_correction(self):
        sample_count = 1000
        time_s = np.arange(sample_count, dtype=float) / 100.0
        phase = np.linspace(0.0, 8.0 * np.pi, sample_count)
        initial_travel = 100.0 * (1.0 - np.cos(phase))
        coefficients = np.array([700.0, 0.2, 0.5])
        scalar_mag = invert_scalar_travel_model(initial_travel, coefficients)
        normalized = (scalar_mag - np.median(scalar_mag)) / np.std(scalar_mag)
        mag_xyz = np.column_stack(
            (
                900.0 + 250.0 * normalized + 20.0 * normalized**2,
                -200.0 + 120.0 * normalized - 15.0 * normalized**2,
                300.0 - 80.0 * normalized + 10.0 * normalized**2,
            )
        )
        mag_xyz += 15.0 * np.column_stack(
            (np.sin(phase / 5.0), np.cos(phase / 7.0), np.sin(phase / 9.0))
        )

        series_meta = {"fs_hz": 100.0}
        ws = {
            "mag": TimeSeries(
                time_s, mag_xyz, "milli-Gauss", "gyro1", series_meta
            ),
            "gyro": TimeSeries(
                time_s,
                np.zeros((sample_count, 3)),
                "deg/s",
                "gyro1",
                series_meta,
            ),
            "scalar": TimeSeries(
                time_s, scalar_mag, "milli-Gauss", "", series_meta
            ),
            "coefficients": coefficients,
            "initial": TimeSeries(time_s, initial_travel, "mm", "travel", series_meta),
        }
        original_initial = ws["initial"].x.copy()
        step = MagNuisanceTravelCorrection(
            name="test_mag_nuisance",
            inputs=("mag", "gyro", "scalar", "coefficients", "initial"),
            outputs=OUTPUTS,
            min_bin_samples=1,
            mag_to_gyro_matrix=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
        )

        step.run(ws)

        self.assertEqual(ws["travel/corrected"].x.shape, (100, 1))
        self.assertEqual(ws["mag/body"].x.shape, (100, 3))
        self.assertEqual(ws["mag/world"].x.shape, (100, 3))
        xyz_path = ws["mag/xyz_path"]
        self.assertEqual(xyz_path.shape[1], 4)
        summary = dict(zip(MAG_NUISANCE_SUMMARY_FIELDS, ws["mag/summary"]))
        self.assertLessEqual(
            summary["credible_travel_max_mm"]
            - summary["credible_travel_min_mm"],
            200.0 + 1e-9,
        )
        self.assertAlmostEqual(
            summary["credible_travel_min_mm"]
            - summary["projection_path_min_mm"],
            10.0,
        )
        self.assertAlmostEqual(
            summary["projection_path_max_mm"]
            - summary["credible_travel_max_mm"],
            10.0,
        )
        self.assertEqual(
            ws["mag/summary"].shape,
            (len(MAG_NUISANCE_SUMMARY_FIELDS),),
        )
        self.assertTrue(np.all(np.isfinite(ws["mag/summary"])))
        self.assertTrue(np.all(np.isfinite(ws["travel/corrected"].x)))
        self.assertAlmostEqual(
            ws["travel/corrected"].meta["gyro_integration_hz"], 100.0
        )
        np.testing.assert_array_equal(ws["initial"].x, original_initial)

        sampled_initial = initial_travel[::10]
        blended = ws["travel/corrected"].x[:, 0]

        ws["scalar_travel"] = TimeSeries(
            time_s, initial_travel, "mm", "travel", series_meta
        )
        full_step = MagNuisanceFullRateCorrection(
            name="test_mag_nuisance_full_rate",
            inputs=(
                "mag",
                "gyro",
                "initial",
                "scalar_travel",
                "travel/corrected",
                "mag/body",
                "mag/world",
                "mag/xyz_path",
            ),
            outputs=FULL_RATE_OUTPUTS,
            transition_width_mg=0.0,
            mag_to_gyro_matrix=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            Runner(
                out_dir=Path(temp_dir),
                make_plots=False,
            ).run(ws, [full_step])
            with np.load(Path(temp_dir) / "cache" / "all.npz") as artifact:
                cached_norm = artifact["mag/corrected_norm__x"]
                self.assertEqual(cached_norm.shape, (sample_count, 1))

        expected_delta = np.interp(
            time_s,
            ws["travel/corrected"].t,
            blended - sampled_initial,
        )
        np.testing.assert_allclose(
            ws["travel/delta_lifted"].x[:, 0],
            initial_travel + expected_delta,
        )
        self.assertEqual(ws["travel/mag_corrected"].x.shape, (sample_count, 1))
        self.assertTrue(np.all(np.isfinite(ws["travel/mag_corrected"].x)))
        full_rotations = integrate_gyro(time_s, ws["gyro"].x)
        state_indices = np.arange(0, sample_count, 10)
        body_full, world_full = interpolate_nuisance_fields(
            time_s,
            ws["travel/corrected"].t,
            full_rotations,
            full_rotations[state_indices],
            ws["mag/body"].x,
            ws["mag/world"].x,
        )
        expected_corrected_norm = np.linalg.norm(
            mag_xyz - body_full - world_full, axis=1
        )
        corrected_norm = ws["mag/corrected_norm"]
        np.testing.assert_allclose(corrected_norm.t, time_s)
        np.testing.assert_allclose(
            corrected_norm.x[:, 0], expected_corrected_norm
        )
        self.assertEqual(corrected_norm.units, "milli-Gauss")
        self.assertEqual(corrected_norm.frame, "gyro1")
        self.assertAlmostEqual(corrected_norm.meta["fs_hz"], 100.0)

    def test_xyz_path_uses_best_supported_bounded_relative_window(self):
        coefficients = np.array([700.0, 0.2, 0.5])
        supported_travel = np.repeat(np.linspace(-20.0, 160.0, 25), 8)
        invalid_travel = np.repeat(np.linspace(260.0, 300.0, 4), 8)
        travel = np.concatenate((supported_travel, invalid_travel))
        scalar = invert_scalar_travel_model(travel, coefficients)
        relative = predict_relative_scalar_travel(scalar, coefficients)
        xyz = np.column_stack(
            (scalar, 0.25 * scalar + 20.0, -0.1 * scalar + 5.0)
        )

        model = fit_scalar_parameterized_xyz(
            scalar,
            xyz,
            coefficients,
            scalar_bin_mg=20.0,
            degree=1,
            min_bin_samples=2,
            travel_max_mm=200.0,
            path_margin_mm=10.0,
        )

        self.assertLessEqual(
            model.credible_travel_max - model.credible_travel_min,
            200.0 + 1e-9,
        )
        self.assertLess(model.credible_travel_max, np.min(relative[-32:]))
        self.assertAlmostEqual(
            model.credible_travel_min - model.travel_min, 10.0
        )
        self.assertAlmostEqual(
            model.travel_max - model.credible_travel_max, 10.0
        )

    def test_full_rate_projection_does_not_require_initial_travel_coverage(self):
        time_s = np.arange(11, dtype=float) / 10.0
        path_travel = np.linspace(-10.0, 20.0, 31)
        path_xyz = np.column_stack(
            (path_travel, np.zeros_like(path_travel), np.zeros_like(path_travel))
        )
        true_relative_travel = np.arange(11, dtype=float)
        mag_xyz = np.column_stack(
            (
                true_relative_travel,
                np.zeros_like(true_relative_travel),
                np.zeros_like(true_relative_travel),
            )
        )
        meta = {"fs_hz": 10.0}
        zeros_xyz = np.zeros((2, 3))
        ws = {
            "mag": TimeSeries(time_s, mag_xyz, "milli-Gauss", "gyro1", meta),
            "gyro": TimeSeries(
                time_s, np.zeros((11, 3)), "deg/s", "gyro1", meta
            ),
            "initial": TimeSeries(
                time_s, np.full(11, 500.0), "mm", "travel", meta
            ),
            "scalar": TimeSeries(time_s, np.zeros(11), "mm", "travel", meta),
            "low": TimeSeries(
                time_s[[0, -1]], np.full(2, 500.0), "mm", "travel", meta
            ),
            "body": TimeSeries(
                time_s[[0, -1]], zeros_xyz, "milli-Gauss", "gyro1", meta
            ),
            "world": TimeSeries(
                time_s[[0, -1]], zeros_xyz, "milli-Gauss", "gyro1", meta
            ),
            "path": np.column_stack((path_travel, path_xyz)),
        }
        step = MagNuisanceFullRateCorrection(
            name="test_outside_projection",
            inputs=(
                "mag",
                "gyro",
                "initial",
                "scalar",
                "low",
                "body",
                "world",
                "path",
            ),
            outputs=FULL_RATE_OUTPUTS,
            output_alpha=1.0,
            transition_width_mg=0.0,
            path_distance_transition_mg=0.0,
            mag_to_gyro_matrix=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
        )

        step.run(ws)

        np.testing.assert_allclose(
            ws["travel/mag_corrected"].x[:, 0], true_relative_travel
        )

    def test_world_field_interpolation_uses_full_rate_rotation(self):
        full_time = np.linspace(0.0, 1.0, 101)
        state_time = full_time[[0, 50, 100]]
        gyro = np.zeros((len(full_time), 3))
        gyro[:, 2] = 90.0
        rotations = integrate_gyro(full_time, gyro)
        state_index = np.array([0, 50, 100])
        world_reference = np.array([100.0, 20.0, -10.0])
        state_world = np.einsum(
            "i,nij->nj", world_reference, rotations[state_index]
        )
        body, world = interpolate_nuisance_fields(
            full_time,
            state_time,
            rotations,
            rotations[state_index],
            np.zeros((3, 3)),
            state_world,
        )

        reconstructed_reference = np.einsum(
            "ni,nji->nj", world, rotations
        )
        np.testing.assert_allclose(body, 0.0, atol=1e-12)
        np.testing.assert_allclose(
            reconstructed_reference,
            np.broadcast_to(world_reference, reconstructed_reference.shape),
            atol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()
