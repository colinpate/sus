from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from classes.time_series import TimeSeries
from mag_nuisance import (
    MagNuisanceFullRateCorrection,
    MagNuisanceTravelCorrection,
)
from mag_nuisance_core import (
    integrate_gyro,
    interpolate_nuisance_fields,
    invert_scalar_travel_model,
)


OUTPUTS = (
    "travel/corrected",
    "travel/proposal",
    "mag/body",
    "mag/world",
    "mag/correction",
    "mag/corrected_xyz",
    "mag/update_mask",
    "mag/xyz_path",
    "mag/iteration_change",
    "mag/summary",
)

FULL_RATE_OUTPUTS = (
    "travel/delta_lifted",
    "travel/mag_corrected",
    "mag/corrected_full",
    "mag/correction_full",
    "mag/confidence_full",
    "mag/full_summary",
)


class MagNuisanceTravelCorrectionTests(unittest.TestCase):
    def test_emits_low_rate_non_destructive_correction(self):
        sample_count = 1000
        time_s = np.arange(sample_count, dtype=float) / 100.0
        phase = np.linspace(0.0, 8.0 * np.pi, sample_count)
        initial_travel = 100.0 * (1.0 - np.cos(phase))
        coefficients = np.array([700.0, 0.2, 0.5])
        scalar_mag = invert_scalar_travel_model(initial_travel, coefficients, 0.0)
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
            "offset": np.array([0.0]),
            "initial": TimeSeries(time_s, initial_travel, "mm", "travel", series_meta),
        }
        original_initial = ws["initial"].x.copy()
        step = MagNuisanceTravelCorrection(
            name="test_mag_nuisance",
            inputs=("mag", "gyro", "scalar", "coefficients", "offset", "initial"),
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
        self.assertEqual(ws["mag/correction"].x.shape, (100, 3))
        self.assertEqual(ws["mag/xyz_path"].shape, (841, 4))
        self.assertEqual(ws["mag/iteration_change"].shape, (4,))
        self.assertTrue(np.all(np.isfinite(ws["travel/corrected"].x)))
        self.assertAlmostEqual(
            ws["travel/corrected"].meta["gyro_integration_hz"], 100.0
        )
        np.testing.assert_array_equal(ws["initial"].x, original_initial)

        update_mask = ws["mag/update_mask"].x[:, 0].astype(bool)
        sampled_initial = initial_travel[::10]
        proposal = ws["travel/proposal"].x[:, 0]
        blended = ws["travel/corrected"].x[:, 0]
        np.testing.assert_allclose(
            blended,
            sampled_initial + 0.75 * (proposal - sampled_initial),
        )
        np.testing.assert_array_equal(proposal[~update_mask], sampled_initial[~update_mask])

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
        full_step.run(ws)

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
        self.assertEqual(ws["mag/corrected_full"].x.shape, (sample_count, 3))
        self.assertEqual(ws["mag/confidence_full"].x.shape, (sample_count, 1))
        self.assertTrue(np.all(np.isfinite(ws["travel/mag_corrected"].x)))

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
