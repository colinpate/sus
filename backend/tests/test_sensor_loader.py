from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from classes.sensor_loader import AngleLoader


class AngleLoaderTests(unittest.TestCase):
    def write_angle_csv(self, raw_counts: list[int]) -> str:
        handle = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        handle.close()
        pd.DataFrame(
            {
                "t_s": np.arange(len(raw_counts), dtype=float) * 0.01,
                "angle_raw": raw_counts,
            }
        ).to_csv(handle.name, index=False)
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        return handle.name

    def test_unwraps_encoder_boundary_before_downstream_filtering(self):
        path = self.write_angle_csv([4094, 4095, 0, 1, 2])

        ws = AngleLoader(
            path=path,
            lag=0,
            interpolate_bad=False,
            mark_bad_samples=False,
        ).load()

        angle = ws["angle"].x[:, 0]
        one_count_rad = 2 * np.pi / 4096
        np.testing.assert_allclose(np.diff(angle), one_count_rad, atol=1e-12)
        self.assertAlmostEqual(float(np.ptp(angle)), 4 * one_count_rad)
        self.assertTrue(ws["angle"].meta["angle_unwrapped"])

    def test_can_disable_unwrapping_for_compatibility(self):
        path = self.write_angle_csv([4095, 0])

        ws = AngleLoader(
            path=path,
            lag=0,
            interpolate_bad=False,
            mark_bad_samples=False,
            unwrap=False,
        ).load()

        self.assertLess(float(np.diff(ws["angle"].x[:, 0])[0]), -6.0)

    def test_fractional_time_lag_interpolates_without_wrapping(self):
        path = self.write_angle_csv([0, 100, 200, 300])

        ws = AngleLoader(
            path=path,
            lag_s=-0.005,
            interpolate_bad=False,
            mark_bad_samples=False,
        ).load()

        count_rad = 2 * np.pi / 4096
        np.testing.assert_allclose(
            ws["angle"].x[:, 0],
            np.array([0.0, 50.0, 150.0, 250.0]) * count_rad,
        )
        np.testing.assert_array_equal(
            ws["angle/bad_mask"].x[:, 0],
            np.array([True, False, False, False]),
        )
        self.assertEqual(ws["angle"].meta["lag_s"], -0.005)

    def test_integer_lag_uses_interpolation_compatibility_path(self):
        path = self.write_angle_csv([10, 20, 30])

        ws = AngleLoader(
            path=path,
            lag=-1,
            interpolate_bad=False,
            mark_bad_samples=False,
        ).load()

        count_rad = 2 * np.pi / 4096
        np.testing.assert_allclose(
            ws["angle"].x[:, 0],
            np.array([10.0, 10.0, 20.0]) * count_rad,
        )
        np.testing.assert_array_equal(
            ws["angle/bad_mask"].x[:, 0],
            np.array([True, False, False]),
        )


if __name__ == "__main__":
    unittest.main()
