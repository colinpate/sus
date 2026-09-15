from __future__ import annotations

import unittest

import numpy as np

from tools.front.mag_offset_calibration.sweep_ref_point_parameters import (
    Config,
    RefChunk,
    aggregate_solver_rows,
    select_reference_points,
)


class ReferenceSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            RefChunk(
                direction="forward",
                mag_bump=np.array([2500.0, 3000.0, 4500.0]),
                rel_x_mm=np.array([10.0, 30.0, 80.0]),
                truth_bump_mm=np.array([11.0, 31.0, 81.0]),
                start_mag_mg=1000.0,
                start_truth_mm=0.0,
            )
        ]

    def test_current_mag_band_keeps_only_points_near_magnetic_center(self) -> None:
        x, mag, truth = select_reference_points(
            self.chunks,
            baseline_mg=1000.0,
            config=Config(name="current"),
        )

        np.testing.assert_array_equal(x, [10.0, 30.0])
        np.testing.assert_array_equal(mag, [2500.0, 3000.0])
        np.testing.assert_array_equal(truth, [11.0, 31.0])

    def test_displacement_selector_is_distinct_from_detection_threshold(self) -> None:
        x, mag, _ = select_reference_points(
            self.chunks,
            baseline_mg=1000.0,
            config=Config(
                name="x-only",
                selector="x",
                x_min_mm=20.0,
                x_max_mm=60.0,
            ),
        )

        np.testing.assert_array_equal(x, [30.0])
        np.testing.assert_array_equal(mag, [3000.0])


class AggregateSolverRowsTests(unittest.TestCase):
    def test_reused_csv_boolean_values_are_aggregated(self) -> None:
        rows = [
            {
                "split": "validation",
                "subgroup": "slayer",
                "config": "candidate",
                "mean_error_mm": "1.0",
                "uncentered_rmse_mm": "2.0",
                "low_uncentered_rmse_mm": "3.0",
                "centered_rmse_mm": "1.5",
                "success": "True",
                "nfev": "4",
            }
        ]

        aggregate = aggregate_solver_rows(rows)
        cohort = next(row for row in aggregate if row["scope"] == "cohort-balanced")

        self.assertEqual(cohort["success"], 1.0)
        self.assertEqual(cohort["uncentered_rmse_mm"], 2.0)


if __name__ == "__main__":
    unittest.main()
