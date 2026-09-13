from __future__ import annotations

import unittest

import numpy as np

from tools.front.mag_offset_calibration.sweep_post_mag_correction_fusion import (
    aggregate_rows,
    centered_error,
    select_floor,
)
from tools.front.mag_offset_calibration.sweep_post_mag_correction_threshold import (
    select_mode,
)


class PostMagCorrectionFusionSweepTest(unittest.TestCase):
    def test_centered_error_uses_one_full_mask_offset(self) -> None:
        truth = np.array([0.0, 10.0, 20.0, 30.0])
        prediction = np.array([3.0, 14.0, 25.0, 36.0])
        mask = np.array([True, True, True, False])

        error = centered_error(prediction, truth, mask)

        np.testing.assert_allclose(error[:3], [-1.0, 0.0, 1.0])
        self.assertTrue(np.isnan(error[3]))

    def test_aggregate_weights_cohorts_equally(self) -> None:
        rows = []
        for cohort, values in (("large", [2.0, 4.0]), ("small", [10.0])):
            for index, value in enumerate(values):
                rows.append(
                    {
                        "split": "tuning",
                        "cohort": cohort,
                        "log": f"{cohort}-{index}",
                        "mag_off_floor": 0.1,
                        "success": True,
                        "corrected_mag_centered_rmse_mm": value,
                        "centered_rmse_mm": value,
                        "low_travel_centered_rmse_mm": value,
                        "uncentered_rmse_mm": value,
                        "runtime_s": 1.0,
                    }
                )

        aggregate = aggregate_rows(rows)
        balanced = next(row for row in aggregate if row["cohort"] == "cohort-balanced")

        self.assertEqual(balanced["n_logs"], 3)
        self.assertAlmostEqual(balanced["low_travel_centered_rmse_mm"], 6.5)

    def test_selection_rejects_partially_converged_candidate(self) -> None:
        aggregate = [
            {
                "split": "tuning",
                "cohort": "cohort-balanced",
                "mag_off_floor": 0.03,
                "success_fraction": 0.9,
                "low_travel_centered_rmse_mm": 1.0,
                "centered_rmse_mm": 1.0,
            },
            {
                "split": "tuning",
                "cohort": "cohort-balanced",
                "mag_off_floor": 0.3,
                "success_fraction": 1.0,
                "low_travel_centered_rmse_mm": 2.0,
                "centered_rmse_mm": 2.0,
            },
        ]

        selected = select_floor(aggregate)

        self.assertEqual(selected["mag_off_floor"], 0.3)

    def test_threshold_selection_requires_every_cohort_to_avoid_regression(self) -> None:
        aggregate = []
        values = {
            "baseline": {"a": (5.0, 4.0), "b": (6.0, 4.5)},
            "aggressive": {"a": (3.0, 3.0), "b": (5.0, 4.6)},
            "robust": {"a": (4.0, 3.5), "b": (5.5, 4.4)},
        }
        for mode, cohorts in values.items():
            for cohort, (low_rmse, overall_rmse) in cohorts.items():
                aggregate.append(
                    {
                        "split": "tuning",
                        "cohort": cohort,
                        "threshold_mode": mode,
                        "success_fraction": 1.0,
                        "low_travel_centered_rmse_mm": low_rmse,
                        "centered_rmse_mm": overall_rmse,
                    }
                )
            aggregate.append(
                {
                    "split": "tuning",
                    "cohort": "cohort-balanced",
                    "threshold_mode": mode,
                    "success_fraction": 1.0,
                    "low_travel_centered_rmse_mm": np.mean(
                        [item[0] for item in cohorts.values()]
                    ),
                    "centered_rmse_mm": np.mean(
                        [item[1] for item in cohorts.values()]
                    ),
                }
            )

        selected, eligible = select_mode(aggregate, ("a", "b"))

        self.assertEqual(selected["threshold_mode"], "robust")
        self.assertEqual(eligible, ["baseline", "robust"])


if __name__ == "__main__":
    unittest.main()
