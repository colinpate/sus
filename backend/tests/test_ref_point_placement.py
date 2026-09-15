from __future__ import annotations

import unittest

from tools.front.mag_offset_calibration.compare_ref_point_placement import (
    aggregate_validation_parents,
    max_offset_delta,
)


class PlacementAggregationTests(unittest.TestCase):
    def test_validation_parents_receive_equal_weight(self) -> None:
        rows = []
        for log, parent, rmse in (
            ("chunk-a", "parent-a", 2.0),
            ("chunk-b", "parent-a", 4.0),
            ("chunk-c", "parent-b", 9.0),
        ):
            rows.append(
                {
                    "split": "validation",
                    "stage": "final_solver",
                    "variant": "post",
                    "log": log,
                    "parent_log": parent,
                    "mean_error_mm": rmse,
                    "uncentered_rmse_mm": rmse,
                    "low_uncentered_rmse_mm": rmse,
                    "centered_rmse_mm": rmse,
                }
            )

        aggregate = aggregate_validation_parents(rows)

        self.assertEqual(len(aggregate), 1)
        self.assertEqual(aggregate[0]["n_parents"], 2)
        self.assertEqual(aggregate[0]["uncentered_rmse_mm"], 6.0)

    def test_offset_cap_uses_the_step_at_the_selected_placement(self) -> None:
        config = {
            "steps": {
                "mag_to_travel_model": {"ref_max_offset_delta_mm": 20.0},
                "apply_mag_travel_ref_point": {"ref_max_offset_delta_mm": 40.0},
            }
        }

        self.assertEqual(max_offset_delta(config, "pre"), 20.0)
        self.assertEqual(max_offset_delta(config, "post"), 40.0)


if __name__ == "__main__":
    unittest.main()
