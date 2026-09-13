from __future__ import annotations

import unittest

import numpy as np

from tools.front.mag_offset_calibration.tabulate_low_travel_mag_stats import (
    range_stats,
    reconstruct_corrected_magnitude,
)
from tools.front.mag_offset_calibration.evaluate_under5_solved_offset import (
    offset_from_detector,
)


class LowTravelMagStatsTests(unittest.TestCase):
    def test_range_stats_uses_strict_travel_bounds(self) -> None:
        count, mean, std = range_stats(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([0.0, 1.0, 4.0, 5.0]),
            0.0,
            5.0,
        )
        self.assertEqual(count, 2)
        self.assertAlmostEqual(mean, 2.5)
        self.assertAlmostEqual(std, 0.5)

    def test_corrected_magnitude_matches_full_rate_frame_math(self) -> None:
        time = np.array([0.0, 0.1, 0.2])
        cache = {
            "mag/lpf__t": time,
            # PRIMARY_MAG_TO_GYRO maps each row to [3, 4, 0].
            "mag/lpf__x": np.tile(np.array([0.0, -4.0, 3.0]), (3, 1)),
            "gyro/lpf/gyro1__x": np.zeros((3, 3)),
            "mag/nuisance/body/10hz__t": time,
            "mag/nuisance/body/10hz__x": np.tile(
                np.array([1.0, 0.0, 0.0]), (3, 1)
            ),
            "mag/nuisance/world/10hz__x": np.zeros((3, 3)),
        }
        np.testing.assert_allclose(
            reconstruct_corrected_magnitude(cache),
            np.sqrt(20.0),
        )

    def test_under5_detector_offset_targets_two_point_five_mm(self) -> None:
        solved = np.array([1.0, 3.0, 100.0])
        detector = np.array([True, True, False])
        offset = offset_from_detector(solved, detector)
        self.assertAlmostEqual(offset, 0.5)
        self.assertAlmostEqual(np.mean((solved + offset)[detector]), 2.5)


if __name__ == "__main__":
    unittest.main()
