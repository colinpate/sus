from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from angle import FindBoringRegions
from classes.time_series import TimeSeries


def make_workspace(values: list[float]) -> dict[str, object]:
    samples = np.asarray(values, dtype=float)
    return {
        "travel": TimeSeries(
            t=np.arange(len(samples), dtype=float),
            x=samples,
            units="mm",
        )
    }


class FindBoringRegionsTests(unittest.TestCase):
    def test_closes_trailing_static_region(self):
        ws = make_workspace([0, 0, 0, 0, 20, 30, 30, 30, 30, 30])
        step = FindBoringRegions(
            name="find_boring_regions",
            inputs=("travel",),
            outputs=("regions", "active_mask", "boring_mask"),
            travel_delta_threshold=10,
            max_travel=200,
            min_region_len_samp=4,
            padding=1,
        )

        step.run(ws)

        np.testing.assert_array_equal(ws["regions"], np.array([(1, 4), (6, 9)]))
        np.testing.assert_array_equal(
            ws["active_mask"],
            np.array([True, False, False, False, True, True, False, False, False, True]),
        )
        np.testing.assert_array_equal(ws["boring_mask"], ws["active_mask"])

    def test_does_not_close_short_trailing_region(self):
        ws = make_workspace([0, 0, 0, 0, 20, 30, 30])
        step = FindBoringRegions(
            name="find_boring_regions",
            inputs=("travel",),
            outputs=("regions", "active_mask", "boring_mask"),
            travel_delta_threshold=10,
            max_travel=200,
            min_region_len_samp=4,
            padding=1,
        )

        step.run(ws)

        np.testing.assert_array_equal(ws["regions"], np.array([(1, 4)]))
        np.testing.assert_array_equal(
            ws["active_mask"],
            np.array([True, False, False, False, True, True, True]),
        )
        np.testing.assert_array_equal(ws["boring_mask"], ws["active_mask"])

    def test_does_not_close_trailing_region_above_max_travel(self):
        ws = make_workspace([0, 0, 0, 0, 20, 250])
        step = FindBoringRegions(
            name="find_boring_regions",
            inputs=("travel",),
            outputs=("regions", "active_mask", "boring_mask"),
            travel_delta_threshold=10,
            max_travel=200,
            min_region_len_samp=1,
            padding=0,
        )

        step.run(ws)

        np.testing.assert_array_equal(ws["regions"], np.array([(0, 5)]))
        np.testing.assert_array_equal(
            ws["active_mask"],
            np.array([False, False, False, False, False, True]),
        )
        np.testing.assert_array_equal(ws["boring_mask"], ws["active_mask"])


if __name__ == "__main__":
    unittest.main()
