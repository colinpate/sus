from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from classes.time_series import TimeSeries
from fusion import GetErrorStats


def make_series(values: list[float], *, t: np.ndarray | None = None) -> TimeSeries:
    if t is None:
        t = np.arange(len(values), dtype=float)
    return TimeSeries(t=t, x=np.asarray(values, dtype=float))


class GetErrorStatsTests(unittest.TestCase):
    def test_excludes_corrupted_angle_samples(self):
        t = np.arange(4, dtype=float)
        ws = {
            "pred": make_series([11.0, 22.0, 103.0, 44.0], t=t),
            "gt": make_series([10.0, 20.0, 100.0, 40.0], t=t),
            "boring_mask": np.ones(4, dtype=bool),
            "angle/bad_mask": TimeSeries(
                t=t,
                x=np.array([False, False, True, False]),
                units="bool",
            ),
        }
        step = GetErrorStats(
            name="test_error_stats",
            inputs=("pred", "gt", "boring_mask"),
            outputs=(),
            gt_thresh=0,
        )

        with patch("fusion.print_err_stats") as print_stats:
            step.run(ws)

        self.assertEqual(print_stats.call_count, 2)
        for call in print_stats.call_args_list:
            np.testing.assert_array_equal(call.args[0], np.array([11.0, 22.0, 44.0]))
            np.testing.assert_array_equal(call.args[1], np.array([10.0, 20.0, 40.0]))


if __name__ == "__main__":
    unittest.main()
