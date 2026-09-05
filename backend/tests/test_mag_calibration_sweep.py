from __future__ import annotations

from pathlib import Path
import os
import sys
from types import SimpleNamespace
import unittest

import numpy as np


os.environ["MPLCONFIGDIR"] = "/private/tmp"

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from mag_calibration import ResolvedWindow
from mag_calibration_experiment import score_prediction
from mag_calibration_sweep import nested_windows, stable_uniform


class SweepScheduleTests(unittest.TestCase):
    def test_stable_randomization_depends_only_on_identity(self):
        first = stable_uniform(1, 20260904, "log-a", 3)
        self.assertEqual(first, stable_uniform(1, 20260904, "log-a", 3))
        self.assertNotEqual(first, stable_uniform(1, 20260904, "log-a", 4))

    def test_windows_share_a_center_and_are_nested(self):
        center, windows = nested_windows(
            300.0,
            [5.0, 20.0, 120.0],
            random_fraction=0.25,
        )
        self.assertGreaterEqual(windows[0][1], windows[1][1])
        self.assertGreaterEqual(windows[1][1], windows[2][1])
        self.assertLessEqual(windows[0][2], windows[1][2])
        self.assertLessEqual(windows[1][2], windows[2][2])
        for _, start, stop in windows:
            self.assertAlmostEqual((start + stop) / 2.0, center)
        self.assertGreaterEqual(windows[-1][1], 0.0)
        self.assertLessEqual(windows[-1][2], 300.0)


class DistributionAwareMetricTests(unittest.TestCase):
    def test_local_and_fixed_alignment_are_reported_separately(self):
        target = SimpleNamespace(
            time_s=np.arange(3, dtype=float),
            travel=np.array([0.0, 2.0, 4.0]),
            activity_mask=np.ones(3, dtype=bool),
        )
        resolved = ResolvedWindow(0, 3, 0.0, 3.0, 3.0)
        score = score_prediction(
            np.array([1.0, 3.0, 5.0]),
            target,
            resolved,
            fixed_alignment_offset_mm=0.0,
        )
        self.assertAlmostEqual(score["aligned_rmse"], 0.0)
        self.assertAlmostEqual(score["fixed_aligned_rmse"], 1.0)
        self.assertAlmostEqual(score["travel_range"], 4.0)
        self.assertAlmostEqual(score["travel_std"], np.std(target.travel))
        self.assertAlmostEqual(score["fixed_aligned_nrmse_std"], 1.0 / np.std(target.travel))
        self.assertEqual(score["bin_occupied_count"], 1)


if __name__ == "__main__":
    unittest.main()
