from __future__ import annotations

from pathlib import Path
import os
import sys
import unittest


os.environ["MPLCONFIGDIR"] = "/private/tmp"

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

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


if __name__ == "__main__":
    unittest.main()
