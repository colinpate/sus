from __future__ import annotations

from pathlib import Path
import os
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd


os.environ["MPLCONFIGDIR"] = "/private/tmp"

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from mag_calibration import ResolvedWindow
from analyze_mag_calibration_cross_setup import crossed_bootstrap, source_balanced_median
from mag_calibration_experiment import score_prediction
from mag_calibration_solver_sweep import endpoint_change
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
        self.assertAlmostEqual(score["aligned_mean_error"], 0.0)
        self.assertAlmostEqual(score["anchored_mean_error"], 1.0)
        self.assertAlmostEqual(score["fixed_aligned_rmse"], 1.0)
        self.assertAlmostEqual(score["fixed_aligned_mean_error"], 1.0)
        self.assertAlmostEqual(score["bin0_anchored_mean_error"], 1.0)
        self.assertAlmostEqual(score["bin0_fixed_mean_error"], 1.0)
        self.assertAlmostEqual(score["bin0_mean_error"], 0.0)
        self.assertAlmostEqual(score["travel_range"], 4.0)
        self.assertAlmostEqual(score["travel_std"], np.std(target.travel))
        self.assertAlmostEqual(score["fixed_aligned_nrmse_std"], 1.0 / np.std(target.travel))
        self.assertEqual(score["bin_occupied_count"], 1)

    def test_solver_endpoint_change_collapses_repeats_within_log(self):
        rows = []
        values = {
            ("log-a", 0): (5.0, 4.0),
            ("log-a", 1): (5.0, 2.0),
            ("log-b", 0): (5.0, 6.0),
            ("log-b", 1): (7.0, 8.0),
        }
        for (log_name, repeat), (short, long) in values.items():
            for duration, value in ((5.0, short), (120.0, long)):
                rows.append({
                    "log": log_name,
                    "repeat": repeat,
                    "duration_s": duration,
                    "evaluation_scope": "full_log",
                    "stage": "solved",
                    "aligned_rmse": value,
                })
        change = endpoint_change(
            rows,
            scope="full_log",
            stage="solved",
            metric="aligned_rmse",
        )
        self.assertEqual(change["n"], 2)
        self.assertAlmostEqual(change["median"], -0.5)
        self.assertAlmostEqual(change["fraction_improved"], 0.5)


class CrossSetupAnalysisTests(unittest.TestCase):
    def test_source_balancing_prevents_pair_count_from_dominating(self):
        frame = pd.DataFrame({
            "train_log": ["source-a", "source-a", "source-b", "source-b"],
            "value": [0.0, 100.0, 10.0, 10.0],
        })
        self.assertAlmostEqual(source_balanced_median(frame, "value"), 30.0)

    def test_crossed_bootstrap_handles_excluded_diagonal_cells(self):
        frame = pd.DataFrame({
            "train_log": ["a", "a", "b", "b"],
            "eval_log": ["a", "b", "a", "b"],
        })
        values = np.array([np.nan, 1.0, 2.0, np.nan])
        low, high = crossed_bootstrap(
            frame,
            values,
            draws=100,
            rng=np.random.default_rng(1),
        )
        self.assertTrue(np.isfinite(low))
        self.assertTrue(np.isfinite(high))
        self.assertLessEqual(low, high)


if __name__ == "__main__":
    unittest.main()
