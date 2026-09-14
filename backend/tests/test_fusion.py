from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from classes.time_series import TimeSeries
from classes.log_config import attach_log_config
from fusion import (
    ApplyMagTravelRefPoint,
    GetErrorStats,
    GetMagBaseline,
    GetMagToTravelModel,
    GetMagTravelRefPoint,
)
from mag_to_travel_model_core import MagToTravelModel


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
            "active_mask": np.ones(4, dtype=bool),
            "angle/bad_mask": TimeSeries(
                t=t,
                x=np.array([False, False, True, False]),
                units="bool",
            ),
        }
        step = GetErrorStats(
            name="test_error_stats",
            inputs=("pred", "gt", "active_mask"),
            outputs=(),
            gt_thresh=0,
        )

        with patch("fusion.print_err_stats") as print_stats:
            step.run(ws)

        self.assertEqual(print_stats.call_count, 2)
        for call in print_stats.call_args_list:
            np.testing.assert_array_equal(call.args[0], np.array([11.0, 22.0, 44.0]))
            np.testing.assert_array_equal(call.args[1], np.array([10.0, 20.0, 40.0]))


class AbsoluteReferenceTests(unittest.TestCase):
    def test_reference_is_applied_to_existing_corrected_mag_travel(self):
        t = np.arange(4, dtype=float)
        meta = {"fs_hz": 100.0}
        ws = {
            "corrected_travel": TimeSeries(
                t=t,
                x=np.array([5.0, 15.0, 25.0, 35.0]),
                units="mm",
                frame="travel",
                meta=meta,
            ),
            "corrected_mag": TimeSeries(
                t=t,
                x=np.array([110.0, 120.0, 130.0, 140.0]),
                units="milli-Gauss",
                frame="gyro1",
                meta=meta,
            ),
            "accel": TimeSeries(
                t=t,
                x=np.array([0.0, 1.0, 2.0, 3.0]),
                units="m/s^2",
                frame="travel",
                meta=meta,
            ),
            "bad_mask": TimeSeries(t=t, x=np.zeros(4, dtype=bool)),
            "reference": np.array([50.0, 120.0]),
            "coefficients": np.array([100.0, 1.0, 1.0]),
        }
        step = ApplyMagTravelRefPoint(
            name="apply_reference",
            inputs=(
                "corrected_travel",
                "corrected_mag",
                "accel",
                "bad_mask",
                "reference",
                "coefficients",
            ),
            outputs=("adjusted",),
        )

        step.run(ws)

        np.testing.assert_allclose(ws["adjusted"].x[:, 0], [35.0, 45.0, 55.0, 65.0])
        self.assertEqual(ws["adjusted"].units, "mm")
        self.assertEqual(ws["adjusted"].frame, "travel")

    def test_fixed_reference_bypasses_estimation(self):
        ws = {}
        attach_log_config(
            ws,
            {"steps": {"reference": {"fixed_reference": [42.0, 3210.0]}}},
        )
        step = GetMagTravelRefPoint(
            name="reference",
            inputs=("mag", "accel", "baseline", "travel"),
            outputs=("reference",),
        )

        step.run(ws)

        np.testing.assert_allclose(ws["reference"], [42.0, 3210.0])

    def test_fixed_baseline_bypasses_estimation(self):
        ws = {}
        attach_log_config(
            ws,
            {"steps": {"baseline": {"fixed_baseline_mG": 1450.0}}},
        )
        step = GetMagBaseline(
            name="baseline",
            inputs=("mag", "accel"),
            outputs=("baseline",),
        )

        step.run(ws)

        np.testing.assert_allclose(ws["baseline"], [1450.0])

    def test_under_supported_reference_uses_finite_fallback(self):
        step = GetMagTravelRefPoint(
            name="reference",
            inputs=(),
            outputs=("reference",),
        )

        ref_x, ref_mag = step.get_abs_pos_ref(
            [np.array([3000.0, 3100.0])],
            [np.array([10.0, 20.0])],
            1000.0,
            min_ref_points=20,
            fallback_ref_mag=1250.0,
        )

        self.assertEqual(ref_x, 0.0)
        self.assertEqual(ref_mag, 1250.0)
        self.assertTrue(np.all(np.isfinite([ref_x, ref_mag])))

    def test_excessive_reference_shift_uses_data_driven_zero(self):
        step = GetMagToTravelModel(
            name="model",
            inputs=(),
            outputs=(),
        )
        step.model = MagToTravelModel(
            pred_soft_mg=50.0,
            coeffs=np.array([1000.0, 1.0, 1.0]),
        )
        mag = np.linspace(1000.0, 1100.0, 101)
        predictions = step.model.pred_x(mag)
        expected_zero_offset = -float(step.model.pred_x(np.percentile(mag, 8)))

        adjusted = step.adjust_with_ref_point(
            predictions,
            ref_x=100.0,
            ref_mag=1000.0,
            mag=mag,
            active_mask=np.ones(len(mag), dtype=bool),
            max_offset_delta_mm=40.0,
        )

        np.testing.assert_allclose(adjusted, predictions + expected_zero_offset)

    def test_nonfinite_reference_never_propagates_nan(self):
        step = GetMagToTravelModel(
            name="model",
            inputs=(),
            outputs=(),
        )
        step.model = MagToTravelModel(
            pred_soft_mg=50.0,
            coeffs=np.array([1000.0, 1.0, 1.0]),
        )
        mag = np.linspace(1000.0, 1100.0, 11)
        predictions = step.model.pred_x(mag)

        adjusted = step.adjust_with_ref_point(
            predictions,
            ref_x=np.nan,
            ref_mag=np.nan,
            mag=mag,
        )

        self.assertTrue(np.all(np.isfinite(adjusted)))


if __name__ == "__main__":
    unittest.main()
