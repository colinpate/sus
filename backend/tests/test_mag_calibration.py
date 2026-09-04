from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from classes.time_series import TimeSeries
from fusion import GetMagToTravelModel, GetRearMagToTravelModel
from mag_calibration import MagTravelCalibration, TimeRange, resolve_window
from mag_to_travel_model_core import MagToTravelChunk, MagToTravelModelCore


def calibration(pipeline: str, feature_key: str) -> MagTravelCalibration:
    return MagTravelCalibration(
        pipeline=pipeline,
        coefficients=(0.0, 2.0, 1.0),
        pred_soft_mg=1.0,
        feature_key=feature_key,
        training_log="source",
        training_time_basis="active",
        training_start_s=0.0,
        training_stop_s=10.0,
        training_sample_start=0,
        training_sample_stop=100,
        training_chunk_count=4,
    )


class WindowResolutionTests(unittest.TestCase):
    def test_elapsed_window_uses_half_open_bounds(self):
        resolved = resolve_window(
            np.arange(6, dtype=float),
            TimeRange(1.0, 4.0),
            time_basis="elapsed",
        )
        self.assertEqual(resolved.sample_range, (1, 4))

    def test_active_window_spans_inactive_wall_time(self):
        resolved = resolve_window(
            np.arange(6, dtype=float),
            TimeRange(0.0, 3.0),
            time_basis="active",
            activity_mask=np.array([False, True, False, True, True, False]),
        )
        self.assertEqual(resolved.sample_range, (1, 5))
        self.assertEqual(resolved.active_duration_s, 3.0)

    def test_adjacent_active_windows_do_not_overlap(self):
        time_s = np.arange(8, dtype=float)
        activity = np.array([False, True, True, False, True, True, True, False])
        first = resolve_window(
            time_s,
            TimeRange(0.0, 2.0),
            time_basis="active",
            activity_mask=activity,
        )
        second = resolve_window(
            time_s,
            TimeRange(2.0, 4.0),
            time_basis="active",
            activity_mask=activity,
        )
        self.assertLessEqual(first.sample_stop, second.sample_start)

    def test_chunk_gate_requires_complete_containment(self):
        model = MagToTravelModelCore()
        chunks = [
            MagToTravelChunk(np.zeros(2), np.arange(2), np.zeros(2), slice(0, 2), 0),
            MagToTravelChunk(np.zeros(2), np.arange(2), np.zeros(2), slice(2, 4), 0),
            MagToTravelChunk(np.zeros(3), np.arange(3), np.zeros(3), slice(3, 6), 0),
        ]
        selected = model.select_chunks_by_sample_range(chunks, (1, 5))
        self.assertEqual([chunk.slice_i for chunk in selected], [slice(2, 4)])


class CalibrationArtifactTests(unittest.TestCase):
    def test_json_round_trip(self):
        original = calibration("rear", "mag/angle/lpf")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            original.save(path)
            restored = MagTravelCalibration.load(path)
        self.assertEqual(restored, original)
        np.testing.assert_allclose(restored.make_model().pred_x(np.array([1.0, 2.0])), [2.0, 4.0])

    def test_oracle_artifact_interpolates_and_clips(self):
        oracle = MagTravelCalibration(
            pipeline="front",
            method="oracle_binned_median",
            feature_key="mag/norm/corr/lpf",
            training_log="source",
            training_time_basis="active",
            training_start_s=0.0,
            training_stop_s=10.0,
            training_sample_start=0,
            training_sample_stop=100,
            training_chunk_count=0,
            mag_knots=(1.0, 2.0, 3.0),
            travel_knots=(10.0, 20.0, 30.0),
        )
        np.testing.assert_allclose(
            oracle.predict(np.array([0.0, 1.5, 4.0])),
            [10.0, 15.0, 30.0],
        )
        with self.assertRaisesRegex(ValueError, "not a power-law"):
            oracle.make_model()

    def test_front_step_can_bypass_training(self):
        feature = "mag/norm/corr/lpf"
        step = GetMagToTravelModel(
            name="front_provided",
            inputs=(feature, "accel", "travel", "bad", "zv", "ref", "baseline"),
            outputs=("raw", "adjusted", "scatter", "coeffs", "offset"),
            apply_ref_point=False,
            provided_calibration=calibration("front", feature),
        )
        t = np.arange(3, dtype=float)
        ws = {
            feature: TimeSeries(t=t, x=np.array([1.0, 2.0, 3.0])),
            "accel": TimeSeries(t=t, x=np.zeros(3)),
            "travel": TimeSeries(t=t, x=np.zeros(3)),
            "bad": TimeSeries(t=t, x=np.zeros(3, dtype=bool)),
            "zv": np.array([], dtype=int),
            "ref": np.array([0.0, 0.0]),
            "baseline": np.array([0.0]),
        }
        with patch.object(step, "train", side_effect=AssertionError("training should be bypassed")):
            step.run(ws)
        np.testing.assert_allclose(ws["raw"].x[:, 0], [2.0, 4.0, 6.0])
        np.testing.assert_allclose(ws["coeffs"], [0.0, 2.0, 1.0])

    def test_rear_step_can_bypass_training(self):
        feature = "mag/angle/lpf"
        step = GetRearMagToTravelModel(
            name="rear_provided",
            inputs=(feature, "accel", "zv"),
            outputs=("raw", "adjusted", "scatter", "coeffs"),
            zero_travel_percentile=0.0,
            provided_calibration=calibration("rear", feature),
        )
        t = np.arange(3, dtype=float)
        ws = {
            feature: TimeSeries(t=t, x=np.array([1.0, 2.0, 3.0])),
            "accel": TimeSeries(t=t, x=np.zeros(3)),
            "zv": np.array([], dtype=int),
        }
        with patch.object(step, "train", side_effect=AssertionError("training should be bypassed")):
            step.run(ws)
        np.testing.assert_allclose(ws["raw"].x[:, 0], [2.0, 4.0, 6.0])
        np.testing.assert_allclose(ws["adjusted"].x[:, 0], [0.0, 2.0, 4.0])


if __name__ == "__main__":
    unittest.main()
