from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from accel_rotation import (
    CorrectStaticOffset,
    FilterChunkPairs,
    FilterColinearPairs,
    RotationFromPairs,
)
from classes.log_config import attach_log_config
from classes.time_series import ChunkedTimeSeries, TimeSeries


def make_chunked(samples: np.ndarray) -> ChunkedTimeSeries:
    samples = np.asarray(samples, dtype=float)
    ts = TimeSeries(t=np.arange(len(samples), dtype=float), x=samples, units="m/s^2")
    return ChunkedTimeSeries(base=ts, spans=[(0, len(samples) - 1)])


def make_pair(vector_a=(0.0, 0.0, 9.81), vector_b=(0.0, 0.0, 9.81)):
    t = np.arange(5, dtype=float)
    chunk_a = TimeSeries(t=t, x=np.tile(vector_a, (5, 1)), units="m/s^2")
    chunk_b = TimeSeries(t=t, x=np.tile(vector_b, (5, 1)), units="m/s^2")
    return [chunk_a, chunk_b]


class AccelerometerAlignmentValidationTests(unittest.TestCase):
    def test_zero_output_sensor_reports_clear_pairing_error(self):
        good = np.tile([0.0, 0.0, 9.81], (6, 1))
        zero = np.zeros((6, 3))
        ws = {"a": make_chunked(good), "b": make_chunked(zero)}
        step = FilterChunkPairs(name="pairs", inputs=("a", "b"), outputs=("pairs",))

        with self.assertRaisesRegex(ValueError, "sensor dropout"):
            step.run(ws)

    def test_single_orientation_reports_under_constrained_alignment(self):
        ws = {"pairs": [make_pair(), make_pair(), make_pair()]}
        step = FilterColinearPairs(
            name="colinear",
            inputs=("pairs",),
            outputs=("filtered",),
        )

        with self.assertRaisesRegex(ValueError, "under-constrained"):
            step.run(ws)

    def test_rotation_rejects_too_few_pairs(self):
        ws = {"pairs": [make_pair()]}
        step = RotationFromPairs(name="rotation", inputs=("pairs",), outputs=("rotation",))

        with self.assertRaisesRegex(ValueError, "at least 2"):
            step.run(ws)

    def test_fixed_rotation_bypasses_pose_estimation(self):
        rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        ws = {"pairs": []}
        attach_log_config(
            ws,
            {"steps": {"rotation": {"fixed_rotation_matrix": rotation.tolist()}}},
        )
        step = RotationFromPairs(name="rotation", inputs=("pairs",), outputs=("rotation",))

        step.run(ws)

        np.testing.assert_allclose(ws["rotation"], rotation)

    def test_underconstrained_pairs_can_be_retained_for_fixed_alignment(self):
        pairs = [make_pair()]
        ws = {"pairs": pairs}
        attach_log_config(
            ws,
            {"steps": {"colinear": {"allow_underconstrained": True}}},
        )
        step = FilterColinearPairs(name="colinear", inputs=("pairs",), outputs=("filtered",))

        step.run(ws)

        self.assertEqual(ws["filtered"], pairs)

    def test_fixed_alignment_zero_pair_path_uses_zero_static_offset(self):
        good = np.tile([0.0, 0.0, 9.81], (6, 1))
        zero = np.zeros((6, 3))
        accel = TimeSeries(
            t=np.arange(6, dtype=float),
            x=good.copy(),
            units="m/s^2",
        )
        ws = {
            "a": make_chunked(good),
            "b": make_chunked(zero),
            "accel": accel,
        }
        attach_log_config(
            ws,
            {
                "steps": {
                    "pairs": {"allow_empty": True},
                    "colinear": {"allow_underconstrained": True},
                    "offset": {"allow_empty": True},
                }
            },
        )

        FilterChunkPairs(name="pairs", inputs=("a", "b"), outputs=("pairs",)).run(ws)
        FilterColinearPairs(
            name="colinear",
            inputs=("pairs",),
            outputs=("filtered", "chunks_a", "chunks_b"),
        ).run(ws)
        CorrectStaticOffset(
            name="offset",
            inputs=("chunks_a", "accel"),
            outputs=("chunks_a", "accel"),
        ).run(ws)

        self.assertEqual(ws["pairs"], [])
        self.assertEqual(ws["filtered"], [])
        self.assertEqual(ws["chunks_a"].shape, (0, 0, 3))
        np.testing.assert_allclose(ws["accel"].x, good)


if __name__ == "__main__":
    unittest.main()
