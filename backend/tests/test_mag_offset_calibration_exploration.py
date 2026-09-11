from __future__ import annotations

import unittest

import numpy as np

from tools.front.mag_offset_calibration.explore_mag_offset_calibration import (
    RefChunk,
    oob_offset,
    ref_point,
)
from tools.front.mag_offset_calibration.validate_mag_offset_legacy import (
    candidate_offsets as legacy_candidate_offsets,
    select_mag_key,
)


class ReferencePointTests(unittest.TestCase):
    def test_start_travel_estimate_is_added_to_relative_integration(self) -> None:
        chunk = RefChunk(
            mag=np.array([2600.0, 2800.0, 3000.0]),
            rel_x=np.array([10.0, 20.0, 30.0]),
            truth=np.array([15.0, 25.0, 35.0]),
            start_mag=500.0,
            start_truth=5.0,
            direction="forward",
        )
        ref_x, ref_mag, count = ref_point(
            [chunk],
            500.0,
            start_travel_estimator=lambda _: 5.0,
        )
        self.assertEqual(count, 1)
        self.assertAlmostEqual(ref_x, 25.0)
        self.assertAlmostEqual(ref_mag, 2800.0)

    def test_direction_filter_falls_back_when_no_chunks_remain(self) -> None:
        chunk = RefChunk(
            mag=np.array([2600.0]),
            rel_x=np.array([10.0]),
            truth=np.array([15.0]),
            start_mag=500.0,
            start_truth=5.0,
            direction="forward",
        )
        ref_x, ref_mag, count = ref_point(
            [chunk], 500.0, direction="reverse"
        )
        self.assertEqual((ref_x, ref_mag, count), (0.0, 2500.0, 0))


class OobOffsetTests(unittest.TestCase):
    def test_nearest_zero_uses_smallest_feasible_translation(self) -> None:
        offset = oob_offset(
            np.array([-10.0, 50.0, 160.0]),
            lower_percentile=0.0,
            upper_percentile=100.0,
            tie_break="nearest-zero",
        )
        self.assertAlmostEqual(offset, 10.0)

    def test_center_places_observed_range_in_middle_of_bounds(self) -> None:
        offset = oob_offset(
            np.array([10.0, 20.0]),
            lower_percentile=0.0,
            upper_percentile=100.0,
            tie_break="center",
        )
        self.assertAlmostEqual(offset, 70.0)


class LegacyValidationTests(unittest.TestCase):
    def test_prefers_corrected_norm_then_corrected_projection(self) -> None:
        self.assertEqual(
            select_mag_key({"mag/norm/corr/lpf__x", "mag/proj/corr/lpf__x"}),
            "mag/norm/corr/lpf__x",
        )
        self.assertEqual(
            select_mag_key({"mag/proj/corr/lpf__x"}),
            "mag/proj/corr/lpf__x",
        )

    def test_sparse_chunk_confidence_falls_back_to_p2(self) -> None:
        raw = np.linspace(-10.0, 90.0, 101)
        candidates = legacy_candidate_offsets(
            raw,
            current_offset=2.0,
            good_mask=np.ones(len(raw), dtype=bool),
            chunk_count=0,
        )
        self.assertAlmostEqual(
            candidates["chunk_confidence_n10_cap0.5"],
            candidates["prediction_p2_to_zero"],
        )

    def test_legacy_safeguard_preserves_current_with_many_chunks(self) -> None:
        raw = np.linspace(-10.0, 90.0, 101)
        candidates = legacy_candidate_offsets(
            raw,
            current_offset=2.0,
            good_mask=np.ones(len(raw), dtype=bool),
            chunk_count=25,
        )
        self.assertAlmostEqual(
            candidates["current_if_25_chunks_else_confidence"], 2.0
        )


if __name__ == "__main__":
    unittest.main()
