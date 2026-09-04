from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import unittest

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from mag_to_travel_model_core import MagToTravelChunk, MagToTravelModelCore


def make_chunk(mag_min: float, *, dx: float) -> MagToTravelChunk:
    chunk = MagToTravelChunk(
        a=np.zeros(2),
        t=np.array([0.0, 0.01]),
        mag=np.array([mag_min, mag_min + 1.0]),
        slice_i=slice(0, 2),
        zv_idx=0,
        badmask=np.zeros(2, dtype=bool),
        x=np.zeros(2),
    )
    chunk.metrics = {
        "badmask_mean": 0.0,
        "dm/dx_median": 0.1,
        "dx": dx,
        "mag_min": mag_min,
    }
    return chunk


class StubMagToTravelModelCore(MagToTravelModelCore):
    def create_chunks(self, *args, **kwargs):
        return self.stub_chunks

    def prepare_chunks(self, chunks):
        pass


class MagToTravelModelCoreTests(unittest.TestCase):
    def test_min_mag_relaxation_uses_only_eligible_chunks(self):
        model = StubMagToTravelModelCore(min_mag_relax_min_chunks=2)
        model.stub_chunks = [
            make_chunk(90.0, dx=20.0),
            make_chunk(80.0, dx=20.0),
            make_chunk(1000.0, dx=1.0),
            make_chunk(900.0, dx=1.0),
        ]

        output = io.StringIO()
        with redirect_stdout(output):
            training_data = model.create_training_data(
                mag=np.zeros(2),
                accel=np.zeros(2),
                train_mask=np.zeros(2, dtype=bool),
                t=np.array([0.0, 0.01]),
                baseline_min_mag=100.0,
                idxs=np.array([], dtype=int),
            )

        self.assertEqual([chunk.metrics["mag_min"] for chunk in model.chunks], [90.0, 80.0])
        self.assertEqual(training_data.shape, (2, 3, 2))
        self.assertIn("Candidate chunks: 4", output.getvalue())
        self.assertIn("Eligible chunks: 2", output.getvalue())
        self.assertIn("Training chunks: 2", output.getvalue())


if __name__ == "__main__":
    unittest.main()
