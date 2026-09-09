from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from classes.runner import Runner


class RunnerCacheTests(unittest.TestCase):
    def test_incomplete_cache_is_not_a_hit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner(out_dir=Path(directory), make_plots=False)
            runner._save_cache("step", {"first": np.array([1])}, ("first",))

            restored = {}
            loaded = runner._load_cache("step", restored, ("first", "second"))

            self.assertFalse(loaded)
            self.assertEqual(restored, {})

    def test_complete_cache_is_a_hit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner(out_dir=Path(directory), make_plots=False)
            expected = {"first": np.array([1]), "second": np.array([2])}
            runner._save_cache("step", expected, tuple(expected))

            restored = {}
            loaded = runner._load_cache("step", restored, tuple(expected))

            self.assertTrue(loaded)
            np.testing.assert_array_equal(restored["first"], expected["first"])
            np.testing.assert_array_equal(restored["second"], expected["second"])


if __name__ == "__main__":
    unittest.main()
