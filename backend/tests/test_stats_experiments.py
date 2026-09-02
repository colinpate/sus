from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from backend.log_registry import ResolvedLog
from backend.run_provenance import build_run_provenance
from tools.stats_experiments import (
    MANIFEST_FILENAME,
    aggregate,
    inspect_cache,
    load_metrics,
    primary_metric_values,
    resolve_experiment,
    write_metrics,
)
from tools.stats import print_comparison


def make_log(root: Path, *, config_value: int = 1) -> ResolvedLog:
    csv_path = root / "input.csv"
    if not csv_path.exists():
        csv_path.write_text("t_s,value\n0,1\n", encoding="utf-8")
    return ResolvedLog(
        log_id="sample",
        registry_path=root / "logs" / "registry.toml",
        csv_path=csv_path,
        source_path=None,
        pipeline="front",
        status="usable",
        profiles=("sample-profile",),
        sets=("test",),
        tags=(),
        metadata={"trail": "Test"},
        processing_config={"steps": {"sample": {"value": config_value}}},
        record_format="dual_mag",
    )


class CacheInspectionTests(unittest.TestCase):
    def test_requires_current_manifest_and_embedded_cache_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "backend").mkdir()
            (root / "backend" / "pipeline.py").write_text("VALUE = 1\n", encoding="utf-8")
            log = make_log(root)
            expected = build_run_provenance(log, pipeline="front", root=root)

            cache_root = root / "artifacts"
            run_dir = cache_root / log.log_id
            (run_dir / "cache").mkdir(parents=True)
            (run_dir / "run.json").write_text(json.dumps(expected.to_dict()), encoding="utf-8")
            np.savez(run_dir / "cache" / "all.npz", __run_fingerprint=np.array(expected.run_fingerprint))

            inspection = inspect_cache(log, cache_root, expected=expected)
            self.assertTrue(inspection.fresh)

            changed = build_run_provenance(make_log(root, config_value=2), pipeline="front", root=root)
            inspection = inspect_cache(log, cache_root, expected=changed)
            self.assertEqual(inspection.status, "stale")

    def test_rejects_cache_that_does_not_match_successful_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "backend").mkdir()
            (root / "backend" / "pipeline.py").write_text("VALUE = 1\n", encoding="utf-8")
            log = make_log(root)
            expected = build_run_provenance(log, pipeline="front", root=root)
            cache_root = root / "artifacts"
            run_dir = cache_root / log.log_id
            (run_dir / "cache").mkdir(parents=True)
            (run_dir / "run.json").write_text(json.dumps(expected.to_dict()), encoding="utf-8")
            np.savez(run_dir / "cache" / "all.npz", __run_fingerprint=np.array("wrong"))

            inspection = inspect_cache(log, cache_root, expected=expected)
            self.assertEqual(inspection.status, "cache-mismatch")


class ExperimentStoreTests(unittest.TestCase):
    def test_centering_is_part_of_metric_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            write_metrics(
                path,
                [
                    {"centering": "uncentered", "section": "error", "log": "a", "comparison": "travel/solved", "metric": "rmse", "value": 3.0},
                    {"centering": "centered", "section": "error", "log": "a", "comparison": "travel/solved", "metric": "rmse", "value": 2.0},
                ],
            )
            metrics = load_metrics(path.parent)
            self.assertEqual(primary_metric_values(metrics, centering="uncentered"), {"a": 3.0})
            self.assertEqual(primary_metric_values(metrics, centering="centered"), {"a": 2.0})
            self.assertEqual(aggregate([1.0, 3.0])["mean"], 2.0)

    def test_name_resolves_to_most_recent_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for experiment_id, created_at in (("old", "2026-01-01T00:00:00Z"), ("new", "2026-02-01T00:00:00Z")):
                experiment_dir = root / experiment_id
                experiment_dir.mkdir()
                (experiment_dir / MANIFEST_FILENAME).write_text(
                    json.dumps({"id": experiment_id, "name": "same-name", "created_at": created_at}),
                    encoding="utf-8",
                )

            path, manifest = resolve_experiment("same-name", root)
            self.assertEqual(path.name, "new")
            self.assertEqual(manifest["id"], "new")

    def test_comparison_excludes_same_log_id_with_changed_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for experiment_id, input_hash, value in (("base", "input-a", 3.0), ("current", "input-b", 2.0)):
                experiment_dir = root / experiment_id
                experiment_dir.mkdir()
                manifest = {
                    "id": experiment_id,
                    "name": experiment_id,
                    "created_at": f"2026-01-0{1 if experiment_id == 'base' else 2}T00:00:00Z",
                    "stats": {"error_threshold": None},
                    "versions": {"pipeline_code_sha256": [f"code-{experiment_id}"]},
                    "logs": [{"log": "same-log", "input_sha256": input_hash}],
                }
                (experiment_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
                write_metrics(
                    experiment_dir / "metrics.csv",
                    [
                        {
                            "centering": centering,
                            "section": "error",
                            "log": "same-log",
                            "comparison": "travel/solved",
                            "metric": "rmse",
                            "value": value,
                        }
                        for centering in ("uncentered", "centered")
                    ],
                )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                print_comparison("base", "current", root, "both", "error", "travel/solved", "rmse", 20)
            self.assertIn("input checksum changed", output.getvalue())
            self.assertIn("0 overlapping logs", output.getvalue())


if __name__ == "__main__":
    unittest.main()
