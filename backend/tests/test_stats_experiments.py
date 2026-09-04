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
from tools.stats_aggregator import collect_report


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


def write_stats_cache(cache_root: Path, log_id: str, *, include_corrected: bool) -> None:
    cache_dir = cache_root / log_id / "cache"
    cache_dir.mkdir(parents=True)
    time_s = np.arange(4, dtype=float) * 0.01
    travel = np.array([0.0, 10.0, 20.0, 30.0])
    payload = {
        "travel__t": time_s,
        "travel__x": travel,
        "boring_mask": np.ones(4, dtype=bool),
    }
    for key, offset in (
        ("travel/mag_model", 3.0),
        ("travel/mag_model/adj", 2.0),
        ("travel/solved", 1.0),
    ):
        payload[f"{key}__t"] = time_s
        payload[f"{key}__x"] = travel + offset
    if include_corrected:
        key = "travel/solved/mag_nuisance/fusion2"
        payload[f"{key}__t"] = time_s
        payload[f"{key}__x"] = travel + 0.5
    np.savez(cache_dir / "all.npz", **payload)


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
    def test_corrected_comparisons_are_optional_per_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory)
            write_stats_cache(cache_root, "old", include_corrected=False)
            write_stats_cache(cache_root, "new", include_corrected=True)

            report = collect_report(
                ["old", "new"],
                cache_root,
                center_errors=False,
                error_threshold=None,
                include_diagnostics=False,
            )

            self.assertFalse(report.failures)
            self.assertEqual(len(report.error_rows["travel/solved"]), 2)
            corrected = report.error_rows["travel/solved/mag_nuisance/fusion2"]
            self.assertEqual([row["log"] for row in corrected], ["new"])

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
                print_comparison(
                    "base",
                    "current",
                    root,
                    "both",
                    "error",
                    "travel/solved",
                    "travel/solved",
                    "rmse",
                    20,
                )
            self.assertIn("input checksum changed", output.getvalue())
            self.assertIn("0 overlapping logs", output.getvalue())

    def test_comparison_can_use_different_baseline_and_current_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = (
                ("base", "travel/solved", 3.0),
                ("current", "travel/solved/mag_nuisance/fusion2", 2.0),
            )
            for experiment_id, comparison, value in specs:
                experiment_dir = root / experiment_id
                experiment_dir.mkdir()
                manifest = {
                    "id": experiment_id,
                    "name": experiment_id,
                    "created_at": "2026-01-01T00:00:00Z",
                    "stats": {"error_threshold": None},
                    "versions": {"pipeline_code_sha256": [f"code-{experiment_id}"]},
                    "logs": [{"log": "same-log", "input_sha256": "same-input"}],
                }
                (experiment_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
                write_metrics(
                    experiment_dir / "metrics.csv",
                    [
                        {
                            "centering": "centered",
                            "section": "error",
                            "log": "same-log",
                            "comparison": comparison,
                            "metric": "rmse",
                            "value": value,
                        }
                    ],
                )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                print_comparison(
                    "base",
                    "current",
                    root,
                    "centered",
                    "error",
                    "travel/solved",
                    "travel/solved/mag_nuisance/fusion2",
                    "rmse",
                    20,
                )
            rendered = output.getvalue()
            self.assertIn("Baseline output: travel/solved", rendered)
            self.assertIn("Current output:  travel/solved/mag_nuisance/fusion2", rendered)
            self.assertIn("delta -1.0000", rendered)


if __name__ == "__main__":
    unittest.main()
