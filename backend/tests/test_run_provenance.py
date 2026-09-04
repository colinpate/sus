from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from backend.log_registry import ResolvedLog
from backend.run_provenance import build_run_provenance


def make_log(tmp_path: Path, *, config: dict | None = None, metadata: dict | None = None) -> ResolvedLog:
    csv_path = tmp_path / "input.csv"
    if not csv_path.exists():
        csv_path.write_text("t_s,value\n0,1\n", encoding="utf-8")
    return ResolvedLog(
        log_id="sample",
        registry_path=tmp_path / "registry.toml",
        csv_path=csv_path,
        source_path=None,
        pipeline="front",
        status="usable",
        profiles=("profile",),
        sets=(),
        tags=(),
        metadata=metadata or {},
        processing_config=config or {},
        record_format="dual_mag",
    )


class RunProvenanceTests(unittest.TestCase):
    def test_fingerprint_covers_input_config_and_uncommitted_source_but_not_notes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._assert_fingerprint_changes(Path(directory))

    def _assert_fingerprint_changes(self, tmp_path: Path) -> None:
        backend = tmp_path / "backend"
        backend.mkdir()
        source = backend / "pipeline.py"
        source.write_text("VALUE = 1\n", encoding="utf-8")
        log = make_log(tmp_path, config={"steps": {"x": {"value": 1}}}, metadata={"trail": "A"})

        initial = build_run_provenance(log, pipeline="front", root=tmp_path)
        note_change = build_run_provenance(replace(log, metadata={"trail": "B"}), pipeline="front", root=tmp_path)
        self.assertEqual(note_change.run_fingerprint, initial.run_fingerprint)

        config_change = build_run_provenance(
            replace(log, processing_config={"steps": {"x": {"value": 2}}}),
            pipeline="front",
            root=tmp_path,
        )
        self.assertNotEqual(config_change.run_fingerprint, initial.run_fingerprint)

        source.write_text("VALUE = 2\n", encoding="utf-8")
        source_change = build_run_provenance(log, pipeline="front", root=tmp_path)
        self.assertNotEqual(source_change.run_fingerprint, initial.run_fingerprint)

        log.csv_path.write_text("t_s,value\n0,2\n", encoding="utf-8")
        input_change = build_run_provenance(log, pipeline="front", root=tmp_path)
        self.assertNotEqual(input_change.run_fingerprint, source_change.run_fingerprint)


if __name__ == "__main__":
    unittest.main()
