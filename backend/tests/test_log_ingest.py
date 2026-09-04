from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from backend.log_registry import LogRegistry, dump_toml
from read_binary import DUAL_MAG_STRUCT
from tools.logs import ingest_one


class LogIngestTests(unittest.TestCase):
    def test_batch_ready_ingest_converts_and_registers_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "registry.toml"
            registry_path.write_text(
                dump_toml(
                    {
                        "schema_version": 1,
                        "profiles": {
                            "front": {
                                "pipeline": "front",
                                "record_format": "dual_mag",
                                "metadata": {"pod_version": 2},
                            }
                        },
                        "presets": {},
                        "logs": {},
                    }
                ),
                encoding="utf-8",
            )
            source = root / "download.bin"
            source.write_bytes(DUAL_MAG_STRUCT.pack(1000, 1, *([0] * 18), 2048, 250))
            registry = LogRegistry.load(registry_path)

            result = ingest_one(
                registry,
                source,
                log_id="new-log",
                base_record={"profiles": ["front"], "status": "pending-review"},
                metadata={"trail": "Ccdh"},
                status=None,
                sets=["test"],
                tags=["park"],
                record_format=None,
            )

            reloaded = LogRegistry.load(registry_path)
            log = reloaded.resolve("new-log")
            self.assertEqual(result.records, 1)
            self.assertTrue(log.require_csv().exists())
            self.assertTrue(log.source_path and log.source_path.exists())
            self.assertEqual(log.status, "pending-review")
            self.assertEqual(log.metadata["trail"], "Ccdh")
            self.assertEqual(log.metadata["pod_version"], 2)
            self.assertEqual(log.sets, ("test",))
            self.assertEqual(log.tags, ("park",))


if __name__ == "__main__":
    unittest.main()
