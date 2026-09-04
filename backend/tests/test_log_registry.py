from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from backend.log_registry import LogRegistry, dump_toml


class LogRegistryTests(unittest.TestCase):
    def test_profiles_merge_in_order_and_metadata_stays_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            self._assert_profiles_merge(tmp_path)

    def _assert_profiles_merge(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "sample.csv"
        csv_path.write_text("t_s\n0\n", encoding="utf-8")
        data = {
        "schema_version": 1,
        "profiles": {
            "hardware": {
                "pipeline": "front",
                "metadata": {"pod_version": 2},
                "config": {"signals": {"mag": {"lag": 1}}},
            },
            "geometry": {
                "metadata": {"fork_model": "Fox 36"},
                "config": {"steps": {"angle_to_travel": {"hypotenuse": 125.0}}},
            },
        },
        "presets": {},
        "logs": {
            "sample": {
                "file": "sample.csv",
                "status": "usable",
                "profiles": ["hardware", "geometry"],
                "metadata": {"trail": "Ccdh"},
                "overrides": {"signals": {"mag": {"lag": 2}}},
            }
        },
    }
        registry_path = tmp_path / "registry.toml"
        registry_path.write_text(dump_toml(data), encoding="utf-8")

        registry = LogRegistry.load(registry_path)
        log = registry.resolve("sample")

        self.assertEqual(log.pipeline, "front")
        self.assertEqual(log.metadata, {"pod_version": 2, "fork_model": "Fox 36", "trail": "Ccdh"})
        self.assertEqual(log.processing_config, {
            "signals": {"mag": {"lag": 2}},
            "steps": {"angle_to_travel": {"hypotenuse": 125.0}},
        })
        self.assertEqual(registry.validate(), [])

    def test_toml_writer_round_trips_registry_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = {
                "schema_version": 1,
                "profiles": {"p": {"config": {"matrix": [[1.0, 0.0], [0.0, -1.0]]}}},
                "presets": {},
                "logs": {"log-0001": {"pipeline": "front", "status": "pending-review", "profiles": ["p"]}},
            }
            path = Path(directory) / "registry.toml"
            path.write_text(dump_toml(data), encoding="utf-8")

            loaded = LogRegistry.load(path)

            self.assertEqual(loaded.data, data)

    def test_nonusable_logs_are_excluded_by_default(self) -> None:
        registry = LogRegistry(
            Path("registry.toml"),
            {
                "schema_version": 1,
                "profiles": {},
                "presets": {},
                "logs": {
                    "good": {"pipeline": "front", "status": "usable"},
                    "bad": {"pipeline": "front", "status": "corrupt"},
                },
            },
        )

        self.assertEqual([log.log_id for log in registry.select()], ["good"])
        self.assertEqual([log.log_id for log in registry.select(usable_only=False)], ["good", "bad"])


if __name__ == "__main__":
    unittest.main()
