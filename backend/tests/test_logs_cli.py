from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

from backend.log_registry import LogRegistry, RegistryError, dump_toml
from tools.logs import command_annotate, parse_nested_key_values


class LogAnnotationTests(unittest.TestCase):
    def make_registry(self, directory: str) -> Path:
        path = Path(directory) / "registry.toml"
        path.write_text(
            dump_toml(
                {
                    "schema_version": 1,
                    "profiles": {
                        "front-v2": {
                            "pipeline": "front",
                            "metadata": {
                                "bike_model": "Specialized Stumpjumper",
                                "pod_version": 2,
                            },
                            "config": {"steps": {"angle_to_travel": {"top_adjacent": 113.5}}},
                        }
                    },
                    "presets": {},
                    "logs": {
                        "match": {
                            "status": "usable",
                            "profiles": ["front-v2"],
                            "sets": ["existing"],
                            "overrides": {"signals": {"angle": {"lag": -1}}},
                        },
                        "corrupt-match": {"status": "corrupt", "profiles": ["front-v2"]},
                        "other": {
                            "pipeline": "rear",
                            "status": "usable",
                            "metadata": {
                                "bike_model": "Specialized Stumpjumper",
                                "pod_version": 2,
                            },
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    def annotation_args(self, registry: Path, *, all_statuses: bool = False) -> Namespace:
        return Namespace(
            registry=registry,
            logs=[],
            where=["bike_model=Specialized Stumpjumper", "pod_version=2", "pipeline=front"],
            all_statuses=all_statuses,
            metadata=[],
            trail=None,
            frame_model=None,
            fork_model=None,
            shock_model=None,
            notes=None,
            status=None,
            reason=None,
            tags=[],
            remove_tags=[],
            sets=["filtered"],
            remove_sets=[],
            overrides=["steps.angle_to_travel.top_zeroangle=1.52788"],
        )

    def test_filtered_annotation_adds_set_and_merges_nested_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry_path = self.make_registry(directory)

            command_annotate(self.annotation_args(registry_path))

            registry = LogRegistry.load(registry_path)
            match = registry.resolve("match")
            self.assertEqual(match.sets, ("existing", "filtered"))
            self.assertEqual(match.processing_config["steps"]["angle_to_travel"], {
                "top_adjacent": 113.5,
                "top_zeroangle": 1.52788,
            })
            self.assertEqual(match.processing_config["signals"]["angle"]["lag"], -1)
            self.assertNotIn("filtered", registry.resolve("corrupt-match").sets)
            self.assertNotIn("filtered", registry.resolve("other").sets)

    def test_all_statuses_includes_nonusable_filtered_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry_path = self.make_registry(directory)

            command_annotate(self.annotation_args(registry_path, all_statuses=True))

            registry = LogRegistry.load(registry_path)
            self.assertIn("filtered", registry.resolve("match").sets)
            self.assertIn("filtered", registry.resolve("corrupt-match").sets)

    def test_explicit_nonusable_log_keeps_previous_annotation_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry_path = self.make_registry(directory)
            args = self.annotation_args(registry_path)
            args.logs = ["corrupt-match"]
            args.where = []

            command_annotate(args)

            registry = LogRegistry.load(registry_path)
            self.assertIn("filtered", registry.resolve("corrupt-match").sets)

    def test_no_filter_matches_are_rejected_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry_path = self.make_registry(directory)
            before = registry_path.read_text(encoding="utf-8")
            args = self.annotation_args(registry_path)
            args.where = ["bike_model=does-not-exist"]

            with self.assertRaisesRegex(RegistryError, "No logs matched"):
                command_annotate(args)
            self.assertEqual(registry_path.read_text(encoding="utf-8"), before)

    def test_annotation_without_ids_or_filters_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry_path = self.make_registry(directory)
            args = self.annotation_args(registry_path)
            args.where = []

            with self.assertRaisesRegex(RegistryError, "log IDs or at least one --where"):
                command_annotate(args)

    def test_nested_override_parser_rejects_conflicting_keys(self) -> None:
        with self.assertRaisesRegex(RegistryError, "conflicts"):
            parse_nested_key_values(["steps.angle_to_travel=1", "steps.angle_to_travel.top=2"])


if __name__ == "__main__":
    unittest.main()
