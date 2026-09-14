from __future__ import annotations

from subprocess import CalledProcessError
from types import SimpleNamespace
import unittest
from unittest.mock import call, patch

from tools.stats import process_logs


class StatsProcessingTests(unittest.TestCase):
    def test_continues_after_pipeline_failure_when_requested(self) -> None:
        logs = [
            SimpleNamespace(log_id="good-before", pipeline="front"),
            SimpleNamespace(log_id="broken", pipeline="rear"),
            SimpleNamespace(log_id="good-after", pipeline="front"),
        ]
        failure = CalledProcessError(7, ["pipeline", "broken"])

        with patch("tools.stats.process_log", side_effect=[None, failure, None]) as process_log:
            failures = process_logs(logs, continue_on_failure=True)

        self.assertEqual(process_log.call_args_list, [call(log) for log in logs])
        self.assertEqual(failures, [(logs[1], failure)])

    def test_pipeline_failure_still_stops_without_continue_mode(self) -> None:
        logs = [
            SimpleNamespace(log_id="broken", pipeline="rear"),
            SimpleNamespace(log_id="not-run", pipeline="front"),
        ]
        failure = CalledProcessError(7, ["pipeline", "broken"])

        with patch("tools.stats.process_log", side_effect=failure) as process_log:
            with self.assertRaises(CalledProcessError):
                process_logs(logs)

        process_log.assert_called_once_with(logs[0])


if __name__ == "__main__":
    unittest.main()
