from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import AppConfig
from main import parse_args


class EnvironmentConfigTest(unittest.TestCase):
    def test_invalid_log_level_falls_back_to_info(self) -> None:
        with patch.dict(os.environ, {"MONITOR_LOG_LEVEL": "getLogger"}):
            self.assertEqual(AppConfig.from_env().log_level, "INFO")

    def test_invalid_cli_activity_settings_fail_at_argument_parsing(self) -> None:
        for option, value in (
            ("--activity-sequence-length", "0"),
            ("--activity-inference-interval", "-1"),
            ("--activity-smoothing-window", "0"),
            ("--activity-confidence-threshold", "nan"),
            ("--activity-confidence-threshold", "1.1"),
        ):
            with self.subTest(option=option, value=value):
                with patch.object(sys, "argv", ["main.py", option, value]):
                    with self.assertRaises(SystemExit) as error:
                        parse_args()
                    self.assertEqual(error.exception.code, 2)

    def test_activity_is_disabled_by_default(self) -> None:
        self.assertEqual(AppConfig().activity_model, "none")

    def test_invalid_boolean_preserves_the_default(self) -> None:
        with patch.dict(os.environ, {"MONITOR_ALLOW_MODEL_DOWNLOADS": "maybe"}):
            config = AppConfig.from_env()

        self.assertTrue(config.allow_model_downloads)

    def test_non_finite_number_preserves_the_default(self) -> None:
        with patch.dict(os.environ, {"MONITOR_EXPRESSION_CONFIDENCE": "nan"}):
            config = AppConfig.from_env()

        self.assertEqual(config.min_concern_expression_confidence, 0.65)

    def test_activity_environment_values_are_loaded(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MONITOR_ACTIVITY_MODEL": "MLP",
                "MONITOR_ACTIVITY_CONFIDENCE": "0.7",
                "MONITOR_ACTIVITY_INTERVAL": "3",
            },
        ):
            config = AppConfig.from_env()

        self.assertEqual(config.activity_model, "mlp")
        self.assertEqual(config.activity_confidence_threshold, 0.7)
        self.assertEqual(config.activity_inference_interval, 3)

    def test_activity_cli_values_are_parsed(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "main.py",
                "--detect",
                "--activity-model",
                "mlp",
                "--activity-sequence-length",
                "16",
                "--activity-confidence-threshold",
                "0.6",
                "--activity-inference-interval",
                "4",
                "--activity-smoothing-window",
                "3",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.activity_model, "mlp")
        self.assertEqual(args.activity_sequence_length, 16)
        self.assertEqual(args.activity_confidence_threshold, 0.6)
        self.assertEqual(args.activity_inference_interval, 4)
        self.assertEqual(args.activity_smoothing_window, 3)


if __name__ == "__main__":
    unittest.main()
