"""Verify telemetry controls are set before dependencies perform inference."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dependency_privacy import disable_onnx_telemetry, disable_ultralytics_telemetry


class DependencyPrivacyTest(unittest.TestCase):
    def test_ultralytics_disables_existing_event_collector_and_sync(self):
        settings = {"sync": True}
        collector = SimpleNamespace(enabled=True)
        with (
            patch.dict(
                sys.modules,
                {
                    "ultralytics": SimpleNamespace(settings=settings),
                    "ultralytics.utils.events": SimpleNamespace(events=collector),
                },
            ),
            patch.dict(os.environ, {"YOLO_OFFLINE": "false"}),
        ):
            disable_ultralytics_telemetry()
            self.assertEqual(os.environ["YOLO_OFFLINE"], "true")
            self.assertFalse(settings["sync"])
            self.assertFalse(collector.enabled)

    def test_onnx_uses_supported_telemetry_opt_out(self):
        runtime = SimpleNamespace(disable_telemetry_events=Mock())
        with patch.dict(sys.modules, {"onnxruntime": runtime}):
            disable_onnx_telemetry()
        runtime.disable_telemetry_events.assert_called_once_with()
