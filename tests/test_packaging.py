"""Runtime-path regressions for installation outside a source checkout."""

from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config


class InstalledPathsTest(unittest.TestCase):
    def test_installed_module_uses_user_data_not_site_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(config, "__file__", str(home / "site-packages" / "config.py")),
                patch.object(Path, "home", return_value=home),
            ):
                settings = config.AppConfig.from_env()
            self.assertEqual(settings.project_root, home / ".integrated-monitoring-poc")
            self.assertEqual(settings.data_dir, settings.project_root / "data")

    def test_home_override_controls_all_writable_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory).resolve()
            with patch.dict(os.environ, {"MONITOR_HOME": str(home)}):
                settings = config.AppConfig.from_env()
            for field in (
                "data_dir",
                "enrollments_dir",
                "input_dir",
                "output_dir",
                "log_dir",
                "event_dir",
                "activity_checkpoint_path",
            ):
                self.assertTrue(getattr(settings, field).is_relative_to(home), field)
            self.assertTrue(settings.emotion_face_model_path.is_file())
            self.assertFalse(settings.emotion_face_model_path.is_relative_to(home))


if __name__ == "__main__":
    unittest.main()
