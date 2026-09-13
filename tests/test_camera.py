from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera import CaptureClock


class CaptureClockTest(unittest.TestCase):
    def clock(self, fps: float, positions: list[float]) -> CaptureClock:
        timestamps = iter(positions)
        capture = Mock()
        capture.get.side_effect = lambda prop: (
            fps if prop == cv2.CAP_PROP_FPS else next(timestamps)
        )
        return CaptureClock(capture, recorded=True)

    def test_recording_uses_source_timestamps(self) -> None:
        clock = self.clock(30.0, [0.0, 40.0, 100.0])
        self.assertEqual([clock.timestamp() for _ in range(3)], [0.0, 0.04, 0.1])

    def test_missing_or_regressing_timestamps_fall_back_to_frame_rate(self) -> None:
        clock = self.clock(25.0, [0.0, 0.0, float("nan"), 10.0, 200.0])
        for expected in [0.0, 0.04, 0.08, 0.12, 0.2]:
            self.assertAlmostEqual(clock.timestamp(), expected)

    def test_invalid_frame_rate_uses_default(self) -> None:
        for fps in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(fps=fps):
                clock = self.clock(fps, [0.0, 0.0])
                self.assertEqual(clock.timestamp(), 0.0)
                self.assertAlmostEqual(clock.timestamp(), 1.0 / 24.0)

    def test_live_source_uses_monotonic_time(self) -> None:
        capture = Mock()
        clock = CaptureClock(capture, recorded=False)
        with patch("camera.time.monotonic", side_effect=[100.0, 100.3]):
            self.assertEqual(clock.timestamp(), 100.0)
            self.assertEqual(clock.timestamp(), 100.3)
        capture.get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
