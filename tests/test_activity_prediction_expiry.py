"""Exercise activity freshness without loading a neural-network checkpoint."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from activity_recognition.live import ActivityPrediction, LiveMLPActivityRecognizer


LANDMARKS = {0: (0.5, 0.5)}
BOX = (0, 0, 100, 100)
FRAME_SHAPE = (100, 100, 3)
WALKING = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
RUNNING = np.array([0.1, 0.9, 0.0, 0.0], dtype=np.float32)


def make_recognizer(sequence_length=2, inference_interval=1):
    # Only model loading is bypassed; update, normalization and smoothing are real.
    recognizer = LiveMLPActivityRecognizer.__new__(LiveMLPActivityRecognizer)
    recognizer.sequence_length = sequence_length
    recognizer.inference_interval = inference_interval
    recognizer.smoothing_window = 5
    recognizer.confidence_threshold = 0.6
    recognizer._track_states = {}
    recognizer._inference_latency_total_ms = 0.0
    recognizer._inference_count = 0
    return recognizer


class ActivityPredictionExpiryTest(unittest.TestCase):
    def test_intermittent_missing_poses_expire_a_prediction(self):
        recognizer = make_recognizer()
        with patch.object(recognizer, "_predict_probabilities", return_value=WALKING):
            recognizer.update(1, LANDMARKS, BOX, FRAME_SHAPE, 0)
            first = recognizer.update(1, LANDMARKS, BOX, FRAME_SHAPE, 1)
            held = recognizer.update(1, {}, BOX, FRAME_SHAPE, 2)
            self.assertEqual(first.label, "walking")
            self.assertEqual(held, first)
            for frame in range(3, 30):
                prediction = recognizer.update(
                    1, LANDMARKS if frame % 2 else {}, BOX, FRAME_SHAPE, frame
                )
                self.assertEqual(prediction, ActivityPrediction("unknown", 0.0))
        self.assertEqual(recognizer.inference_count, 1)

    def test_recovery_uses_fresh_probabilities_and_preserves_other_tracks(self):
        recognizer = make_recognizer()
        with patch.object(recognizer, "_predict_probabilities", return_value=WALKING):
            for frame in (0, 1):
                for track in (1, 2):
                    recognizer.update(track, LANDMARKS, BOX, FRAME_SHAPE, frame)
            recognizer.update(1, {}, BOX, FRAME_SHAPE, 2)
            recognizer.update(1, LANDMARKS, BOX, FRAME_SHAPE, 3)
            other_track = recognizer.update(2, LANDMARKS, BOX, FRAME_SHAPE, 3)
        with patch.object(recognizer, "_predict_probabilities", return_value=RUNNING):
            recovered = recognizer.update(1, LANDMARKS, BOX, FRAME_SHAPE, 4)
        self.assertEqual(other_track.label, "walking")
        self.assertEqual(recovered.label, "running")
        self.assertAlmostEqual(recovered.confidence, 0.9)

    def test_scheduled_inference_does_not_expire_healthy_smoothing(self):
        recognizer = make_recognizer(sequence_length=1, inference_interval=3)
        with patch.object(recognizer, "_predict_probabilities", return_value=WALKING):
            first = recognizer.update(1, LANDMARKS, BOX, FRAME_SHAPE, 0)
            for frame in (1, 2):
                self.assertEqual(
                    recognizer.update(1, LANDMARKS, BOX, FRAME_SHAPE, frame), first
                )
        with patch.object(recognizer, "_predict_probabilities", return_value=RUNNING):
            prediction = recognizer.update(1, LANDMARKS, BOX, FRAME_SHAPE, 3)
        self.assertEqual(prediction.label, "unknown")
        self.assertAlmostEqual(prediction.confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
