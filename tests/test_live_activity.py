from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from activity_recognition.labels import ACTIVITY_LABELS
from activity_recognition.live import (
    ActivityModelError,
    ActivityPrediction,
    LiveMLPActivityRecognizer,
)
from activity_recognition.models import build_activity_classifier
from config import DEFAULT_CONFIG
from detection import MonitoringProcessor
from events import EventRecorder
from pose import PoseResult

FRAME_SHAPE = (120, 160, 3)
PERSON_BOX = (20, 10, 140, 115)
LANDMARKS = {index: (0.25 + index * 0.02, 0.20 + index * 0.025) for index in range(17)}
_UNSET = object()


def write_mlp_checkpoint(
    path: Path,
    *,
    sequence_length: int = 2,
    output_bias: tuple[float, float, float, float] = (4.0, 0.0, 0.0, 0.0),
    labels: tuple[str, ...] = ACTIVITY_LABELS,
) -> None:
    input_dim = sequence_length * 17 * 3
    model = build_activity_classifier("mlp", input_dim)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[3].bias.copy_(torch.tensor(output_bias))
    torch.save(
        {
            "model_name": "mlp",
            "input_dim": input_dim,
            "state_dict": model.state_dict(),
            "feature_mean": torch.zeros(input_dim),
            "feature_std": torch.ones(input_dim),
            "labels": list(labels),
            "label_to_index": {label: index for index, label in enumerate(labels)},
            "frames_per_sample": sequence_length,
        },
        path,
    )


class SequencePoseAnalyzer:
    def __init__(self, frames: list[list[PoseResult]]) -> None:
        self.frames = list(frames)
        self.reset_count = 0

    def analyze(self, frame: np.ndarray) -> list[PoseResult]:
        return self.frames.pop(0) if self.frames else []

    def draw_landmarks(self, frame, landmarks, color) -> None:
        return None

    def reset_tracking(self) -> None:
        self.reset_count += 1


class RecordingActivityRecognizer:
    def __init__(
        self,
        prediction: ActivityPrediction | None = None,
    ) -> None:
        self.prediction = prediction
        self.update_calls: list[int] = []
        self.removed_tracks: list[int] = []
        self.reset_count = 0
        self.average_inference_latency_ms = None
        self.inference_count = 0

    def update(self, track_key, landmarks, box, frame_shape, frame_number, *, timestamp=None):
        self.update_calls.append(track_key)
        return self.prediction

    def remove_track(self, track_key: int) -> None:
        self.removed_tracks.append(track_key)

    def reset(self) -> None:
        self.reset_count += 1


class LiveMLPActivityRecognizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.checkpoint_path = Path(self.temporary_directory.name) / "activity-mlp.pt"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def recognizer(self, **overrides) -> LiveMLPActivityRecognizer:
        settings = {
            "sequence_length": 2,
            "confidence_threshold": 0.5,
            "inference_interval": 1,
            "smoothing_window": 3,
        }
        settings.update(overrides)
        write_mlp_checkpoint(
            self.checkpoint_path,
            sequence_length=settings["sequence_length"],
        )
        return LiveMLPActivityRecognizer(self.checkpoint_path, **settings)

    def test_missing_checkpoint_has_a_clear_error(self) -> None:
        with self.assertRaisesRegex(
            ActivityModelError,
            "checkpoint not found",
        ):
            LiveMLPActivityRecognizer(self.checkpoint_path)

    def test_checkpoint_label_order_is_validated(self) -> None:
        write_mlp_checkpoint(
            self.checkpoint_path,
            labels=("running", "walking", "standing", "sitting"),
        )

        with self.assertRaisesRegex(ActivityModelError, "labels must be"):
            LiveMLPActivityRecognizer(
                self.checkpoint_path,
                sequence_length=2,
            )

    def test_checkpoint_sequence_length_is_validated(self) -> None:
        write_mlp_checkpoint(self.checkpoint_path, sequence_length=2)

        with self.assertRaisesRegex(
            ActivityModelError,
            "sequence length does not match",
        ):
            LiveMLPActivityRecognizer(
                self.checkpoint_path,
                sequence_length=3,
            )

    def test_insufficient_and_missing_pose_history_do_not_infer(self) -> None:
        recognizer = self.recognizer()

        first = recognizer.update(1, LANDMARKS, PERSON_BOX, FRAME_SHAPE, frame_number=0)
        missing = recognizer.update(1, {}, PERSON_BOX, FRAME_SHAPE, frame_number=1)

        self.assertIsNone(first)
        self.assertIsNone(missing)
        self.assertEqual(recognizer.inference_count, 0)

    def test_all_zero_pose_sequence_becomes_unknown_without_inference(self) -> None:
        recognizer = self.recognizer()

        recognizer.update(1, {}, PERSON_BOX, FRAME_SHAPE, frame_number=0)
        prediction = recognizer.update(1, {}, PERSON_BOX, FRAME_SHAPE, frame_number=1)

        self.assertEqual(prediction, ActivityPrediction("unknown", 0.0))
        self.assertEqual(recognizer.inference_count, 0)

    def test_low_confidence_prediction_becomes_unknown(self) -> None:
        write_mlp_checkpoint(
            self.checkpoint_path,
            output_bias=(0.0, 0.0, 0.0, 0.0),
        )
        recognizer = LiveMLPActivityRecognizer(
            self.checkpoint_path,
            sequence_length=2,
            confidence_threshold=0.5,
            inference_interval=1,
            smoothing_window=1,
        )

        recognizer.update(1, LANDMARKS, PERSON_BOX, FRAME_SHAPE, 0)
        prediction = recognizer.update(1, LANDMARKS, PERSON_BOX, FRAME_SHAPE, 1)

        self.assertEqual(prediction.label, "unknown")
        self.assertAlmostEqual(prediction.confidence, 0.25)

    def test_pose_histories_are_isolated_by_track(self) -> None:
        recognizer = self.recognizer()

        recognizer.update(10, LANDMARKS, PERSON_BOX, FRAME_SHAPE, 0)
        recognizer.update(20, LANDMARKS, PERSON_BOX, FRAME_SHAPE, 0)
        track_ten = recognizer.update(10, LANDMARKS, PERSON_BOX, FRAME_SHAPE, 1)

        self.assertEqual(track_ten.label, "walking")
        self.assertEqual(recognizer.inference_count, 1)
        track_twenty = recognizer.update(20, LANDMARKS, PERSON_BOX, FRAME_SHAPE, 1)
        self.assertEqual(track_twenty.label, "walking")
        self.assertEqual(recognizer.inference_count, 2)

    def test_probability_smoothing_reduces_single_frame_label_changes(self) -> None:
        recognizer = self.recognizer(
            sequence_length=1,
            confidence_threshold=0.6,
        )
        probabilities = (
            np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32),
            np.array([0.1, 0.9, 0.0, 0.0], dtype=np.float32),
            np.array([0.1, 0.9, 0.0, 0.0], dtype=np.float32),
        )

        with patch.object(
            recognizer,
            "_predict_probabilities",
            side_effect=probabilities,
        ):
            first = recognizer.update(1, LANDMARKS, PERSON_BOX, FRAME_SHAPE, 0)
            second = recognizer.update(1, LANDMARKS, PERSON_BOX, FRAME_SHAPE, 1)
            third = recognizer.update(1, LANDMARKS, PERSON_BOX, FRAME_SHAPE, 2)

        self.assertEqual(first.label, "walking")
        self.assertEqual(second.label, "unknown")
        self.assertEqual(third.label, "running")

    def test_inference_interval_reuses_the_last_smoothed_prediction(self) -> None:
        recognizer = self.recognizer(
            sequence_length=1,
            inference_interval=3,
            smoothing_window=1,
        )
        with patch.object(
            recognizer,
            "_predict_probabilities",
            side_effect=(
                np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32),
                np.array([0.1, 0.9, 0.0, 0.0], dtype=np.float32),
            ),
        ) as predict:
            first = recognizer.update(1, LANDMARKS, PERSON_BOX, FRAME_SHAPE, 0)
            held = recognizer.update(1, LANDMARKS, PERSON_BOX, FRAME_SHAPE, 1)
            later = recognizer.update(1, LANDMARKS, PERSON_BOX, FRAME_SHAPE, 3)

        self.assertEqual(first.label, "walking")
        self.assertEqual(held.label, "walking")
        self.assertEqual(later.label, "running")
        self.assertEqual(predict.call_count, 2)

    def test_track_removal_and_reset_clear_pose_history(self) -> None:
        recognizer = self.recognizer()
        recognizer.update(1, LANDMARKS, PERSON_BOX, FRAME_SHAPE, 0)
        recognizer.remove_track(1)
        self.assertIsNone(recognizer.update(1, LANDMARKS, PERSON_BOX, FRAME_SHAPE, 1))

        recognizer.update(2, LANDMARKS, PERSON_BOX, FRAME_SHAPE, 0)
        recognizer.reset()
        self.assertIsNone(recognizer.update(2, LANDMARKS, PERSON_BOX, FRAME_SHAPE, 1))


class MonitoringActivityIntegrationTest(unittest.TestCase):
    def processor(
        self,
        pose_frames: list[list[PoseResult]],
        activity_recognizer=_UNSET,
        **config_overrides,
    ) -> MonitoringProcessor:
        config = replace(
            DEFAULT_CONFIG,
            allow_model_downloads=False,
            event_logging_enabled=False,
            expression_interval_frames=1,
            **config_overrides,
        )
        processor_arguments = {
            "pose_analyzer": SequencePoseAnalyzer(pose_frames),
            "emotion_analyzer": None,
            "identity_matcher": None,
            "event_recorder": EventRecorder(None),
        }
        if activity_recognizer is not _UNSET:
            processor_arguments["activity_recognizer"] = activity_recognizer
        return MonitoringProcessor(config, **processor_arguments)

    def test_disabled_activity_does_not_load_the_live_model(self) -> None:
        with patch("activity_recognition.live.LiveMLPActivityRecognizer") as loader:
            processor = self.processor([], activity_model="none")
            try:
                self.assertFalse(processor.activity_enabled)
            finally:
                processor.close()

        loader.assert_not_called()

    def test_missing_checkpoint_is_ignored_when_activity_is_disabled(self) -> None:
        processor = self.processor(
            [],
            activity_model="none",
            activity_checkpoint_path=Path("missing-activity-checkpoint.pt"),
        )
        try:
            annotated = processor.process_frame(
                np.zeros(FRAME_SHAPE, dtype=np.uint8),
                timestamp=0.0,
            )
        finally:
            processor.close()

        self.assertEqual(annotated.shape, FRAME_SHAPE)

    def test_no_person_frame_skips_activity_inference(self) -> None:
        recognizer = RecordingActivityRecognizer()
        processor = self.processor([[]], recognizer)
        try:
            processor.process_frame(
                np.zeros(FRAME_SHAPE, dtype=np.uint8),
                timestamp=0.0,
            )
        finally:
            processor.close()

        self.assertEqual(recognizer.update_calls, [])

    def test_untracked_pose_skips_activity_inference(self) -> None:
        pose = PoseResult(None, PERSON_BOX, LANDMARKS)
        recognizer = RecordingActivityRecognizer()
        processor = self.processor([[pose]], recognizer)
        try:
            processor.process_frame(
                np.zeros(FRAME_SHAPE, dtype=np.uint8),
                timestamp=0.0,
            )
        finally:
            processor.close()

        self.assertEqual(recognizer.update_calls, [])

    def test_overlay_shows_activity_without_changing_review_tier(self) -> None:
        pose = PoseResult(7, PERSON_BOX, LANDMARKS)
        recognizer = RecordingActivityRecognizer(ActivityPrediction("walking", 0.82))
        processor = self.processor([[pose]], recognizer)
        rendered_text: list[str] = []
        original_put_text = cv2.putText

        def record_text(frame, text, *args, **kwargs):
            rendered_text.append(text)
            return original_put_text(frame, text, *args, **kwargs)

        try:
            with patch("detection.cv2.putText", side_effect=record_text):
                processor.process_frame(
                    np.zeros(FRAME_SHAPE, dtype=np.uint8),
                    timestamp=0.0,
                )
        finally:
            processor.close()

        snapshot = processor.last_snapshots[0]
        self.assertEqual(snapshot.activity_label, "walking")
        self.assertAlmostEqual(snapshot.activity_confidence, 0.82)
        self.assertEqual(snapshot.tier_label, "CLEAR")
        self.assertIn("Activity: Walking 82%", rendered_text)

    def test_expired_track_and_reset_clear_activity_state(self) -> None:
        pose = PoseResult(7, PERSON_BOX, LANDMARKS)
        recognizer = RecordingActivityRecognizer()
        processor = self.processor(
            [[pose], []],
            recognizer,
            stale_track_frames=0,
        )
        try:
            processor.process_frame(
                np.zeros(FRAME_SHAPE, dtype=np.uint8),
                timestamp=0.0,
            )
            processor.process_frame(
                np.zeros(FRAME_SHAPE, dtype=np.uint8),
                timestamp=1.0,
            )
            processor.reset_tracking(source="camera")
        finally:
            processor.close()

        self.assertEqual(recognizer.removed_tracks, [7])
        self.assertEqual(recognizer.reset_count, 1)


if __name__ == "__main__":
    unittest.main()
