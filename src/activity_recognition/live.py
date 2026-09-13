"""Optional per-track MLP activity inference for the live pose pipeline."""

from __future__ import annotations

import pickle
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from activity_recognition.labels import ACTIVITY_LABELS, LABEL_TO_INDEX
from activity_recognition.preprocessing import KEYPOINT_COUNT, normalize_pose
from activity_recognition.sampling import observation_gap_limit, validate_sampling_interval


UNKNOWN_ACTIVITY = "unknown"


class ActivityModelError(RuntimeError):
    """Raised when an explicitly enabled activity checkpoint cannot be used."""


@dataclass(frozen=True)
class ActivityPrediction:
    label: str
    confidence: float


@dataclass
class _TrackActivityState:
    pose_history: deque[np.ndarray]
    valid_history: deque[bool]
    probability_history: deque[np.ndarray]
    last_inference_frame: int = -1
    prediction: ActivityPrediction | None = None
    last_observation_timestamp: float | None = None
    next_sample_timestamp: float | None = None
    last_observation_pose: np.ndarray | None = None


class LiveMLPActivityRecognizer:
    """Classify fixed-length normalized pose sequences independently per track."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        sequence_length: int = 16,
        confidence_threshold: float = 0.5,
        inference_interval: int = 5,
        smoothing_window: int = 5,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("Activity sequence length must be positive")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("Activity confidence threshold must be between 0 and 1")
        if inference_interval < 1:
            raise ValueError("Activity inference interval must be positive")
        if smoothing_window < 1:
            raise ValueError("Activity smoothing window must be positive")

        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.inference_interval = inference_interval
        self.smoothing_window = smoothing_window
        self.sampling_interval_seconds: float | None = None
        self._track_states: dict[int, _TrackActivityState] = {}
        self._inference_latency_total_ms = 0.0
        self._inference_count = 0

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise ActivityModelError(
                f"Activity MLP checkpoint not found: {checkpoint_path}"
            )

        try:
            import torch

            from activity_recognition.models import build_activity_classifier

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(checkpoint, dict):
                raise ActivityModelError(
                    f"Activity checkpoint must contain a mapping: {checkpoint_path}"
                )
            self._validate_checkpoint(checkpoint, checkpoint_path)
            self._model = build_activity_classifier(
                "mlp",
                int(checkpoint["input_dim"]),
            )
            self._model.load_state_dict(checkpoint["state_dict"])
            if any(
                not bool(torch.isfinite(parameter).all())
                for parameter in self._model.parameters()
            ):
                raise ActivityModelError(
                    f"Activity checkpoint contains non-finite weights: "
                    f"{checkpoint_path}"
                )
            self._model.eval()
            self._feature_mean = checkpoint["feature_mean"].detach().cpu().float()
            self._feature_std = checkpoint["feature_std"].detach().cpu().float()
            self._torch = torch
        except ActivityModelError:
            raise
        except (
            ImportError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            pickle.UnpicklingError,
        ) as exc:
            raise ActivityModelError(
                f"Activity MLP checkpoint is incompatible: {checkpoint_path}: {exc}"
            ) from exc

    @property
    def average_inference_latency_ms(self) -> float | None:
        if not self._inference_count:
            return None
        return self._inference_latency_total_ms / self._inference_count

    @property
    def inference_count(self) -> int:
        return self._inference_count

    def update(
        self,
        track_key: int,
        landmarks: dict[int, tuple[float, float]],
        box: tuple[int, int, int, int],
        frame_shape: tuple[int, ...],
        frame_number: int,
        *,
        timestamp: float | None = None,
    ) -> ActivityPrediction | None:
        started = perf_counter()
        state = self._track_states.setdefault(track_key, self._new_track_state())
        pose_features = normalize_pose(landmarks, box, frame_shape)
        if not np.isfinite(pose_features).all():
            pose_features = np.zeros_like(pose_features)
        valid_pose = bool(np.any(pose_features[:, 2] > 0.0))
        interval = getattr(self, "sampling_interval_seconds", None)
        if interval is not None:
            observation_time = perf_counter() if timestamp is None else float(timestamp)
            if not np.isfinite(observation_time):
                raise ValueError("Activity timestamp must be finite")
            previous_time = state.last_observation_timestamp
            if previous_time is not None and (
                observation_time < previous_time
                or observation_time - previous_time > observation_gap_limit(interval)
            ):
                state = self._new_track_state()
                self._track_states[track_key] = state
            if state.next_sample_timestamp is None:
                state.next_sample_timestamp = observation_time
            sampled = False
            while state.next_sample_timestamp <= observation_time + 1e-8:
                # Hold the most recent observation at each source-time grid point.
                sampled_pose = (
                    state.last_observation_pose
                    if state.next_sample_timestamp < observation_time - 1e-8
                    and state.last_observation_pose is not None else pose_features
                )
                state.pose_history.append(sampled_pose)
                state.valid_history.append(bool(np.any(sampled_pose[:, 2] > 0)))
                state.next_sample_timestamp += interval
                sampled = True
            state.last_observation_timestamp = observation_time
            state.last_observation_pose = pose_features
            if not sampled:
                return state.prediction
        else:
            state.pose_history.append(pose_features)
            state.valid_history.append(valid_pose)

        if len(state.pose_history) < self.sequence_length:
            return state.prediction
        if not any(state.valid_history):
            state.probability_history.clear()
            state.prediction = ActivityPrediction(UNKNOWN_ACTIVITY, 0.0)
            return state.prediction
        if not all(state.valid_history):
            # Tolerate brief pose loss, but do not hold a label indefinitely
            # when intermittent missing poses keep blocking inference.
            max_prediction_age = max(self.sequence_length, self.inference_interval)
            if (
                state.last_inference_frame >= 0
                and frame_number - state.last_inference_frame >= max_prediction_age
            ):
                state.probability_history.clear()
                state.prediction = ActivityPrediction(UNKNOWN_ACTIVITY, 0.0)
            return state.prediction
        if (
            state.last_inference_frame >= 0
            and frame_number - state.last_inference_frame < self.inference_interval
        ):
            return state.prediction

        pose_sequence = np.stack(state.pose_history).astype(np.float32, copy=False)
        if not np.any(pose_sequence):
            state.probability_history.clear()
            state.prediction = ActivityPrediction(UNKNOWN_ACTIVITY, 0.0)
            return state.prediction

        probabilities = self._predict_probabilities(pose_sequence)
        state.probability_history.append(probabilities)
        smoothed = np.mean(state.probability_history, axis=0)
        predicted_index = int(np.argmax(smoothed))
        confidence = float(smoothed[predicted_index])
        label = (
            ACTIVITY_LABELS[predicted_index]
            if confidence >= self.confidence_threshold
            else UNKNOWN_ACTIVITY
        )
        state.prediction = ActivityPrediction(label, confidence)
        state.last_inference_frame = frame_number
        self._inference_latency_total_ms += (perf_counter() - started) * 1000.0
        self._inference_count += 1
        return state.prediction

    def remove_track(self, track_key: int) -> None:
        self._track_states.pop(track_key, None)

    def reset(self) -> None:
        self._track_states.clear()

    def _new_track_state(self) -> _TrackActivityState:
        return _TrackActivityState(
            pose_history=deque(maxlen=self.sequence_length),
            valid_history=deque(maxlen=self.sequence_length),
            probability_history=deque(maxlen=self.smoothing_window),
        )

    def _predict_probabilities(self, pose_sequence: np.ndarray) -> np.ndarray:
        flattened = pose_sequence.reshape(1, -1)
        features = self._torch.from_numpy(flattened).float()
        standardized = (features - self._feature_mean) / self._feature_std
        with self._torch.inference_mode():
            logits = self._model(standardized)
            probabilities = self._torch.softmax(logits, dim=1)
        values = probabilities.squeeze(0).cpu().numpy()
        if values.shape != (len(ACTIVITY_LABELS),) or not np.isfinite(values).all():
            raise ActivityModelError(
                "Activity MLP produced invalid class probabilities"
            )
        return values

    def _validate_checkpoint(
        self,
        checkpoint: dict[str, Any],
        checkpoint_path: Path,
    ) -> None:
        import torch

        if checkpoint.get("model_name") != "mlp":
            raise ActivityModelError(
                f"Activity checkpoint is not an MLP checkpoint: {checkpoint_path}"
            )
        if (
            tuple(checkpoint.get("labels", ())) != ACTIVITY_LABELS
            or checkpoint.get("label_to_index") != LABEL_TO_INDEX
        ):
            raise ActivityModelError(
                "Activity checkpoint labels must be walking, running, "
                "standing, sitting in that order"
            )
        checkpoint_sequence_length = checkpoint.get("frames_per_sample")
        if checkpoint_sequence_length != self.sequence_length:
            raise ActivityModelError(
                "Activity sequence length does not match checkpoint: "
                f"configured {self.sequence_length}, checkpoint "
                f"{checkpoint_sequence_length}"
            )
        if "sampling_interval_seconds" in checkpoint:
            try:
                self.sampling_interval_seconds = validate_sampling_interval(
                    checkpoint["sampling_interval_seconds"]
                )
            except (TypeError, ValueError) as exc:
                raise ActivityModelError(f"Invalid checkpoint sampling interval: {exc}") from exc
        else:
            warnings.warn(
                "Legacy activity checkpoint has no source-time sampling metadata; "
                "using adjacent observations. Retrain with new activity caches for "
                "consistent offline/live timing.",
                UserWarning, stacklevel=2,
            )
        expected_input_dim = self.sequence_length * KEYPOINT_COUNT * 3
        if checkpoint.get("input_dim") != expected_input_dim:
            raise ActivityModelError(
                "Activity checkpoint input width does not match normalized "
                f"pose sequences: expected {expected_input_dim}"
            )

        mean = checkpoint.get("feature_mean")
        standard_deviation = checkpoint.get("feature_std")
        if not isinstance(mean, torch.Tensor) or not isinstance(
            standard_deviation,
            torch.Tensor,
        ):
            raise ActivityModelError(
                f"Activity checkpoint is missing feature scaling: {checkpoint_path}"
            )
        if tuple(mean.shape) != (expected_input_dim,) or tuple(
            standard_deviation.shape
        ) != (expected_input_dim,):
            raise ActivityModelError(
                f"Activity checkpoint feature scaling has an invalid shape: "
                f"{checkpoint_path}"
            )
        if not bool(torch.isfinite(mean).all()) or not bool(
            torch.isfinite(standard_deviation).all()
        ):
            raise ActivityModelError(
                f"Activity checkpoint feature scaling is not finite: {checkpoint_path}"
            )
        if bool((standard_deviation <= 0).any()):
            raise ActivityModelError(
                f"Activity checkpoint feature standard deviation is invalid: "
                f"{checkpoint_path}"
            )
