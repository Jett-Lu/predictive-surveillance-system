"""Camera source helpers for OpenCV capture."""

from __future__ import annotations

import math
import os
import time

import cv2

DEFAULT_VIDEO_FPS = 24.0


class CaptureClock:
    """Use media time for recordings and monotonic time for live sources."""

    def __init__(self, capture: cv2.VideoCapture, *, recorded: bool) -> None:
        self.capture = capture
        self.recorded = recorded
        self._previous: float | None = None
        fps = capture.get(cv2.CAP_PROP_FPS) if recorded else DEFAULT_VIDEO_FPS
        self.fps = fps if math.isfinite(fps) and fps > 0 else DEFAULT_VIDEO_FPS

    def timestamp(self) -> float:
        """Read once after each successfully decoded frame."""
        if not self.recorded:
            return time.monotonic()
        position = self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if (
            not math.isfinite(position)
            or position < 0
            or (self._previous is not None and position <= self._previous)
        ):
            position = 0.0 if self._previous is None else self._previous + 1.0 / self.fps
        self._previous = position
        return position


def normalize_camera_source(source: str | int) -> int | str:
    """Convert numeric camera choices to OpenCV camera indexes."""
    if isinstance(source, int):
        return source

    stripped = source.strip()
    return int(stripped) if stripped.isdigit() else stripped


def prompt_camera_source(default: str = "1") -> int | str:
    """Ask for a camera index/path, defaulting to the usual external camera."""
    response = input(f"Camera source [{default}=external, 0=built-in]: ").strip()
    return normalize_camera_source(response or default)


def open_capture(source: str | int) -> cv2.VideoCapture:
    """Open a camera or video source with a Windows-friendly backend fallback."""
    normalized = normalize_camera_source(source)

    if isinstance(normalized, int) and os.name == "nt":
        capture = cv2.VideoCapture(normalized, cv2.CAP_DSHOW)
        if capture.isOpened():
            return capture
        capture.release()

    return cv2.VideoCapture(normalized)
