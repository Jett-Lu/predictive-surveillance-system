"""Read-only assets included in source and wheel installations."""

from pathlib import Path

FACE_DETECTOR_PATH = Path(__file__).with_name("blaze_face_short_range.tflite")
