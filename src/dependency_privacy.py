"""Dependency opt-outs; these are not an operating-system network sandbox."""

import os


def disable_ultralytics_telemetry() -> None:
    # Set before import: suppress its connectivity check and Sentry startup.
    # Application-managed, hash-verified downloads remain available.
    os.environ["YOLO_OFFLINE"] = "true"
    from ultralytics import settings
    from ultralytics.utils.events import events

    settings.update({"sync": False})
    # The collector may already exist if a library user imported YOLO first.
    events.enabled = False


def disable_onnx_telemetry() -> None:
    import onnxruntime

    onnxruntime.disable_telemetry_events()
