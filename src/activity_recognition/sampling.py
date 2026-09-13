"""Source-time contract shared by new offline caches and live checkpoints.

Samples start at the first observation and use the most recent source frame at
each grid time. Short offline clips repeat their final frame.
"""
from __future__ import annotations

import math

DEFAULT_SAMPLING_INTERVAL_SECONDS = 0.1
SAMPLING_VERSION = 1


def validate_sampling_interval(value: float) -> float:
    interval = float(value)
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("sampling_interval_seconds must be finite and positive")
    return interval


def observation_gap_limit(interval: float) -> float:
    return max(0.5, interval * 3)
