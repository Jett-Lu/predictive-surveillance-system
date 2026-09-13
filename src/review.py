from __future__ import annotations

from dataclasses import dataclass, field

from gesture import WaveAlertState


CLEAR_COLOR = (0, 255, 0)
MONITOR_COLOR = (0, 255, 255)
REVIEW_COLOR = (0, 165, 255)
HIGH_COLOR = (0, 0, 255)
CONCERN_EXPRESSIONS = frozenset(
    {
        "Anger",
        "Contempt",
        "Disgust",
        "Fear",
        "Sadness",
    }
)


@dataclass(frozen=True)
class ReviewState:
    tier_label: str
    color: tuple[int, int, int]
    score: float
    recent_wave_count: int
    concern_strength: float
    concern_label: str | None

    @property
    def concern_expression_active(self) -> bool:
        return self.concern_strength > 0.0


@dataclass
class ReviewLevelMonitor:
    """Keep uncertain expression context separate from behavior-based tiers."""

    wave_allowance: int = 2
    concern_smoothing_alpha: float = 0.35
    min_concern_confidence: float = 0.65
    _concern_strength: float = field(default=0.0, init=False)
    _concern_label: str | None = field(default=None, init=False)

    def observe_expression(
        self,
        label: str | None,
        confidence: float | None,
    ) -> bool:
        """Update the live concern signal and report when it first becomes visible."""
        previously_active = self._concern_strength > 0.0

        if label is None or confidence is None:
            self._concern_strength = 0.0
            self._concern_label = None
            return False

        target_strength = (
            float(confidence)
            if label in CONCERN_EXPRESSIONS and float(confidence) >= self.min_concern_confidence
            else 0.0
        )
        self._concern_strength = (
            self.concern_smoothing_alpha * target_strength
            + (1.0 - self.concern_smoothing_alpha) * self._concern_strength
        )
        if target_strength > 0:
            self._concern_label = label
        elif self._concern_strength < 0.01:
            self._concern_strength = 0.0
            self._concern_label = None

        return self._concern_strength > 0.0 and not previously_active

    def update(
        self,
        wave_state: WaveAlertState,
    ) -> ReviewState:
        behavior_score = max(0, wave_state.recent_wave_count - self.wave_allowance)
        score = float(behavior_score)

        if score == 0:
            tier = "CLEAR"
            color = CLEAR_COLOR
        elif score <= 2:
            tier = "MONITOR"
            color = MONITOR_COLOR
        elif score <= 4:
            tier = "REVIEW"
            color = REVIEW_COLOR
        else:
            tier = "HIGH"
            color = HIGH_COLOR

        return ReviewState(
            tier_label=tier,
            color=color,
            score=score,
            recent_wave_count=wave_state.recent_wave_count,
            concern_strength=self._concern_strength,
            concern_label=self._concern_label if self._concern_strength > 0.0 else None,
        )
