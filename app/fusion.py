"""Combine visual and audio signals to decide when to trigger the relay.

Three sensitivity levels let the user trade reaction speed for accuracy:

- **fast** : any single strong signal fires (quickest, more false positives)
- **balanced** : visual alone fires, or horn + keyword together
- **accurate** : requires two or more agreeing signals
"""

from __future__ import annotations


class SignalFusion:

    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"
    _VALID = {FAST, BALANCED, ACCURATE}

    def __init__(self, sensitivity: str = "balanced") -> None:
        if sensitivity not in self._VALID:
            raise ValueError(
                f"Unknown sensitivity '{sensitivity}'. Choose from: {', '.join(sorted(self._VALID))}"
            )
        self.sensitivity = sensitivity

    def should_trigger(
        self,
        visual_score: float,
        visual_threshold: float,
        horn_confidence: float,
        keyword_detected: bool,
        audio_available: bool = False,
    ) -> bool:
        """Return True when the combined signals warrant a goal trigger."""
        # Small tolerance avoids UI rounding confusion, e.g. 0.6996 shown as 0.700.
        visual_hit = visual_score >= (visual_threshold - 0.001)
        horn_hit = horn_confidence >= 0.5

        # Without audio the only signal is visual, regardless of sensitivity.
        if not audio_available:
            return visual_hit

        if self.sensitivity == self.FAST:
            return visual_hit or horn_hit or keyword_detected

        if self.sensitivity == self.BALANCED:
            if visual_hit:
                return True
            if horn_hit and keyword_detected:
                return True
            return False

        # accurate
        signals = sum([visual_hit, horn_hit, keyword_detected])
        return signals >= 2
