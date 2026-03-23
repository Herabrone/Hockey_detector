from __future__ import annotations

import time


class CooldownGate:
    def __init__(self, cooldown_seconds: float) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.last_trigger = 0.0

    def can_trigger(self, now: float | None = None) -> bool:
        now_val = now if now is not None else time.time()
        return (now_val - self.last_trigger) >= self.cooldown_seconds

    def mark_triggered(self, now: float | None = None) -> None:
        self.last_trigger = now if now is not None else time.time()
