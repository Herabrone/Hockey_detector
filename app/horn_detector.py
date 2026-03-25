"""Detect goal horn/buzzer using frequency-domain energy analysis."""

from __future__ import annotations

import threading

import numpy as np


class HornDetector:
    """Looks for sustained energy in the goal horn frequency band via FFT.

    NHL goal horns typically sit in the 200-1000 Hz range.  When the ratio of
    energy in that band to total energy exceeds *energy_threshold* for enough
    consecutive chunks, the detector reports high confidence.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        freq_low: int = 200,
        freq_high: int = 1000,
        energy_threshold: float = 0.4,
        sustain_seconds: float = 1.0,
        chunk_duration: float = 0.5,
        max_history: int | None = 120,
    ) -> None:
        self.sample_rate = sample_rate
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.energy_threshold = energy_threshold
        self.sustain_chunks = max(1, int(sustain_seconds / chunk_duration))
        self._lock = threading.Lock()
        self._events: list[tuple[float, float]] = []  # (timestamp, ratio)
        self._max_history = max_history

    # ------------------------------------------------------------------
    # Called from the audio thread
    # ------------------------------------------------------------------

    def analyze(self, chunk: np.ndarray, timestamp: float) -> None:
        if len(chunk) < 2:
            return

        fft_vals = np.fft.rfft(chunk)
        magnitudes = np.abs(fft_vals)
        freqs = np.fft.rfftfreq(len(chunk), d=1.0 / self.sample_rate)

        band_mask = (freqs >= self.freq_low) & (freqs <= self.freq_high)
        band_energy = float(np.sum(magnitudes[band_mask] ** 2))
        total_energy = float(np.sum(magnitudes ** 2))

        ratio = band_energy / total_energy if total_energy > 0 else 0.0

        with self._lock:
            self._events.append((timestamp, ratio))
            if self._max_history is not None and len(self._events) > self._max_history:
                self._events = self._events[-self._max_history:]

    # ------------------------------------------------------------------
    # Queried from the main (video) thread
    # ------------------------------------------------------------------

    def confidence_at(self, timestamp: float, window: float = 3.0) -> float:
        """Horn confidence around *timestamp* (0.0 to 1.0).

        Counts how many chunks within ±window had horn-band energy above the
        threshold, then divides by the minimum sustained-chunk count.
        """
        with self._lock:
            relevant = [r for t, r in self._events if abs(t - timestamp) <= window]
        if not relevant:
            return 0.0
        spikes = sum(1 for r in relevant if r >= self.energy_threshold)
        return min(1.0, spikes / self.sustain_chunks)

    @property
    def last_confidence(self) -> float:
        """Confidence computed from the most recent chunks (useful for live)."""
        with self._lock:
            if not self._events:
                return 0.0
            window = max(self.sustain_chunks * 2, 6)
            recent = self._events[-window:]
        spikes = sum(1 for _, r in recent if r >= self.energy_threshold)
        return min(1.0, spikes / self.sustain_chunks)
