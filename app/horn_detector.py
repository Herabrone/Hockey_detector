"""Detect goal horn/buzzer using frequency-domain energy analysis."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from shutil import which

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
        reference_clip_path: str = "",
        reference_similarity_threshold: float = 0.85,
        reference_profile: np.ndarray | None = None,
        sustain_seconds: float = 1.0,
        chunk_duration: float = 0.5,
        max_history: int | None = 120,
    ) -> None:
        self.sample_rate = sample_rate
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.energy_threshold = energy_threshold
        self.reference_similarity_threshold = reference_similarity_threshold
        self.sustain_chunks = max(1, int(sustain_seconds / chunk_duration))
        self._lock = threading.Lock()
        self._events: list[tuple[float, float]] = []  # (timestamp, ratio)
        self._max_history = max_history
        self.reference_profile = reference_profile
        if self.reference_profile is None and reference_clip_path:
            self.reference_profile = self._load_reference_profile(
                reference_clip_path,
                sample_rate=self.sample_rate,
            )
        self.uses_reference_match = self.reference_profile is not None
        self._event_threshold = (
            self.reference_similarity_threshold if self.uses_reference_match else self.energy_threshold
        )

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

        band_ratio = band_energy / total_energy if total_energy > 0 else 0.0
        score = band_ratio
        if self.reference_profile is not None:
            score = self._reference_similarity(magnitudes)

        with self._lock:
            self._events.append((timestamp, score))
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
        spikes = sum(1 for r in relevant if r >= self._event_threshold)
        return min(1.0, spikes / self.sustain_chunks)

    @property
    def last_confidence(self) -> float:
        """Confidence computed from the most recent chunks (useful for live)."""
        with self._lock:
            if not self._events:
                return 0.0
            window = max(self.sustain_chunks * 2, 6)
            recent = self._events[-window:]
        spikes = sum(1 for _, r in recent if r >= self._event_threshold)
        return min(1.0, spikes / self.sustain_chunks)

    @property
    def last_timestamp(self) -> float:
        with self._lock:
            if not self._events:
                return 0.0
            return self._events[-1][0]

    @staticmethod
    def _normalized_spectrum(magnitudes: np.ndarray) -> np.ndarray:
        if magnitudes.size == 0:
            return magnitudes
        norm = float(np.linalg.norm(magnitudes))
        if norm == 0.0:
            return np.zeros_like(magnitudes, dtype=np.float32)
        return (magnitudes / norm).astype(np.float32)

    def _reference_similarity(self, magnitudes: np.ndarray) -> float:
        if self.reference_profile is None:
            return 0.0
        current_profile = self._normalized_spectrum(magnitudes)
        reference_profile = self.reference_profile
        if len(reference_profile) != len(current_profile):
            x_old = np.linspace(0.0, 1.0, num=len(reference_profile), dtype=np.float32)
            x_new = np.linspace(0.0, 1.0, num=len(current_profile), dtype=np.float32)
            reference_profile = np.interp(x_new, x_old, reference_profile).astype(np.float32)
            reference_profile = self._normalized_spectrum(reference_profile)
        return float(np.dot(current_profile, reference_profile))

    @classmethod
    def _load_reference_profile(cls, reference_clip_path: str, sample_rate: int | None = None) -> np.ndarray | None:
        clip_path = Path(reference_clip_path)
        if not clip_path.exists():
            print(f"Horn reference clip not found: {clip_path}")
            return None

        ffmpeg_cmd = cls._resolve_ffmpeg_command()
        if ffmpeg_cmd is None:
            print("ffmpeg not found. Horn reference matching disabled.")
            return None

        target_rate = sample_rate or 16000
        cmd = [
            ffmpeg_cmd,
            "-i", str(clip_path),
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ac", "1",
            "-ar", str(target_rate),
            "-v", "quiet",
            "pipe:1",
        ]
        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                startupinfo=startupinfo,
                check=False,
            )
        except Exception as exc:
            print(f"Failed to load horn reference clip: {exc}")
            return None

        if proc.returncode != 0 or not proc.stdout:
            print("Horn reference matching disabled; could not decode reference clip.")
            return None

        samples = np.frombuffer(proc.stdout, dtype=np.float32)
        if len(samples) < 2:
            print("Horn reference matching disabled; reference clip is too short.")
            return None

        magnitudes = np.abs(np.fft.rfft(samples))
        return cls._normalized_spectrum(magnitudes)

    @staticmethod
    def _resolve_ffmpeg_command() -> str | None:
        ffmpeg_path = which("ffmpeg")
        if ffmpeg_path:
            return ffmpeg_path

        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None
