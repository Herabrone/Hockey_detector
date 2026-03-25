"""Spot goal-related keywords in audio using vosk speech recognition.

vosk is an optional dependency.  If not installed or no model is configured,
keyword detection is silently disabled.
"""

from __future__ import annotations

import json
import threading

import numpy as np

_DEFAULT_KEYWORDS: set[str] = {"score", "scores", "goal", "he scores"}


class KeywordDetector:
    """Offline keyword spotting backed by vosk."""

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        keywords: set[str] | None = None,
        max_history: int | None = 60,
    ) -> None:
        self.keywords = keywords or _DEFAULT_KEYWORDS
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._events: list[tuple[float, str]] = []  # (timestamp, matched text)
        self._max_history = max_history
        self._recognizer = None

        try:
            from vosk import KaldiRecognizer, Model

            model = Model(model_path)
            self._recognizer = KaldiRecognizer(model, sample_rate)
        except ImportError:
            print("vosk not installed. Keyword detection disabled.")
            print("Install with: pip install vosk")
        except Exception as exc:
            print(f"Failed to load vosk model at {model_path}: {exc}")

    @property
    def available(self) -> bool:
        return self._recognizer is not None

    # ------------------------------------------------------------------
    # Called from the audio thread
    # ------------------------------------------------------------------

    def analyze(self, chunk: np.ndarray, timestamp: float) -> None:
        if self._recognizer is None:
            return

        pcm_data = (chunk * 32767).astype(np.int16).tobytes()

        if self._recognizer.AcceptWaveform(pcm_data):
            result = json.loads(self._recognizer.Result())
            text = result.get("text", "").lower()
        else:
            partial = json.loads(self._recognizer.PartialResult())
            text = partial.get("partial", "").lower()

        if any(kw in text for kw in self.keywords):
            with self._lock:
                self._events.append((timestamp, text))
                if self._max_history is not None and len(self._events) > self._max_history:
                    self._events = self._events[-self._max_history:]

    # ------------------------------------------------------------------
    # Queried from the main (video) thread
    # ------------------------------------------------------------------

    def detected_at(self, timestamp: float, window: float = 5.0) -> bool:
        """True if a keyword was spotted within ±window of *timestamp*."""
        with self._lock:
            return any(abs(t - timestamp) <= window for t, _ in self._events)
