"""Background thread that reads audio from a video file (via ffmpeg) or microphone."""

from __future__ import annotations

import subprocess
import sys
import threading
from typing import Protocol

import numpy as np


class AudioAnalyzer(Protocol):
    def analyze(self, chunk: np.ndarray, timestamp: float) -> None: ...


class AudioStream:
    """Reads audio and feeds chunks to registered analyzers in a daemon thread."""

    def __init__(
        self,
        source: str,
        sample_rate: int = 16000,
        chunk_duration: float = 0.5,
    ) -> None:
        self.source = source
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.chunk_duration = chunk_duration
        self._analyzers: list[AudioAnalyzer] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.active = False

    def register(self, analyzer: AudioAnalyzer) -> None:
        self._analyzers.append(analyzer)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        if self.source.isdigit():
            self._thread = threading.Thread(target=self._run_mic, daemon=True)
        else:
            self._thread = threading.Thread(target=self._run_ffmpeg, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self.active = False

    def _emit(self, chunk: np.ndarray, timestamp: float) -> None:
        for analyzer in self._analyzers:
            analyzer.analyze(chunk, timestamp)

    def _run_ffmpeg(self) -> None:
        """Extract audio from video file using ffmpeg subprocess."""
        cmd = [
            "ffmpeg", "-i", self.source,
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", "1", "-ar", str(self.sample_rate),
            "-v", "quiet", "pipe:1",
        ]
        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                startupinfo=startupinfo,
            )
        except FileNotFoundError:
            print("ffmpeg not found. Audio detection disabled.")
            print("Install ffmpeg and make sure it is on your PATH.")
            return

        self.active = True
        bytes_per_chunk = self.chunk_samples * 4  # float32 = 4 bytes
        timestamp = 0.0

        try:
            while not self._stop_event.is_set():
                data = proc.stdout.read(bytes_per_chunk)
                if not data:
                    break
                samples = np.frombuffer(data, dtype=np.float32)
                self._emit(samples, timestamp)
                timestamp += self.chunk_duration
        finally:
            proc.terminate()
            proc.wait()
            self.active = False

    def _run_mic(self) -> None:
        """Capture audio from the default microphone."""
        try:
            import sounddevice as sd
        except ImportError:
            print("sounddevice not installed. Live audio detection disabled.")
            print("Install with: pip install sounddevice")
            return

        import time as _time

        self.active = True
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_samples,
            ) as stream:
                while not self._stop_event.is_set():
                    data, _overflowed = stream.read(self.chunk_samples)
                    self._emit(data.flatten(), _time.time())
        except Exception as exc:
            print(f"Mic capture failed: {exc}")
        finally:
            self.active = False
