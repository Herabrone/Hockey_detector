"""Background thread that reads audio from a video file or stream via ffmpeg."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from shutil import which
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
        input_format: str | None = None,
    ) -> None:
        self.source = source
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.chunk_duration = chunk_duration
        self.input_format = input_format
        self.is_file_source = self._detect_file_source(source, input_format)
        self._analyzers: list[AudioAnalyzer] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.active = False
        self.had_output = False

    def register(self, analyzer: AudioAnalyzer) -> None:
        self._analyzers.append(analyzer)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
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
        ffmpeg_cmd = self._resolve_ffmpeg_command()
        if ffmpeg_cmd is None:
            print("ffmpeg not found. Audio detection disabled.")
            print("Run: pip install imageio-ffmpeg  or install ffmpeg on PATH.")
            return

        cmd = [ffmpeg_cmd]
        if self.input_format:
            cmd += ["-f", self.input_format]
        cmd += [
            "-i", self.source,
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
        except Exception as exc:
            print(f"Audio stream start failed: {exc}")
            return

        self.active = True
        bytes_per_chunk = self.chunk_samples * 4  # float32 = 4 bytes
        timestamp = 0.0

        try:
            while not self._stop_event.is_set():
                if proc.stdout is None:
                    break
                data = proc.stdout.read(bytes_per_chunk)
                if not data:
                    break
                self.had_output = True
                samples = np.frombuffer(data, dtype=np.float32)
                self._emit(samples, timestamp)
                timestamp += self.chunk_duration
        finally:
            proc.terminate()
            proc.wait()
            self.active = False

    @staticmethod
    def _resolve_ffmpeg_command() -> str | None:
        """Pick ffmpeg from PATH first, then from imageio-ffmpeg package."""
        ffmpeg_path = which("ffmpeg")
        if ffmpeg_path:
            return ffmpeg_path

        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None

    @staticmethod
    def _detect_file_source(source: str, input_format: str | None) -> bool:
        """Best-effort check for regular media files such as MP4."""
        if input_format:
            return False
        if source.isdigit():
            return False
        return Path(source).exists()
