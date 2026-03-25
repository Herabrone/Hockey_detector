"""Background thread that reads audio from a video file or stream via ffmpeg."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from shutil import which
from typing import Callable, Protocol

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
        video_position_callback: Callable[[], float] | None = None,
    ) -> None:
        self.source = source
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.chunk_duration = chunk_duration
        self.input_format = input_format
        self.is_file_source = self._detect_file_source(source, input_format)
        self._analyzers: list[AudioAnalyzer] = []
        self._video_position_callback = video_position_callback
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.active = False
        self.had_output = False
        self.started_at: float | None = None
        self.last_output_at: float | None = None
        self.last_timestamp = 0.0
        self.process_returncode: int | None = None
        self.last_error: str = ""
        self._stderr_lines: list[str] = []
        self._warned_no_output = False
        self._warned_stalled = False

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

    def seconds_since_start(self) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, time.time() - self.started_at)

    def seconds_since_output(self) -> float | None:
        if self.last_output_at is None:
            return None
        return max(0.0, time.time() - self.last_output_at)

    def is_stalled(self, timeout_seconds: float) -> bool:
        if not self.active or not self.had_output:
            return False
        since_output = self.seconds_since_output()
        return since_output is not None and since_output > timeout_seconds

    def _emit(self, chunk: np.ndarray, timestamp: float) -> None:
        for analyzer in self._analyzers:
            analyzer.analyze(chunk, timestamp)

    def _read_stderr(self, pipe) -> None:
        if pipe is None:
            return
        try:
            for raw_line in iter(pipe.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                self._stderr_lines.append(line)
                self._stderr_lines = self._stderr_lines[-20:]
        finally:
            pipe.close()

    def _current_video_position(self) -> float | None:
        if self._video_position_callback is None:
            return None
        try:
            position = float(self._video_position_callback())
        except Exception:
            return None
        return max(0.0, position)

    def _chunk_timestamp(self, fallback_timestamp: float) -> float:
        position = self._current_video_position()
        if position is None:
            return fallback_timestamp
        self.last_timestamp = max(self.last_timestamp, position)
        return self.last_timestamp

    def _report_process_issue(self) -> None:
        if self.process_returncode in (None, 0):
            return
        if self._stop_event.is_set():
            return
        details = self.last_error or "No ffmpeg stderr output captured."
        print(f"Audio stream exited with code {self.process_returncode}: {details}")

    def _run_ffmpeg(self) -> None:
        """Extract audio from video file using ffmpeg subprocess."""
        ffmpeg_cmd = self._resolve_ffmpeg_command()
        if ffmpeg_cmd is None:
            print("ffmpeg not found. Audio detection disabled.")
            print("Run: pip install imageio-ffmpeg  or install ffmpeg on PATH.")
            return

        cmd = [ffmpeg_cmd]
        if self.is_file_source:
            # Pace file decoding to wall clock so callback-based timestamps stay aligned.
            cmd.append("-re")
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
                stderr=subprocess.PIPE,
                startupinfo=startupinfo,
            )
        except Exception as exc:
            print(f"Audio stream start failed: {exc}")
            return

        self.active = True
        self.started_at = time.time()
        bytes_per_chunk = self.chunk_samples * 4  # float32 = 4 bytes
        fallback_timestamp = 0.0
        stderr_thread = threading.Thread(target=self._read_stderr, args=(proc.stderr,), daemon=True)
        stderr_thread.start()

        try:
            while not self._stop_event.is_set():
                if proc.stdout is None:
                    break
                data = proc.stdout.read(bytes_per_chunk)
                if not data:
                    break
                self.had_output = True
                self.last_output_at = time.time()
                samples = np.frombuffer(data, dtype=np.float32)
                timestamp = self._chunk_timestamp(fallback_timestamp)
                self._emit(samples, timestamp)
                fallback_timestamp += self.chunk_duration
        finally:
            if proc.poll() is None:
                proc.terminate()
            self.process_returncode = proc.wait()
            stderr_thread.join(timeout=1)
            if self._stderr_lines:
                self.last_error = self._stderr_lines[-1]
            self._report_process_issue()
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
