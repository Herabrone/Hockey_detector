from __future__ import annotations

import time as _time
from pathlib import Path

import cv2


class VideoSource:
    def __init__(self, source: str) -> None:
        self.source = self._normalize_source(source)
        self._is_live = isinstance(self.source, int)
        self.cap = cv2.VideoCapture(self.source)

    @staticmethod
    def _normalize_source(source: str):
        # Numeric values are treated as camera indices, everything else as file path.
        if source.isdigit():
            return int(source)
        return str(Path(source))

    def is_open(self) -> bool:
        return self.cap.isOpened()

    def read(self):
        return self.cap.read()

    def timestamp(self) -> float:
        """Current position in seconds (video-time for files, wall-clock for live)."""
        if self._is_live:
            return _time.time()
        return self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    def release(self) -> None:
        self.cap.release()
