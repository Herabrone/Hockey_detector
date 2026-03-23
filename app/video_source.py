from __future__ import annotations

from pathlib import Path

import cv2


class VideoSource:
    def __init__(self, source: str) -> None:
        self.source = self._normalize_source(source)
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

    def release(self) -> None:
        self.cap.release()
