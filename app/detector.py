from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class GoalTemplateDetector:
    def __init__(self, template_path: str) -> None:
        template = cv2.imread(str(Path(template_path)), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(f"Template not found or unreadable: {template_path}")
        self.template = template
        self.height, self.width = self.template.shape[:2]

    def score(self, roi_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        if gray.shape[0] < self.height or gray.shape[1] < self.width:
            return 0.0

        result = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return float(max_val)
