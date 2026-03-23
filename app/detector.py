from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class GoalTemplateDetector:
    def __init__(self, template_path: str, team_templates: dict[str, str] | None = None) -> None:
        self.template = self._load_template(template_path)
        self.height, self.width = self.template.shape[:2]
        self.team_templates = self._load_team_templates(team_templates or {})

    @staticmethod
    def _load_template(template_path: str) -> np.ndarray:
        template = cv2.imread(str(Path(template_path)), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(f"Template not found or unreadable: {template_path}")
        return template

    def _load_team_templates(self, team_templates: dict[str, str]) -> dict[str, np.ndarray]:
        loaded_templates: dict[str, np.ndarray] = {}
        for team_name, template_path in team_templates.items():
            path = Path(template_path)
            if not path.exists():
                continue
            template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if template is None:
                continue
            loaded_templates[team_name] = template
        return loaded_templates

    @staticmethod
    def _match_score(gray: np.ndarray, template: np.ndarray) -> float:
        if gray.shape[0] < template.shape[0] or gray.shape[1] < template.shape[1]:
            return 0.0

        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return float(max_val)

    def score(self, roi_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        return self._match_score(gray, self.template)

    def detect_team(self, roi_bgr: np.ndarray) -> str:
        if not self.team_templates:
            return "unknown"

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        best_team = "unknown"
        best_score = float("-inf")

        for team_name, template in self.team_templates.items():
            score = self._match_score(gray, template)
            if score > best_score:
                best_score = score
                best_team = team_name

        return best_team
