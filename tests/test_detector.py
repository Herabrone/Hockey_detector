from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from app.detector import GoalTemplateDetector


BASE_PATTERN = np.array(
    [
        [255, 128, 64],
        [128, 64, 128],
        [64, 128, 255],
    ],
    dtype=np.uint8,
)

HOME_PATTERN = np.array(
    [
        [0, 255, 0],
        [255, 255, 0],
        [0, 255, 255],
    ],
    dtype=np.uint8,
)

AWAY_PATTERN = np.array(
    [
        [255, 0, 0],
        [255, 0, 255],
        [0, 0, 255],
    ],
    dtype=np.uint8,
)


def _write_template(path: Path, pattern: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), pattern)
    assert ok


def _to_bgr(pattern: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR) if len(pattern.shape) == 2 else pattern


def test_detector_scores_matching_template(tmp_path: Path) -> None:
    template_path = tmp_path / "goal_template.png"
    _write_template(template_path, BASE_PATTERN)

    detector = GoalTemplateDetector(str(template_path))
    roi = _to_bgr(BASE_PATTERN)

    assert detector.score(roi) > 0.99


def test_detector_detects_best_team(tmp_path: Path) -> None:
    goal_template_path = tmp_path / "goal_template.png"
    home_template_path = tmp_path / "home_template.png"
    away_template_path = tmp_path / "away_template.png"

    _write_template(goal_template_path, BASE_PATTERN)
    _write_template(home_template_path, HOME_PATTERN)
    _write_template(away_template_path, AWAY_PATTERN)

    detector = GoalTemplateDetector(
        str(goal_template_path),
        {
            "home": str(home_template_path),
            "away": str(away_template_path),
        },
    )

    home_roi = _to_bgr(HOME_PATTERN)

    assert detector.detect_team(home_roi) == "home"


def test_detector_returns_unknown_without_team_templates(tmp_path: Path) -> None:
    template_path = tmp_path / "goal_template.png"
    _write_template(template_path, BASE_PATTERN)

    detector = GoalTemplateDetector(str(template_path))
    roi = _to_bgr(BASE_PATTERN)

    assert detector.detect_team(roi) == "unknown"


def test_detector_raises_for_missing_goal_template() -> None:
    with pytest.raises(FileNotFoundError):
        GoalTemplateDetector("missing_goal_template.png")
