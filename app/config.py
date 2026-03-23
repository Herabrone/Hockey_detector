from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ROIConfig:
    y1: int = 0
    y2: int = 150
    x1: int = 0
    x2: int = -1  # -1 means full width


@dataclass
class AppConfig:
    video_source: str = "0"  # camera index as string or video file path
    output_width: int = 640
    output_height: int = 360
    threshold: float = 0.7
    cooldown_seconds: float = 10.0
    show_window: bool = True
    show_roi_box: bool = True
    template_path: str = "assets/goal_template.png"
    relay_mode: str = "print"  # print or usb_serial
    serial_port: str = "COM3"
    serial_baudrate: int = 9600
    serial_command: str = "ON\\n"
    team_templates: dict[str, str] = field(default_factory=dict)
    roi: ROIConfig = field(default_factory=ROIConfig)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    roi_raw = raw.get("roi", {}) if isinstance(raw, dict) else {}
    team_templates_raw = raw.get("team_templates", {}) if isinstance(raw, dict) else {}
    roi = ROIConfig(
        y1=_as_int(roi_raw.get("y1"), 0),
        y2=_as_int(roi_raw.get("y2"), 150),
        x1=_as_int(roi_raw.get("x1"), 0),
        x2=_as_int(roi_raw.get("x2"), -1),
    )
    team_templates = {
        str(name): str(template_path)
        for name, template_path in team_templates_raw.items()
    } if isinstance(team_templates_raw, dict) else {}

    cfg = AppConfig(
        video_source=str(raw.get("video_source", "0")),
        output_width=_as_int(raw.get("output_width"), 640),
        output_height=_as_int(raw.get("output_height"), 360),
        threshold=_as_float(raw.get("threshold"), 0.7),
        cooldown_seconds=_as_float(raw.get("cooldown_seconds"), 10.0),
        show_window=bool(raw.get("show_window", True)),
        show_roi_box=bool(raw.get("show_roi_box", True)),
        template_path=str(raw.get("template_path", "assets/goal_template.png")),
        relay_mode=str(raw.get("relay_mode", "print")),
        serial_port=str(raw.get("serial_port", "COM3")),
        serial_baudrate=_as_int(raw.get("serial_baudrate"), 9600),
        serial_command=str(raw.get("serial_command", "ON\\n")),
        team_templates=team_templates,
        roi=roi,
    )

    return cfg
