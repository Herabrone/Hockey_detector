from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from app.config import AppConfig, load_config
from app.detector import GoalTemplateDetector
from app.gate import CooldownGate
from app.relay import PrintRelay, UsbSerialRelay
from app.video_source import VideoSource


def _build_relay(cfg: AppConfig):
    if cfg.relay_mode == "usb_serial":
        return UsbSerialRelay(
            port=cfg.serial_port,
            baudrate=cfg.serial_baudrate,
            command=cfg.serial_command,
        )
    return PrintRelay()


def _compute_roi(frame, cfg: AppConfig):
    h, w = frame.shape[:2]
    x1 = max(0, min(cfg.roi.x1, w - 1))
    x2 = w if cfg.roi.x2 < 0 else max(x1 + 1, min(cfg.roi.x2, w))
    y1 = max(0, min(cfg.roi.y1, h - 1))
    y2 = max(y1 + 1, min(cfg.roi.y2, h))
    return x1, y1, x2, y2


def run(cfg: AppConfig) -> None:
    source = VideoSource(cfg.video_source)
    if not source.is_open():
        raise RuntimeError(f"Unable to open video source: {cfg.video_source}")

    detector = None
    template_path = Path(cfg.template_path)
    if template_path.exists():
        detector = GoalTemplateDetector(str(template_path))
    else:
        print(f"Template missing: {template_path}. Detection is disabled.")

    gate = CooldownGate(cfg.cooldown_seconds)
    relay = _build_relay(cfg)

    print("Press ESC or q to quit.")

    try:
        while True:
            ret, frame = source.read()
            if not ret:
                print("Frame read failed or stream ended.")
                break

            frame = cv2.resize(frame, (cfg.output_width, cfg.output_height))
            x1, y1, x2, y2 = _compute_roi(frame, cfg)
            roi = frame[y1:y2, x1:x2]

            score = 0.0
            if detector is not None:
                score = detector.score(roi)
                if score >= cfg.threshold and gate.can_trigger():
                    print(f"GOAL DETECTED score={score:.3f} at {time.strftime('%H:%M:%S')}")
                    relay.trigger()
                    gate.mark_triggered()

            if cfg.show_window:
                display = frame.copy()
                if cfg.show_roi_box:
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    display,
                    f"score={score:.3f} threshold={cfg.threshold:.2f}",
                    (10, cfg.output_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("feed", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        source.release()
        if hasattr(relay, "close"):
            relay.close()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NHL goal overlay detector MVP")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
