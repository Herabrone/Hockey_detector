from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from app.config import load_config
from app.video_source import VideoSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture GOAL template from ROI")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config yaml")
    parser.add_argument(
        "--output",
        default="assets/goal_template.png",
        help="Output path for template image",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    source = VideoSource(cfg.video_source)
    if not source.is_open():
        raise RuntimeError(f"Unable to open source: {cfg.video_source}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            ret, frame = source.read()
            if not ret:
                break

            frame = cv2.resize(frame, (cfg.output_width, cfg.output_height))
            h, w = frame.shape[:2]
            x1 = max(0, min(cfg.roi.x1, w - 1))
            x2 = w if cfg.roi.x2 < 0 else max(x1 + 1, min(cfg.roi.x2, w))
            y1 = max(0, min(cfg.roi.y1, h - 1))
            y2 = max(y1 + 1, min(cfg.roi.y2, h))
            roi = frame[y1:y2, x1:x2]

            preview = frame.copy()
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                preview,
                "Press s to save ROI as template. Press q or ESC to quit.",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imshow("template-capture", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                cv2.imwrite(str(output_path), roi)
                print(f"Saved template to {output_path}")
                break
            if key in (27, ord("q")):
                break
    finally:
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
