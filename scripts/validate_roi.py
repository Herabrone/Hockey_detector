from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from app.config import load_config
from app.video_source import VideoSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview and validate the configured ROI")
    parser.add_argument("--config", default="config/test_game.yaml", help="Path to config yaml")
    parser.add_argument(
        "--output",
        default="assets/roi_preview.png",
        help="Optional screenshot path when you press s",
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
                print("Frame read failed or stream ended.")
                break

            frame = cv2.resize(frame, (cfg.output_width, cfg.output_height))
            h, w = frame.shape[:2]
            x1 = max(0, min(cfg.roi.x1, w - 1))
            x2 = w if cfg.roi.x2 < 0 else max(x1 + 1, min(cfg.roi.x2, w))
            y1 = max(0, min(cfg.roi.y1, h - 1))
            y2 = max(y1 + 1, min(cfg.roi.y2, h))

            preview = frame.copy()
            roi = frame[y1:y2, x1:x2]
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                preview,
                f"ROI x={x1}:{x2} y={y1}:{y2} | s=save q=quit",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imshow("roi-preview", preview)
            cv2.imshow("roi-crop", roi)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                ok = cv2.imwrite(str(output_path), preview)
                if ok:
                    print(f"Saved ROI preview to {output_path}")
            if key in (27, ord("q")):
                break
    finally:
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
