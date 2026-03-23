from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from app.config import AppConfig, load_config
from app.detector import GoalTemplateDetector
from app.fusion import SignalFusion
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


def _setup_audio(cfg: AppConfig):
    """Initialise audio stream, horn detector, and keyword detector.

    Returns (audio_stream, horn_detector, keyword_detector).
    Any component that cannot be loaded is returned as None.
    """
    if not cfg.audio.enabled:
        return None, None, None

    from app.audio_stream import AudioStream
    from app.horn_detector import HornDetector

    horn = HornDetector(
        sample_rate=cfg.audio.sample_rate,
        freq_low=cfg.audio.horn_freq_low,
        freq_high=cfg.audio.horn_freq_high,
        energy_threshold=cfg.audio.horn_energy_threshold,
        sustain_seconds=cfg.audio.horn_sustain_seconds,
        chunk_duration=cfg.audio.chunk_duration,
    )

    audio = AudioStream(
        source=cfg.video_source,
        sample_rate=cfg.audio.sample_rate,
        chunk_duration=cfg.audio.chunk_duration,
    )
    audio.register(horn)

    keyword = None
    if cfg.audio.vosk_model_path:
        from app.keyword_detector import KeywordDetector

        kw = KeywordDetector(
            model_path=cfg.audio.vosk_model_path,
            sample_rate=cfg.audio.sample_rate,
        )
        if kw.available:
            audio.register(kw)
            keyword = kw

    audio.start()
    print("Audio detection started.")
    return audio, horn, keyword


def run(cfg: AppConfig) -> None:
    source = VideoSource(cfg.video_source)
    if not source.is_open():
        raise RuntimeError(f"Unable to open video source: {cfg.video_source}")

    detector = None
    template_path = Path(cfg.template_path)
    if template_path.exists():
        detector = GoalTemplateDetector(str(template_path), cfg.team_templates)
    else:
        print(f"Template missing: {template_path}. Visual detection disabled.")

    audio, horn, keyword = _setup_audio(cfg)
    audio_on = audio is not None

    fusion = SignalFusion(cfg.sensitivity)
    gate = CooldownGate(cfg.cooldown_seconds)
    relay = _build_relay(cfg)

    print(f"Sensitivity: {cfg.sensitivity} | Audio: {'on' if audio_on else 'off'}")
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

            video_ts = source.timestamp()

            visual_score = detector.score(roi) if detector else 0.0
            horn_conf = horn.confidence_at(video_ts) if horn else 0.0
            kw_hit = keyword.detected_at(video_ts) if keyword else False

            team_name = "unknown"
            if fusion.should_trigger(visual_score, cfg.threshold, horn_conf, kw_hit, audio_on):
                if gate.can_trigger():
                    if detector is not None:
                        team_name = detector.detect_team(roi)
                    parts = [f"team={team_name}", f"visual={visual_score:.3f}"]
                    if audio_on:
                        parts += [f"horn={horn_conf:.2f}", f"kw={kw_hit}"]
                    print(f"[GOAL] {' '.join(parts)} | {time.strftime('%H:%M:%S')}")
                    relay.trigger()
                    gate.mark_triggered()

            if cfg.show_window:
                display = frame.copy()
                if cfg.show_roi_box:
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                info = f"v={visual_score:.3f} thr={cfg.threshold:.2f}"
                if audio_on:
                    info += f" horn={horn_conf:.2f} kw={'Y' if kw_hit else 'N'}"
                info += f" [{cfg.sensitivity}]"
                cv2.putText(
                    display, info,
                    (10, cfg.output_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )
                cv2.imshow("feed", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        if audio:
            audio.stop()
        source.release()
        if hasattr(relay, "close"):
            relay.close()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NHL goal overlay detector MVP")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config yaml")
    parser.add_argument(
        "--video-source",
        help="Camera index such as 0 or a path to a video file such as assets/Game1.mp4",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Disable the OpenCV preview window and only print detections",
    )
    parser.add_argument(
        "--sensitivity",
        choices=["fast", "balanced", "accurate"],
        help="Detection sensitivity (fast = quick reaction, accurate = fewer false positives)",
    )
    parser.add_argument(
        "--audio", dest="audio", action="store_true", default=None,
        help="Enable audio detection",
    )
    parser.add_argument(
        "--no-audio", dest="audio", action="store_false",
        help="Disable audio detection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.video_source is not None:
        cfg.video_source = str(args.video_source)

    if args.no_window:
        cfg.show_window = False

    if args.sensitivity is not None:
        cfg.sensitivity = args.sensitivity

    if args.audio is not None:
        cfg.audio.enabled = args.audio

    run(cfg)


if __name__ == "__main__":
    main()

    run(cfg)


if __name__ == "__main__":
    main()
