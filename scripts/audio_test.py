from __future__ import annotations

import argparse
import time

from app.audio_stream import AudioStream
from app.config import load_config
from app.horn_detector import HornDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the audio pipeline and print horn confidence")
    parser.add_argument("--config", default="config/test_game.yaml", help="Path to config yaml")
    parser.add_argument(
        "--source",
        help="Override audio source. Defaults to audio.source or video_source from config.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between confidence reports",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    audio_source = args.source or cfg.audio.source or cfg.video_source

    horn = HornDetector(
        sample_rate=cfg.audio.sample_rate,
        freq_low=cfg.audio.horn_freq_low,
        freq_high=cfg.audio.horn_freq_high,
        energy_threshold=cfg.audio.horn_energy_threshold,
        sustain_seconds=cfg.audio.horn_sustain_seconds,
        chunk_duration=cfg.audio.chunk_duration,
        max_history=None,
    )

    audio = AudioStream(
        source=audio_source,
        sample_rate=cfg.audio.sample_rate,
        chunk_duration=cfg.audio.chunk_duration,
        input_format=cfg.audio.input_format or None,
    )
    audio.register(horn)

    print(f"Testing audio source: {audio_source}")
    audio.start()

    try:
        while audio._thread is not None and audio._thread.is_alive():
            ts = horn.last_timestamp
            conf = horn.confidence_at(ts) if ts > 0 else 0.0
            print(
                f"audio_active={audio.active} had_output={audio.had_output} "
                f"timestamp={ts:.2f} horn_conf={conf:.2f}"
            )
            time.sleep(args.interval)
    finally:
        audio.stop()

    print(
        f"Finished. had_output={audio.had_output} "
        f"last_timestamp={horn.last_timestamp:.2f} max_conf={horn.last_confidence:.2f}"
    )


if __name__ == "__main__":
    main()
