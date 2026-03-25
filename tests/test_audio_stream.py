import time

from app.audio_stream import AudioStream


def test_detect_file_source_for_mp4_path() -> None:
    assert AudioStream._detect_file_source("assets/Game1.mp4", None)


def test_detect_file_source_rejects_capture_inputs() -> None:
    assert not AudioStream._detect_file_source("0", None)
    assert not AudioStream._detect_file_source("assets/Game1.mp4", "dshow")


def test_chunk_timestamp_uses_video_callback() -> None:
    stream = AudioStream("assets/Game1.mp4", video_position_callback=lambda: 12.5)

    assert stream._chunk_timestamp(1.0) == 12.5


def test_chunk_timestamp_falls_back_when_callback_missing() -> None:
    stream = AudioStream("assets/Game1.mp4")

    assert stream._chunk_timestamp(1.5) == 1.5


def test_seconds_since_output_none_before_audio() -> None:
    stream = AudioStream("assets/Game1.mp4")

    assert stream.seconds_since_output() is None


def test_is_stalled_after_timeout() -> None:
    stream = AudioStream("assets/Game1.mp4")
    stream.active = True
    stream.had_output = True
    stream.last_output_at = time.time() - 6.0

    assert stream.is_stalled(5.0)
