import numpy as np

from app.horn_detector import HornDetector


def _sine_wave(freq: float, duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate a pure sine wave at the given frequency."""
    t = np.arange(int(sample_rate * duration)) / sample_rate
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_horn_detected_for_in_band_tone() -> None:
    """A sustained tone inside the horn band should yield high confidence."""
    det = HornDetector(
        sample_rate=16000,
        freq_low=200,
        freq_high=1000,
        energy_threshold=0.3,
        sustain_seconds=1.0,
        chunk_duration=0.5,
    )

    tone = _sine_wave(500, duration=0.5, sample_rate=16000)
    for i in range(4):
        det.analyze(tone, timestamp=i * 0.5)

    assert det.confidence_at(1.0) > 0.5


def test_horn_not_detected_for_out_of_band_tone() -> None:
    """A tone outside the horn band should stay below threshold."""
    det = HornDetector(
        sample_rate=16000,
        freq_low=200,
        freq_high=1000,
        energy_threshold=0.3,
        sustain_seconds=1.0,
        chunk_duration=0.5,
    )

    tone = _sine_wave(5000, duration=0.5, sample_rate=16000)
    for i in range(4):
        det.analyze(tone, timestamp=i * 0.5)

    assert det.confidence_at(1.0) < 0.3


def test_horn_silence_yields_zero() -> None:
    det = HornDetector()
    assert det.confidence_at(0.0) == 0.0
    assert det.last_confidence == 0.0


def test_last_confidence_reflects_recent_chunks() -> None:
    det = HornDetector(
        sample_rate=16000,
        freq_low=200,
        freq_high=1000,
        energy_threshold=0.3,
        sustain_seconds=0.5,
        chunk_duration=0.5,
    )

    tone = _sine_wave(400, duration=0.5, sample_rate=16000)
    det.analyze(tone, 0.0)
    det.analyze(tone, 0.5)
    det.analyze(tone, 1.0)

    assert det.last_confidence > 0.5


def test_unbounded_history_keeps_older_file_events() -> None:
    det = HornDetector(
        sample_rate=16000,
        freq_low=200,
        freq_high=1000,
        energy_threshold=0.3,
        sustain_seconds=0.5,
        chunk_duration=0.5,
        max_history=None,
    )

    tone = _sine_wave(400, duration=0.5, sample_rate=16000)
    for i in range(200):
        det.analyze(tone, i * 0.5)

    assert det.confidence_at(1.0) > 0.5


def test_reference_profile_detects_matching_tone() -> None:
    reference = _sine_wave(500, duration=0.5, sample_rate=16000)
    reference_profile = HornDetector._normalized_spectrum(np.abs(np.fft.rfft(reference)))
    det = HornDetector(
        sample_rate=16000,
        reference_profile=reference_profile,
        reference_similarity_threshold=0.95,
        sustain_seconds=0.5,
        chunk_duration=0.5,
    )

    det.analyze(reference, 0.0)

    assert det.confidence_at(0.0) > 0.5


def test_reference_profile_rejects_different_tone() -> None:
    reference = _sine_wave(500, duration=0.5, sample_rate=16000)
    different = _sine_wave(900, duration=0.5, sample_rate=16000)
    reference_profile = HornDetector._normalized_spectrum(np.abs(np.fft.rfft(reference)))
    det = HornDetector(
        sample_rate=16000,
        reference_profile=reference_profile,
        reference_similarity_threshold=0.95,
        sustain_seconds=0.5,
        chunk_duration=0.5,
    )

    det.analyze(different, 0.0)

    assert det.confidence_at(0.0) == 0.0
