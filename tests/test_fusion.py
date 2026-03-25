import pytest

from app.fusion import SignalFusion


def test_fast_triggers_on_visual_alone() -> None:
    f = SignalFusion("fast")
    assert f.should_trigger(0.8, 0.7, 0.0, False, audio_available=True)


def test_fast_triggers_on_horn_alone() -> None:
    f = SignalFusion("fast")
    assert f.should_trigger(0.0, 0.7, 0.6, False, audio_available=True)


def test_fast_triggers_on_keyword_alone() -> None:
    f = SignalFusion("fast")
    assert f.should_trigger(0.0, 0.7, 0.0, True, audio_available=True)


def test_balanced_triggers_on_visual_alone() -> None:
    f = SignalFusion("balanced")
    assert f.should_trigger(0.8, 0.7, 0.0, False, audio_available=True)


def test_balanced_triggers_on_horn_plus_keyword() -> None:
    f = SignalFusion("balanced")
    assert f.should_trigger(0.0, 0.7, 0.6, True, audio_available=True)


def test_balanced_does_not_trigger_on_horn_alone() -> None:
    f = SignalFusion("balanced")
    assert not f.should_trigger(0.0, 0.7, 0.6, False, audio_available=True)


def test_balanced_allows_horn_without_keyword_detector() -> None:
    f = SignalFusion("balanced")
    assert f.should_trigger(0.0, 0.7, 0.6, False, audio_available=True, keyword_available=False)


def test_require_horn_and_keyword_overrides_other_signals() -> None:
    f = SignalFusion("fast")
    assert not f.should_trigger(
        0.9, 0.7, 0.6, False, audio_available=True, require_horn_and_keyword=True
    )
    assert f.should_trigger(
        0.1, 0.7, 0.6, True, audio_available=True, require_horn_and_keyword=True
    )


def test_accurate_requires_two_signals() -> None:
    f = SignalFusion("accurate")
    assert not f.should_trigger(0.8, 0.7, 0.0, False, audio_available=True)
    assert f.should_trigger(0.8, 0.7, 0.6, False, audio_available=True)
    assert f.should_trigger(0.8, 0.7, 0.0, True, audio_available=True)
    assert f.should_trigger(0.0, 0.7, 0.6, True, audio_available=True)


def test_no_audio_falls_back_to_visual() -> None:
    """Without audio the detector should behave like the old visual-only mode."""
    for mode in ("fast", "balanced", "accurate"):
        f = SignalFusion(mode)
        assert f.should_trigger(0.8, 0.7, 0.0, False, audio_available=False)
        assert not f.should_trigger(0.5, 0.7, 0.0, False, audio_available=False)


def test_visual_tolerance_handles_rounding_edge() -> None:
    f = SignalFusion("balanced")
    # Value might display as 0.700 in UI while being slightly below threshold.
    assert f.should_trigger(0.6996, 0.7, 0.0, False, audio_available=True)


def test_invalid_sensitivity_raises() -> None:
    with pytest.raises(ValueError):
        SignalFusion("turbo")
