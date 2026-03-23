from app.gate import CooldownGate


def test_cooldown_gate_allows_first_trigger() -> None:
    gate = CooldownGate(10.0)
    assert gate.can_trigger(now=100.0)


def test_cooldown_gate_blocks_early_trigger() -> None:
    gate = CooldownGate(10.0)
    gate.mark_triggered(now=100.0)
    assert not gate.can_trigger(now=105.0)


def test_cooldown_gate_allows_after_window() -> None:
    gate = CooldownGate(10.0)
    gate.mark_triggered(now=100.0)
    assert gate.can_trigger(now=110.0)
