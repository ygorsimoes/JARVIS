from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TurnPreset:
    name: str
    user_speech_timeout: float
    settle_secs: float
    trailing_secs: float
    incomplete_secs: float


TURN_PRESETS: dict[str, TurnPreset] = {
    "balanced": TurnPreset(
        name="balanced",
        user_speech_timeout=1.0,
        settle_secs=0.4,
        trailing_secs=1.4,
        incomplete_secs=3.2,
    ),
    "patient": TurnPreset(
        name="patient",
        user_speech_timeout=1.3,
        settle_secs=0.55,
        trailing_secs=1.8,
        incomplete_secs=4.2,
    ),
    "very-patient": TurnPreset(
        name="very-patient",
        user_speech_timeout=1.8,
        settle_secs=0.8,
        trailing_secs=2.3,
        incomplete_secs=5.8,
    ),
}


def get_turn_preset(name: str) -> TurnPreset:
    normalized = name.strip().lower()
    if normalized not in TURN_PRESETS:
        available = ", ".join(sorted(TURN_PRESETS))
        raise ValueError(f"Preset de turno invalido: {name}. Use um de: {available}")
    return TURN_PRESETS[normalized]
