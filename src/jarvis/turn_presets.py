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
        user_speech_timeout=0.7,
        settle_secs=0.25,
        trailing_secs=0.8,
        incomplete_secs=2.0,
    ),
    "patient": TurnPreset(
        name="patient",
        user_speech_timeout=1.0,
        settle_secs=0.35,
        trailing_secs=1.2,
        incomplete_secs=3.0,
    ),
    "very-patient": TurnPreset(
        name="very-patient",
        user_speech_timeout=1.4,
        settle_secs=0.5,
        trailing_secs=1.8,
        incomplete_secs=4.5,
    ),
}


def get_turn_preset(name: str) -> TurnPreset:
    normalized = name.strip().lower()
    if normalized not in TURN_PRESETS:
        available = ", ".join(sorted(TURN_PRESETS))
        raise ValueError(f"Preset de turno invalido: {name}. Use um de: {available}")
    return TURN_PRESETS[normalized]
