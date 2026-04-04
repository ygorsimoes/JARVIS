from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TurnPreset:
    name: str
    user_speech_timeout: float
    resume_delay_secs: float
    settle_secs: float
    trailing_secs: float
    incomplete_secs: float


TURN_PRESETS: dict[str, TurnPreset] = {
    "balanced": TurnPreset(
        name="balanced",
        user_speech_timeout=1.0,
        resume_delay_secs=1.5,
        settle_secs=0.25,
        trailing_secs=4.0,
        incomplete_secs=8.0,
    ),
    "patient": TurnPreset(
        name="patient",
        user_speech_timeout=1.3,
        resume_delay_secs=2.5,
        settle_secs=0.35,
        trailing_secs=6.0,
        incomplete_secs=12.0,
    ),
    "very-patient": TurnPreset(
        name="very-patient",
        user_speech_timeout=1.8,
        resume_delay_secs=3.5,
        settle_secs=0.5,
        trailing_secs=8.0,
        incomplete_secs=16.0,
    ),
}


def get_turn_preset(name: str) -> TurnPreset:
    normalized = name.strip().lower()
    if normalized not in TURN_PRESETS:
        available = ", ".join(sorted(TURN_PRESETS))
        raise ValueError(f"Preset de turno invalido: {name}. Use um de: {available}")
    return TURN_PRESETS[normalized]
