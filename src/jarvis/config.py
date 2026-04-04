from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_FALLBACK_MODEL = "qwen3:4b"
DEFAULT_KOKORO_VOICE = "pm_alex"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo-q4"
DEFAULT_OLLAMA_KEEP_ALIVE = "30m"


@dataclass(slots=True, frozen=True)
class AppConfig:
    log_level: str = DEFAULT_LOG_LEVEL
    language: str = "pt"
    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 24000
    input_device_index: int | None = None
    output_device_index: int | None = None
    whisper_model: str = DEFAULT_WHISPER_MODEL
    whisper_temperature: float = 0.0
    whisper_no_speech_prob: float = 0.6
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_fallback_model: str = DEFAULT_OLLAMA_FALLBACK_MODEL
    ollama_temperature: float = 0.35
    ollama_max_tokens: int = 96
    ollama_keep_alive: str = DEFAULT_OLLAMA_KEEP_ALIVE
    kokoro_voice: str = DEFAULT_KOKORO_VOICE
    kokoro_model_path: Path | None = None
    kokoro_voices_path: Path | None = None
    prewarm_enabled: bool = True
    ollama_prewarm_enabled: bool = False
    echo_suppression_enabled: bool = True
    echo_suppression_release_ms: int = 350


def load_config(env_file: str | None = None) -> AppConfig:
    if env_file:
        load_dotenv(env_file, override=False)
    else:
        load_dotenv(override=False)

    return AppConfig(
        log_level=os.getenv("JARVIS_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
        language="pt",
        audio_in_sample_rate=_int_env("JARVIS_AUDIO_IN_SAMPLE_RATE", 16000),
        audio_out_sample_rate=_int_env("JARVIS_AUDIO_OUT_SAMPLE_RATE", 24000),
        input_device_index=_optional_int_env("JARVIS_INPUT_DEVICE_INDEX"),
        output_device_index=_optional_int_env("JARVIS_OUTPUT_DEVICE_INDEX"),
        whisper_model=os.getenv(
            "JARVIS_WHISPER_MODEL",
            DEFAULT_WHISPER_MODEL,
        ),
        whisper_temperature=_float_env("JARVIS_WHISPER_TEMPERATURE", 0.0),
        whisper_no_speech_prob=_float_env("JARVIS_WHISPER_NO_SPEECH_PROB", 0.6),
        ollama_base_url=os.getenv("JARVIS_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        ollama_model=os.getenv("JARVIS_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        ollama_fallback_model=os.getenv(
            "JARVIS_OLLAMA_FALLBACK_MODEL",
            DEFAULT_OLLAMA_FALLBACK_MODEL,
        ),
        ollama_temperature=_float_env("JARVIS_OLLAMA_TEMPERATURE", 0.35),
        ollama_max_tokens=_int_env("JARVIS_OLLAMA_MAX_TOKENS", 96),
        ollama_keep_alive=os.getenv("JARVIS_OLLAMA_KEEP_ALIVE", DEFAULT_OLLAMA_KEEP_ALIVE),
        kokoro_voice=os.getenv("JARVIS_KOKORO_VOICE", DEFAULT_KOKORO_VOICE),
        kokoro_model_path=_optional_path_env("JARVIS_KOKORO_MODEL_PATH"),
        kokoro_voices_path=_optional_path_env("JARVIS_KOKORO_VOICES_PATH"),
        prewarm_enabled=_bool_env("JARVIS_PREWARM_ENABLED", True),
        ollama_prewarm_enabled=_bool_env("JARVIS_OLLAMA_PREWARM_ENABLED", False),
        echo_suppression_enabled=_bool_env("JARVIS_ECHO_SUPPRESSION_ENABLED", True),
        echo_suppression_release_ms=_int_env("JARVIS_ECHO_SUPPRESSION_RELEASE_MS", 350),
    )


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def _optional_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    return int(raw) if raw else None


def _optional_path_env(name: str) -> Path | None:
    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else None


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    return default
