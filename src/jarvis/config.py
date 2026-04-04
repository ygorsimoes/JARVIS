from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_FALLBACK_MODEL = "qwen3:4b"
DEFAULT_KOKORO_VOICE = "af_heart"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo-q4"


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
    kokoro_voice: str = DEFAULT_KOKORO_VOICE
    kokoro_model_path: Path | None = None
    kokoro_voices_path: Path | None = None


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
        kokoro_voice=os.getenv("JARVIS_KOKORO_VOICE", DEFAULT_KOKORO_VOICE),
        kokoro_model_path=_optional_path_env("JARVIS_KOKORO_MODEL_PATH"),
        kokoro_voices_path=_optional_path_env("JARVIS_KOKORO_VOICES_PATH"),
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
