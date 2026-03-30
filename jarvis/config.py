from __future__ import annotations

from typing import List, Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .prompts import BASE_SYSTEM_PROMPT


DEFAULT_SYSTEM_ALLOWED_APPS = [
    "Safari",
    "Spotify",
    "Notes",
    "Calendar",
    "Music",
    "Mail",
    "Terminal",
]


class JarvisConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="JARVIS_",
        extra="ignore",
    )

    app_name: str = "J.A.R.V.I.S."
    metal_memory_limit_gb: float = 9.5
    metal_wired_limit_gb: float = 8.5
    metal_cache_limit_gb: float = 0.5
    llm_max_kv_size: int = 4096

    stt_backend: str = "speech_analyzer"
    stt_locale: str = "pt-BR"
    stt_bridge_bin: str = "./swift/SpeechAnalyzerCLI/.build/release/speechanalyzer-cli"

    llm_hot_path: str = "foundation_models"
    llm_hot_path_url: str = "http://127.0.0.1:8008"
    llm_hot_path_bridge_bin: str = (
        "./swift/FoundationModelsBridge/.build/release/foundation-models-bridge"
    )
    llm_deliberative: str = "mlx_lm"
    llm_deliberative_model: str = "mlx-community/Qwen3-8B-4bit"
    llm_deliberative_temperature: float = 0.2
    llm_deliberative_top_p: float = 0.9
    llm_deliberative_repetition_penalty: float = 1.0
    llm_response_max_tokens: int = 512

    tts_backend: str = "mlx_audio_kokoro"
    tts_model: str = "mlx-community/Kokoro-82M-bf16"
    tts_voice: str = "pm_santa"
    tts_lang_code: str = "p"
    tts_avspeech_voice: Optional[str] = "Luciana"
    tts_avspeech_rate: int = 175
    tts_sample_rate_hz: int = 24000
    playback_backend: str = "sounddevice"

    activation_backend: str = Field(
        default="push_to_talk",
        validation_alias=AliasChoices("ACTIVATION_BACKEND", "WAKE_WORD_BACKEND"),
    )
    activation_hotkey: str = "<ctrl>+<alt>+space"
    activation_terminal_fallback: bool = True
    allowed_file_roots: List[str] = Field(default_factory=list)
    system_allowed_apps: List[str] = Field(
        default_factory=lambda: list(DEFAULT_SYSTEM_ALLOWED_APPS)
    )
    working_memory_turns: int = 12
    event_bus_queue_size: int = 128

    sentence_min_tokens: int = 8
    sentence_min_soft_boundary_chars: int = 40
    sentence_max_pending_segments: int = 2
    sentence_backpressure_poll_ms: int = 10

    turn_silence_timeout_ms: int = 800
    turn_partial_commit_min_chars: int = 16
    turn_partial_stability_ms: int = 250
    turn_tick_interval_ms: int = 100
    turn_max_duration_s: float = 30.0

    log_level: str = "INFO"
    system_prompt_override: Optional[str] = None

    @field_validator("allowed_file_roots", "system_allowed_apps", mode="before")
    @classmethod
    def parse_string_list(cls, value: object) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        raise TypeError("field must be a list or comma-separated string")

    @property
    def metal_memory_limit_bytes(self) -> int:
        return int(self.metal_memory_limit_gb * 1024**3)

    @property
    def metal_wired_limit_bytes(self) -> int:
        return int(self.metal_wired_limit_gb * 1024**3)

    @property
    def metal_cache_limit_bytes(self) -> int:
        return int(self.metal_cache_limit_gb * 1024**3)

    @property
    def system_prompt(self) -> str:
        return self.system_prompt_override or BASE_SYSTEM_PROMPT


def load_config() -> JarvisConfig:
    return JarvisConfig()
