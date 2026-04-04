from __future__ import annotations

import json
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import mlx_whisper
import numpy as np
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from loguru import logger

from .config import AppConfig


def prepare_stt_config(config: AppConfig) -> AppConfig:
    resolved_whisper_model = resolve_whisper_model_path(config.whisper_model)
    if resolved_whisper_model == config.whisper_model:
        return config

    logger.info("[stt] usando snapshot local do whisper")
    return replace(config, whisper_model=resolved_whisper_model)


def prewarm_whisper_model(config: AppConfig) -> None:
    started_at = time.perf_counter()
    silence = np.zeros(config.audio_in_sample_rate, dtype=np.float32)
    mlx_whisper.transcribe(
        silence,
        path_or_hf_repo=config.whisper_model,
        temperature=config.whisper_temperature,
        language=config.language,
    )
    logger.info("[warmup] whisper pronto | {:.1f}ms", _to_ms(time.perf_counter() - started_at))


def prewarm_ollama_model(config: AppConfig) -> None:
    payload = {
        "model": config.ollama_model,
        "messages": [{"role": "user", "content": "Responda apenas com ok."}],
        "stream": False,
        "keep_alive": config.ollama_keep_alive,
    }

    request = Request(
        f"{config.ollama_base_url.removesuffix('/v1')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    started_at = time.perf_counter()
    try:
        with urlopen(request, timeout=180) as response:
            json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError("Falha ao aquecer o modelo Ollama local.") from exc

    logger.info("[warmup] ollama pronto | {:.1f}ms", _to_ms(time.perf_counter() - started_at))


@lru_cache(maxsize=4)
def resolve_whisper_model_path(model_ref: str) -> str:
    model_path = Path(model_ref).expanduser()
    if model_path.exists():
        return str(model_path)

    try:
        return snapshot_download(model_ref, local_files_only=True)
    except LocalEntryNotFoundError:
        logger.info("[stt] baixando assets do whisper pela primeira vez")
        return snapshot_download(model_ref)


def _to_ms(seconds: float) -> float:
    return seconds * 1000.0
