from __future__ import annotations

import json
import re
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


def resolve_ollama_model(base_url: str, preferred: str, fallback: str) -> str:
    available_models = _fetch_ollama_tags(base_url)

    resolved_preferred = _match_model_name(preferred, available_models)
    if resolved_preferred:
        return resolved_preferred

    resolved_fallback = _match_model_name(fallback, available_models)
    if resolved_fallback:
        return resolved_fallback

    available = ", ".join(sorted(available_models)) if available_models else "nenhum"
    raise RuntimeError(
        "Nenhum modelo Ollama compativel foi encontrado. "
        f"Esperado: {preferred} ou {fallback}. Disponiveis: {available}. "
        f"Rode `ollama pull {preferred}`."
    )


def build_ollama_extra(model_name: str, *, keep_alive: str | None = None) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    extra_body: dict[str, Any] = {}
    if uses_reasoning_effort_none(model_name):
        extra_body["reasoning"] = {"effort": "none"}
    if keep_alive:
        extra_body["keep_alive"] = keep_alive
    if extra_body:
        extra["extra_body"] = extra_body
    return extra


def uses_reasoning_effort_none(model_name: str) -> bool:
    return is_qwen35_model(model_name)


def is_qwen35_model(model_name: str) -> bool:
    return model_name.startswith("qwen3.5:")


def select_smaller_qwen_model(primary: str, fallback: str) -> str:
    primary_size = _extract_model_size(primary)
    fallback_size = _extract_model_size(fallback)

    if primary_size is None:
        return fallback
    if fallback_size is None:
        return primary
    return primary if primary_size <= fallback_size else fallback


def _fetch_ollama_tags(base_url: str) -> set[str]:
    tags_url = f"{base_url.removesuffix('/v1')}/api/tags"
    try:
        with urlopen(tags_url, timeout=3) as response:
            payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError(
            "Nao foi possivel acessar o Ollama em "
            f"{tags_url}. Verifique se `ollama serve` esta ativo."
        ) from exc

    return {item.get("name", "") for item in payload.get("models", []) if item.get("name")}


def _match_model_name(name: str, available_models: set[str]) -> str | None:
    if name in available_models:
        return name

    candidates = [candidate for candidate in available_models if candidate.startswith(f"{name}:")]
    if len(candidates) == 1:
        return candidates[0]

    return None


def _extract_model_size(model_name: str) -> float | None:
    match = re.search(r":(\d+(?:\.\d+)?)b", model_name)
    if not match:
        return None
    return float(match.group(1))
