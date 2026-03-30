from __future__ import annotations

import shlex
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urlparse


SHELL_CONTROL_TOKENS = ("&&", "||", "|", ";", ">", "<", "`", "$(", "\n", "\r")


def normalize_roots(paths: Iterable[str]) -> List[Path]:
    return [Path(path).expanduser().resolve() for path in paths if str(path).strip()]


def ensure_path_within_roots(target: str, roots: Iterable[Path]) -> Path:
    candidate = Path(target).expanduser().resolve()
    for root in roots:
        try:
            candidate.relative_to(root)
            return candidate
        except ValueError:
            continue
    raise PermissionError("path %s is outside the allowed roots" % candidate)


def parse_safe_shell_command(command: str) -> List[str]:
    normalized = command.strip()
    if not normalized:
        raise ValueError("Comando vazio")
    if any(token in normalized for token in SHELL_CONTROL_TOKENS):
        raise ValueError("Operadores de shell nao sao permitidos")
    try:
        return shlex.split(normalized)
    except ValueError as exc:
        raise ValueError("Comando invalido: %s" % exc) from exc


def validate_http_url(url: str) -> str:
    normalized = url.strip()
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Somente URLs HTTP(s) sao permitidas")
    if not parsed.netloc:
        raise ValueError("URL invalida: host ausente")
    return normalized


def build_trust_metadata(
    *, source: str, trusted: bool, detail: str | None = None
) -> dict[str, object]:
    metadata: dict[str, object] = {"source": source, "trusted": trusted}
    if detail:
        metadata["detail"] = detail
    return metadata
