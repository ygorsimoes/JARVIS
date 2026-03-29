from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Pattern

from ..models.conversation import RouteDecision, RouteTarget


@dataclass(frozen=True)
class _DirectIntent:
    tool_name: str
    pattern: Pattern[str]
    reason: str


class ComplexityRouter:
    def __init__(self) -> None:
        self._direct_intents = (
            _DirectIntent(
                tool_name="system.get_time",
                pattern=re.compile(r"\b(que horas s[aã]o|que horas e|hor[aá]rio agora)\b", re.IGNORECASE),
                reason="consulta objetiva de horario",
            ),
            _DirectIntent(
                tool_name="timer.start",
                pattern=re.compile(r"\b(timer|cron[oô]metro|alarme)\b", re.IGNORECASE),
                reason="intencao direta de timer",
            ),
            _DirectIntent(
                tool_name="browser.search",
                pattern=re.compile(r"\b(pesquise|procure|busque na web|pesquisa na web)\b", re.IGNORECASE),
                reason="busca direta na web",
            ),
            _DirectIntent(
                tool_name="system.open_app",
                pattern=re.compile(r"\b(abre|abrir)\b", re.IGNORECASE),
                reason="acao direta de sistema",
            ),
        )
        self._reasoning_markers = (
            "analisa",
            "analise",
            "explica",
            "explique",
            "por que",
            "compare",
            "plano",
            "estrategia",
            "estrategia",
            "debug",
            "erro",
            "arquitetura",
            "planeje",
            "resuma",
            "resume",
        )
        self._multi_step_markers = (
            "depois",
            "em seguida",
            "passo a passo",
            "liste",
            "organize",
        )
        self._subordinate_markers = (
            " porque ",
            " quando ",
            " embora ",
            " se ",
            " enquanto ",
        )

    def route(
        self,
        text: str,
        recalled_memories: int = 0,
        tool_chain_depth: int = 0,
        recent_turns: int = 0,
    ) -> RouteDecision:
        normalized = " ".join(text.strip().lower().split())
        for intent in self._direct_intents:
            if intent.pattern.search(normalized):
                return RouteDecision(
                    target=RouteTarget.DIRECT_TOOL,
                    tool_name=intent.tool_name,
                    reason=intent.reason,
                    confidence=0.95,
                )

        token_count = self._token_count(normalized)
        subordinate_hits = sum(marker in normalized for marker in self._subordinate_markers)
        reasoning_hits = sum(marker in normalized for marker in self._reasoning_markers)
        multi_step_hits = sum(marker in normalized for marker in self._multi_step_markers)

        if recalled_memories > 0:
            return RouteDecision(
                target=RouteTarget.DELIBERATIVE,
                reason="memoria recuperada exige contexto adicional",
                confidence=0.85,
            )

        if tool_chain_depth > 0 or reasoning_hits > 0 or multi_step_hits > 0:
            return RouteDecision(
                target=RouteTarget.DELIBERATIVE,
                reason="pedido com sinal de raciocinio ou multiplas etapas",
                confidence=0.9,
            )

        if subordinate_hits > 1 or token_count > 12 or recent_turns > 8:
            return RouteDecision(
                target=RouteTarget.DELIBERATIVE,
                reason="pedido longo ou contextual demais para hot path",
                confidence=0.8,
            )

        return RouteDecision(
            target=RouteTarget.HOT_PATH,
            reason="pedido curto e simples para resposta rapida",
            confidence=0.8,
        )

    @staticmethod
    def _token_count(text: str) -> int:
        return len([token for token in text.split(" ") if token])
