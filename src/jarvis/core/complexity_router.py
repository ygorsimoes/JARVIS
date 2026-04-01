from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern

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
                pattern=re.compile(
                    r"\b(que horas s[aã]o|que horas e|hor[aá]rio agora)\b",
                    re.IGNORECASE,
                ),
                reason="consulta objetiva de horario",
            ),
            _DirectIntent(
                tool_name="system.set_volume",
                pattern=re.compile(
                    r"(?:\b(?:volume|som)\b[^\n]*\b\d{1,3}\b)|(?:\b\d{1,3}\b[^\n]*\b(?:volume|som)\b)",
                    re.IGNORECASE,
                ),
                reason="ajuste direto de volume",
            ),
            _DirectIntent(
                tool_name="timer.cancel",
                pattern=re.compile(
                    r"\b(cancela|cancelar|pare|parar|remove|remover)\b[^\n]*\b(timer|cron[oô]metro|alarme)\b",
                    re.IGNORECASE,
                ),
                reason="cancelamento direto de timer",
            ),
            _DirectIntent(
                tool_name="timer.list",
                pattern=re.compile(
                    r"(?:\b(lista|listar|mostra|mostrar|quais|tenho)\b[^\n]*\b(timers?|cron[oô]metros?|alarmes?)\b)|(?:\b(timers?|cron[oô]metros?|alarmes?)\b[^\n]*\bativos?\b)",
                    re.IGNORECASE,
                ),
                reason="consulta direta de timers ativos",
            ),
            _DirectIntent(
                tool_name="timer.start",
                pattern=re.compile(
                    r"(?:\b(defina|define|crie|criar|inicie|iniciar|coloque|marque|programe)\b[^\n]*\b(timer|cron[oô]metro|alarme)\b)|(?:\b(timer|cron[oô]metro|alarme)\b[^\n]*\b\d+\b)",
                    re.IGNORECASE,
                ),
                reason="intencao direta de timer",
            ),
            _DirectIntent(
                tool_name="browser.search",
                pattern=re.compile(
                    r"\b(pesquise|procure|busque na web|pesquisa na web)\b",
                    re.IGNORECASE,
                ),
                reason="busca direta na web",
            ),
            _DirectIntent(
                tool_name="browser.fetch_url",
                pattern=re.compile(r"https?://\S+", re.IGNORECASE),
                reason="leitura direta de url",
            ),
            _DirectIntent(
                tool_name="calendar.list_events",
                pattern=re.compile(
                    r"(?:\b(quais|mostra|mostrar|lista|listar|tenho)\b[^\n]*\b(eventos?|agenda|calend[aá]rio)\b)|(?:\b(eventos?|agenda|calend[aá]rio)\b[^\n]*\b(hoje|amanh[aã]|semana|dias?)\b)",
                    re.IGNORECASE,
                ),
                reason="consulta direta de calendario",
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
            "debug",
            "erro",
            "arquitetura",
            "planeje",
            "resuma",
            "resume",
            "implemente",
            "corrija",
            "refatore",
            "teste",
            "codigo",
            "código",
            "função",
            "funcao",
            "arquivo",
        )
        self._multi_step_markers = (
            "depois",
            "em seguida",
            "passo a passo",
            "liste",
            "organize",
            "primeiro",
            "segundo",
            "entao",
            "então",
        )
        self._subordinate_markers = (
            " porque ",
            " quando ",
            " embora ",
            " se ",
            " enquanto ",
            " para que ",
        )

    def route(
        self,
        text: str,
        recalled_memories: int = 0,
        tool_chain_depth: int = 0,
        recent_turns: int = 0,
    ) -> RouteDecision:
        normalized = " ".join(text.strip().lower().split())
        if not normalized:
            return RouteDecision(
                target=RouteTarget.HOT_PATH,
                reason="pedido vazio tratado pelo hot path",
                confidence=0.4,
            )

        direct_matches = [
            intent
            for intent in self._direct_intents
            if intent.pattern.search(normalized)
        ]
        token_count = self._token_count(normalized)
        subordinate_hits = sum(
            marker in normalized for marker in self._subordinate_markers
        )
        reasoning_hits = sum(marker in normalized for marker in self._reasoning_markers)
        multi_step_hits = sum(
            marker in normalized for marker in self._multi_step_markers
        )
        estimated_tool_depth = max(
            tool_chain_depth,
            self._estimate_tool_chain_depth(normalized, direct_matches),
        )
        complexity_score = 0
        complexity_score += reasoning_hits
        complexity_score += multi_step_hits
        complexity_score += subordinate_hits
        complexity_score += min(recalled_memories, 2)
        complexity_score += min(max(estimated_tool_depth - 1, 0), 2)
        if token_count > 12:
            complexity_score += 1
        if token_count > 24:
            complexity_score += 1
        if recent_turns > 8:
            complexity_score += 1

        if len(direct_matches) > 1 or estimated_tool_depth > 1:
            return RouteDecision(
                target=RouteTarget.DELIBERATIVE,
                reason="pedido combina multiplas acoes ou depende de cadeia de tools",
                confidence=0.92,
            )

        if direct_matches and multi_step_hits == 0 and estimated_tool_depth <= 1:
            intent = direct_matches[0]
            return RouteDecision(
                target=RouteTarget.DIRECT_TOOL,
                tool_name=intent.tool_name,
                reason=intent.reason,
                confidence=0.95,
            )

        if recalled_memories > 1:
            return RouteDecision(
                target=RouteTarget.DELIBERATIVE,
                reason="memoria recuperada exige contexto adicional",
                confidence=0.85,
            )

        if reasoning_hits > 0 or multi_step_hits > 0 or complexity_score >= 2:
            return RouteDecision(
                target=RouteTarget.DELIBERATIVE,
                reason="pedido com sinais de raciocinio, contexto ou multiplas etapas",
                confidence=0.88,
            )

        return RouteDecision(
            target=RouteTarget.HOT_PATH,
            reason="pedido curto e simples para resposta rapida",
            confidence=0.8,
        )

    def _estimate_tool_chain_depth(
        self, normalized: str, direct_matches: list[_DirectIntent]
    ) -> int:
        action_markers = 0
        action_markers += normalized.count(" e depois ")
        action_markers += normalized.count(" e entao ")
        action_markers += normalized.count(" e então ")
        action_markers += normalized.count(" depois ")
        action_markers += normalized.count(" em seguida ")
        if len(direct_matches) > 1:
            return len(direct_matches)
        if action_markers <= 0:
            return 1 if direct_matches else 0
        return max(2, action_markers + 1)

    @staticmethod
    def _token_count(text: str) -> int:
        return len([token for token in text.split(" ") if token])
