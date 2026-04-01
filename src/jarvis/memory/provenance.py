from __future__ import annotations

import re
from dataclasses import dataclass

from ..models.memory import MemorySource


@dataclass(frozen=True)
class ProvenanceResult:
    """Enriched provenance metadata for a candidate memory."""

    source: MemorySource
    confidence: float
    recency_weight: float
    scope: str


class ProvenanceEnricher:
    """Determines the origin, confidence, and scope of a memory fragment.

    The confidence score is derived from simple lexical heuristics that measure
    how strongly the content asserts a fact vs. expresses uncertainty.  It is
    intentionally cheap — the hot path must never block on ML inference.

    scope values
    ~~~~~~~~~~~~
    ``"session"``       Valid only for the current conversation.
    ``"global"``        User-level long-term fact.
    ``"project:<name>"``  Relevant to a specific project context.
    """

    # Phrases that signal explicit, reliable user assertions.
    _EXPLICIT_PATTERNS = re.compile(
        r"""
        \b(
          meu\s+nome\s+[eé]       |  # meu nome é / meu nome e
          eu\s+(sou|me\s+chamo)   |  # eu sou / eu me chamo
          eu\s+(prefiro|gosto\s+de|odeio|detesto)  |
          minha\s+(cidade|empresa|trabalho)  |
          me\s+lembra\s+que       |
          anota\s+que             |
          salva\s+que             |
          n[aã]o\s+esque[cç]a?    |
          sempre\s+(fa[cç]o|uso)  |
          todo\s+dia
        )\b
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    # Phrases that signal uncertainty — lower confidence.
    _HEDGE_PATTERNS = re.compile(
        r"\b(talvez|acho|n[aã]o\s+sei|pode\s+ser|provavelmente|quem\s+sabe)\b",
        re.IGNORECASE,
    )

    # Scope hints: project-level context signals.
    _PROJECT_PATTERNS = re.compile(
        r"\b(projeto|reposit[oó]rio|codebase|sistema|app|servi[cç]o)\s+(\w+)\b",
        re.IGNORECASE,
    )

    def enrich(self, content: str, inferred: bool = False) -> ProvenanceResult:
        """Return provenance metadata for *content*.

        Parameters
        ----------
        content:
            The raw text of the memory candidate.
        inferred:
            ``True`` if the memory was extracted by the LLM from conversation
            context rather than being stated explicitly by the user.
        """
        source = MemorySource.INFERRED if inferred else self._detect_source(content)
        confidence = self._score_confidence(content, source)
        recency_weight = 1.0
        scope = self._detect_scope(content)
        return ProvenanceResult(
            source=source,
            confidence=confidence,
            recency_weight=recency_weight,
            scope=scope,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_source(self, content: str) -> MemorySource:
        if self._EXPLICIT_PATTERNS.search(content):
            return MemorySource.EXPLICIT
        return MemorySource.INFERRED

    def _score_confidence(self, content: str, source: MemorySource) -> float:
        base = 0.85 if source == MemorySource.EXPLICIT else 0.60
        if self._HEDGE_PATTERNS.search(content):
            base *= 0.70
        return round(min(1.0, max(0.0, base)), 3)

    def _detect_scope(self, content: str) -> str:
        project_match = self._PROJECT_PATTERNS.search(content)
        if project_match:
            project_name = project_match.group(2).lower()
            return "project:%s" % project_name
        return "global"
