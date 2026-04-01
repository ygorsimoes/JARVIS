from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ..models.memory import MemoryCategory


class RelevanceDecision(str, Enum):
    """Outcome of the relevance classifier for a given turn."""

    PERSIST = "persist"  # Save to long-term memory.
    SKIP = "skip"  # Too ephemeral or low-value.
    SESSION_ONLY = "session"  # Keep in working memory, do not persist.


@dataclass(frozen=True)
class ClassificationResult:
    decision: RelevanceDecision
    reason: str
    score: float
    category: MemoryCategory | None = None


class RelevanceClassifier:
    """Decides whether a conversation turn should be persisted as a memory.

    Design contract
    ~~~~~~~~~~~~~~~
    * Zero ML dependencies — pure regex + heuristics, O(n) on input length.
    * Must be safe to call synchronously inside the asyncio event loop.
    * Score is in [0.0, 1.0].  Threshold for PERSIST is ``>= 0.5``.
    """

    # --- High-value patterns: facts worth persisting. ----------------

    _PERSONAL_FACT_RE = re.compile(
        r"""
        \b(
          meu\s+nome\s+[eé]                 |
          eu\s+(sou|me\s+chamo)             |
          eu\s+(trabalho|moro|vivo)\s+       |
          minha\s+(cidade|empresa|profiss)   |
          eu\s+(prefiro|gosto\s+de|odeio|detesto)  |
          meu\s+(time|partido|hobby|esporte) |
          eu\s+tenho\s+\d+\s+anos           |
          meu\s+(email|telefone|endere[cç]o) |
          n[aã]o\s+(gosto|suporto)\s+de     |
          (me\s+lembra?|anota|salva)\s+que  |
          n[aã]o\s+esque[cç]a?
        )\b
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    _PREFERENCE_RE = re.compile(
        r"""
        \b(
          prefiro\s+            |  # prefiro X a Y
          sempre\s+(fa[cç]o|uso|como|bebo)  |
          todo\s+(dia|manha|tarde|fim\s+de\s+semana)  |
          rotina\b              |
          h[aá]bito\b
        )\b
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    _PROCEDURE_RE = re.compile(
        r"""
        \b(
          para\s+(fazer|instalar|configurar|rodar|executar)  |
          o\s+comando\s+[eé]     |
          a\s+senha\s+[eé]       |
          o\s+processo\s+[eé]    |
          o\s+workflow\s+[eé]
        )\b
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    # --- Low-value patterns: ephemeral / small-talk. -----------------

    _SMALLTALK_RE = re.compile(
        r"""
        ^(
            ol[aá]          |
            oi\b            |
            tudo\s+bem      |
            que\s+horas\s+s[aã]o |
            obrigado?       |
            valeu           |
            ok\b            |
            certo\b         |
            entendido
        )[\s!?.]*$
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    _SHORT_THRESHOLD = 12  # chars — too short to carry a useful fact

    def classify(self, user_text: str, assistant_text: str) -> ClassificationResult:
        """Classify a completed turn for memory persistence.

        Parameters
        ----------
        user_text:
            The user's utterance.
        assistant_text:
            The assistant's response (used to detect confirmations).
        """
        combined = "%s %s" % (user_text, assistant_text)

        if self._SMALLTALK_RE.search(user_text.strip()):
            return ClassificationResult(
                decision=RelevanceDecision.SKIP,
                reason="small talk",
                score=0.05,
                category=None,
            )

        if len(user_text.strip()) < self._SHORT_THRESHOLD:
            return ClassificationResult(
                decision=RelevanceDecision.SKIP,
                reason="too short",
                score=0.10,
                category=None,
            )

        score = 0.0
        category = self._infer_category(combined)

        if self._PERSONAL_FACT_RE.search(combined):
            score += 0.70

        if self._PREFERENCE_RE.search(combined):
            score += 0.30

        if self._PROCEDURE_RE.search(combined):
            score += 0.55

        score = min(1.0, score)

        if score >= 0.5:
            return ClassificationResult(
                decision=RelevanceDecision.PERSIST,
                reason="relevant personal/procedural content (score=%.2f)" % score,
                score=score,
                category=category,
            )

        if score >= 0.2:
            return ClassificationResult(
                decision=RelevanceDecision.SESSION_ONLY,
                reason="low relevance — keep in session only (score=%.2f)" % score,
                score=score,
                category=category,
            )

        return ClassificationResult(
            decision=RelevanceDecision.SKIP,
            reason="ephemeral content (score=%.2f)" % score,
            score=score,
            category=category,
        )

    def _infer_category(self, content: str) -> MemoryCategory:
        if self._PERSONAL_FACT_RE.search(content) or self._PREFERENCE_RE.search(
            content
        ):
            return MemoryCategory.PROFILE
        if self._PROCEDURE_RE.search(content):
            return MemoryCategory.PROCEDURAL
        return MemoryCategory.EPISODIC
