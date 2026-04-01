from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from ...memory.embedding import EmbeddingProvider
from ...memory.provenance import ProvenanceEnricher
from ...memory.relevance import RelevanceClassifier, RelevanceDecision
from ...memory.store import MemoryStore
from ...models.memory import Memory, MemoryCategory
from ...observability import get_logger

logger = get_logger(__name__)


class SQLiteVecMemoryAdapter:
    """Concrete MemoryAdapter backed by SQLiteVec MemoryStore.

    Implements the ``MemoryAdapter`` Protocol defined in
    ``jarvis.adapters.interfaces``.

    Responsibilities
    ~~~~~~~~~~~~~~~~
    * Wrap ``MemoryStore`` with higher-level semantics (relevance + provenance).
    * Provide a simple async API consumed by ``JarvisRuntime``.
    * Expose ``open()`` / ``close()`` for lifecycle management.

    Usage
    ~~~~~
    ::

        adapter = SQLiteVecMemoryAdapter(db_path="~/.jarvis/memory.db")
        await adapter.open()

        # Persist a turn if relevant:
        await adapter.maybe_persist_turn(user_text, assistant_text)

        # Retrieve before composing prompt:
        memories = await adapter.search(user_text, top_k=5)
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        embedding_backend: str = "auto",
    ) -> None:
        self._store = MemoryStore(
            db_path=db_path,
            embedding_provider=EmbeddingProvider(preferred_backend=embedding_backend),
        )
        self._relevance = RelevanceClassifier()
        self._provenance = ProvenanceEnricher()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def open(self) -> None:
        await self._store.open()
        logger.debug("SQLiteVecMemoryAdapter: store opened")

    async def close(self) -> None:
        await self._store.close()

    async def shutdown(self) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # MemoryAdapter Protocol
    # ------------------------------------------------------------------

    async def search(self, query: str, top_k: int = 5) -> List[Memory]:
        """Retrieve up to *top_k* memories relevant to *query*.

        Uses FTS5 for the hot path (< 5ms) and falls back to hybrid search
        (FTS5 + semantic) when sqlite-vec is available.
        """
        if not query.strip():
            return []
        try:
            return await self._store.search(query, top_k=top_k)
        except Exception as exc:
            logger.warning("Memory search failed: %s", exc)
            return []

    async def search_fts(self, query: str, top_k: int = 5) -> List[Memory]:
        if not query.strip():
            return []
        try:
            return await self._store.search_fts(query, top_k=top_k)
        except Exception as exc:
            logger.warning("FTS memory search failed: %s", exc)
            return []

    async def search_semantic(self, query: str, top_k: int = 5) -> List[Memory]:
        if not query.strip():
            return []
        try:
            return await self._store.search_semantic(query, top_k=top_k)
        except Exception as exc:
            logger.warning("Semantic memory search failed: %s", exc)
            return []

    async def save(self, content: str, metadata: Optional[dict] = None) -> Memory:
        """Persist a memory with auto-detected provenance."""
        meta = metadata or {}
        inferred = meta.get("inferred", False)
        enriched = self._provenance.enrich(content, inferred=inferred)

        classification = self._relevance.classify(content, "")
        default_category = classification.category or MemoryCategory.EPISODIC
        category_str = meta.get("category", default_category.value)
        try:
            category = MemoryCategory(category_str)
        except ValueError:
            category = default_category

        return await self._store.save(
            content=content,
            category=category,
            source=enriched.source,
            confidence=enriched.confidence,
            recency_weight=enriched.recency_weight,
            scope=enriched.scope,
        )

    def should_persist(self, turn: Any) -> bool:
        """Return True if a completed turn carries persistable facts.

        *turn* may be any object with ``user_text`` / ``assistant_text``
        string attributes, or a dict with those keys.
        """
        user_text = _extract_text(turn, "user_text")
        assistant_text = _extract_text(turn, "assistant_text")
        result = self._relevance.classify(user_text, assistant_text)
        return result.decision == RelevanceDecision.PERSIST

    # ------------------------------------------------------------------
    # High-level convenience: called after every turn
    # ------------------------------------------------------------------

    async def maybe_persist_turn(
        self,
        user_text: str,
        assistant_text: str,
        category: MemoryCategory | None = None,
    ) -> Optional[Memory]:
        """Classify the turn and persist it if relevant.

        Returns the saved Memory if persisted, else ``None``.
        """
        result = self._relevance.classify(user_text, assistant_text)
        if result.decision != RelevanceDecision.PERSIST:
            logger.debug(
                "Turn not persisted: %s (score=%.2f)", result.reason, result.score
            )
            return None

        # Use user_text as the canonical content to store.
        enriched = self._provenance.enrich(user_text)
        resolved_category = category or result.category or MemoryCategory.EPISODIC
        memory = await self._store.save(
            content=user_text,
            category=resolved_category,
            source=enriched.source,
            confidence=enriched.confidence,
            recency_weight=enriched.recency_weight,
            scope=enriched.scope,
        )
        logger.debug(
            "Persisted memory (score=%.2f, scope=%s): %.60s…",
            result.score,
            memory.scope,
            user_text,
        )
        return memory


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _extract_text(obj: Any, key: str) -> str:
    if isinstance(obj, dict):
        return str(obj.get(key, ""))
    return str(getattr(obj, key, ""))
