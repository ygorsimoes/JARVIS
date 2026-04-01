from __future__ import annotations

import asyncio
import math
import re
import sqlite3
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from ..models.memory import Memory, MemoryCategory, MemorySource
from ..observability import get_logger
from .embedding import EMBEDDING_DIM, EmbeddingProvider

logger = get_logger(__name__)

_CREATE_MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    category        TEXT NOT NULL,
    source          TEXT NOT NULL,
    confidence      REAL NOT NULL,
    recency_weight  REAL NOT NULL DEFAULT 1.0,
    scope           TEXT NOT NULL DEFAULT 'global',
    created_at      TEXT NOT NULL,
    last_accessed   TEXT NOT NULL
);
"""

_CREATE_FTS5_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    content='memories',
    content_rowid='rowid',
    tokenize='unicode61'
);
"""

_CREATE_FTS5_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS memories_ai
    AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, content)
        VALUES (new.rowid, new.content);
    END;
CREATE TRIGGER IF NOT EXISTS memories_ad
    AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, content)
        VALUES ('delete', old.rowid, old.content);
    END;
CREATE TRIGGER IF NOT EXISTS memories_au
    AFTER UPDATE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, content)
        VALUES ('delete', old.rowid, old.content);
        INSERT INTO memories_fts(rowid, content)
        VALUES (new.rowid, new.content);
    END;
"""

# sqlite-vec virtual table — dimension matches EmbeddingProvider.EMBEDDING_DIM
_CREATE_VEC_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
    memory_id TEXT PRIMARY KEY,
    embedding float[{dim}]
);
""".format(dim=EMBEDDING_DIM)


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _normalize_fts_query(query: str) -> str:
    tokens = re.findall(r"[\wÀ-ÿ]+", query, flags=re.UNICODE)
    return " ".join(tokens)


def _serialize_f32(vector: List[float]) -> bytes:
    return struct.pack("%sf" % len(vector), *vector)


def _row_to_memory(row: sqlite3.Row) -> Memory:
    return Memory(
        content=row["content"],
        category=MemoryCategory(row["category"]),
        source=MemorySource(row["source"]),
        confidence=row["confidence"],
        recency_weight=row["recency_weight"],
        scope=row["scope"],
        created_at=datetime.fromisoformat(row["created_at"]),
        last_accessed=datetime.fromisoformat(row["last_accessed"]),
    )


def _row_to_memory_with_overrides(
    row: sqlite3.Row,
    *,
    recency_weight: float | None = None,
    last_accessed: datetime | None = None,
) -> Memory:
    memory = _row_to_memory(row)
    if recency_weight is not None:
        memory.recency_weight = recency_weight
    if last_accessed is not None:
        memory.last_accessed = last_accessed
    return memory


class MemoryStore:
    """Persistent memory store backed by SQLite + FTS5 + sqlite-vec.

    Architecture
    ~~~~~~~~~~~~
    *  ``memories``      — canonical rows with all metadata.
    *  ``memories_fts``  — FTS5 virtual table for fast lexical search.
    *  ``memories_vec``  — vec0 virtual table for semantic kNN search.

    All blocking I/O is executed in ``asyncio.to_thread`` to avoid stalling
    the event loop.  The public API is fully async.

    sqlite-vec pin
    ~~~~~~~~~~~~~~
    This class requires sqlite-vec==0.1.3.  The extension is loaded via the
    standard ``sqlite_vec.load(db)`` call.  If the package is not installed the
    store degrades gracefully — FTS5 search remains functional; semantic search
    is disabled.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        self._db_path = str(db_path)
        self._embedding = embedding_provider or EmbeddingProvider(
            preferred_backend="auto"
        )
        self._conn: Optional[sqlite3.Connection] = None
        self._vec_available = False
        self._init_lock = asyncio.Lock()
        self._pending_embedding_tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def open(self) -> None:
        """Open (or create) the database and initialise the schema."""
        async with self._init_lock:
            if self._conn is not None:
                return
            await asyncio.to_thread(self._sync_open)

    async def close(self) -> None:
        if self._pending_embedding_tasks:
            await asyncio.gather(*self._pending_embedding_tasks, return_exceptions=True)
        if self._conn is not None:
            await asyncio.to_thread(self._conn.close)
            self._conn = None

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def save(
        self,
        content: str,
        category: MemoryCategory,
        source: MemorySource,
        confidence: float,
        recency_weight: float = 1.0,
        scope: str = "global",
    ) -> Memory:
        """Persist a new memory and (asynchronously) index its embedding."""
        await self._ensure_open()
        now = _utc_now()
        memory_id = str(uuid.uuid4())
        memory = Memory(
            content=content,
            category=category,
            source=source,
            confidence=confidence,
            recency_weight=recency_weight,
            scope=scope,
            created_at=datetime.fromisoformat(now),
            last_accessed=datetime.fromisoformat(now),
        )
        await asyncio.to_thread(self._sync_insert, memory_id, memory)
        # Embed asynchronously — fire and forget to avoid blocking the caller.
        task = asyncio.create_task(self._embed_and_store(memory_id, content))
        self._pending_embedding_tasks.add(task)
        task.add_done_callback(self._pending_embedding_tasks.discard)
        return memory

    async def delete(self, memory_id: str) -> None:
        await self._ensure_open()
        await asyncio.to_thread(self._sync_delete, memory_id)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def search_fts(self, query: str, top_k: int = 5) -> List[Memory]:
        """Fast lexical search via FTS5.  Target latency: < 5ms.

        This is the primary retrieval path for the hot path (Foundation Models).
        """
        normalized_query = _normalize_fts_query(query)
        if not normalized_query:
            return []
        await self._ensure_open()
        return await asyncio.to_thread(
            self._sync_fts_search,
            normalized_query,
            top_k,
        )

    async def search_semantic(self, query: str, top_k: int = 5) -> List[Memory]:
        """Semantic kNN search via sqlite-vec.

        Falls back to FTS5 if the vec backend is unavailable.
        Target latency: < 50ms (dominated by embedding generation).
        """
        if not query.strip():
            return []
        await self._ensure_open()
        if not self._vec_available:
            logger.debug("sqlite-vec not available; falling back to FTS5 search")
            return await self.search_fts(query, top_k)

        query_bytes = await self._embedding.embed(query)
        return await asyncio.to_thread(self._sync_vec_search, query, query_bytes, top_k)

    async def search(self, query: str, top_k: int = 5) -> List[Memory]:
        """Hybrid search: FTS5 first, then deduplicate with semantic results."""
        fts_results = await self.search_fts(query, top_k)
        if not self._vec_available or not query.strip():
            return fts_results

        sem_results = await self.search_semantic(query, top_k)
        # Merge, deduplicate by content, preserve FTS order.
        seen: set[str] = {m.content for m in fts_results}
        merged = list(fts_results)
        for m in sem_results:
            if m.content not in seen:
                merged.append(m)
                seen.add(m.content)
        return merged[:top_k]

    async def list_all(self, limit: int = 100) -> List[Memory]:
        await self._ensure_open()
        return await asyncio.to_thread(self._sync_list_all, limit)

    # ------------------------------------------------------------------
    # Synchronous helpers (run inside asyncio.to_thread)
    # ------------------------------------------------------------------

    def _sync_open(self) -> None:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.executescript(_CREATE_MEMORIES_TABLE)
        conn.executescript(_CREATE_FTS5_TABLE)
        conn.executescript(_CREATE_FTS5_TRIGGERS)
        self._conn = conn

        # Attempt to load sqlite-vec extension.
        try:
            import sqlite_vec  # type: ignore[import]

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            conn.executescript(_CREATE_VEC_TABLE)
            self._vec_available = True
            logger.debug("sqlite-vec loaded — semantic search enabled")
        except Exception as exc:
            logger.info(
                "sqlite-vec not available (%s) — semantic search disabled, FTS5 only",
                exc,
            )
            self._vec_available = False

    def _sync_insert(self, memory_id: str, memory: Memory) -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            INSERT INTO memories
                (id, content, category, source, confidence, recency_weight,
                 scope, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                memory.content,
                memory.category.value,
                memory.source.value,
                memory.confidence,
                memory.recency_weight,
                memory.scope,
                memory.created_at.isoformat(),
                memory.last_accessed.isoformat(),
            ),
        )
        self._conn.commit()

    def _sync_delete(self, memory_id: str) -> None:
        assert self._conn is not None
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        if self._vec_available:
            self._conn.execute(
                "DELETE FROM memories_vec WHERE memory_id = ?", (memory_id,)
            )
        self._conn.commit()

    def _sync_fts_search(self, query: str, top_k: int) -> List[Memory]:
        assert self._conn is not None
        candidate_limit = max(top_k * 4, top_k)
        rows = self._conn.execute(
            """
            SELECT m.*
            FROM memories m
            JOIN memories_fts f ON m.rowid = f.rowid
            WHERE memories_fts MATCH ?
            ORDER BY bm25(memories_fts)
            LIMIT ?
            """,
            (query, candidate_limit),
        ).fetchall()
        return self._rerank_and_mark_accessed(rows, query, top_k)

    def _sync_vec_search(
        self, query: str, query_bytes: bytes, top_k: int
    ) -> List[Memory]:
        assert self._conn is not None
        candidate_limit = max(top_k * 4, top_k)
        rows = self._conn.execute(
            """
            SELECT m.*
            FROM memories m
            JOIN memories_vec v ON m.id = v.memory_id
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY distance
            """,
            (query_bytes, candidate_limit),
        ).fetchall()
        return self._rerank_and_mark_accessed(rows, query, top_k)

    def _sync_list_all(self, limit: int) -> List[Memory]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def _sync_store_embedding(self, memory_id: str, embedding_bytes: bytes) -> None:
        assert self._conn is not None
        self._conn.execute(
            "INSERT OR REPLACE INTO memories_vec(memory_id, embedding) VALUES (?, ?)",
            (memory_id, embedding_bytes),
        )
        self._conn.commit()

    def _rerank_and_mark_accessed(
        self, rows: list[sqlite3.Row], query: str, top_k: int
    ) -> list[Memory]:
        if not rows:
            return []

        ranked_rows = sorted(
            list(enumerate(rows)),
            key=lambda item: self._memory_rank(
                item[1],
                query=query,
                source_index=item[0],
                total_candidates=len(rows),
            ),
            reverse=True,
        )[:top_k]
        selected_rows = [row for _, row in ranked_rows]
        accessed_at = datetime.now(tz=timezone.utc)
        self._sync_mark_accessed([row["id"] for row in selected_rows], accessed_at)
        return [
            _row_to_memory_with_overrides(
                row,
                recency_weight=self._decayed_recency(row, now=accessed_at),
                last_accessed=accessed_at,
            )
            for row in selected_rows
        ]

    def _sync_mark_accessed(
        self, memory_ids: list[str], accessed_at: datetime | None = None
    ) -> None:
        assert self._conn is not None
        if not memory_ids:
            return

        timestamp = (accessed_at or datetime.now(tz=timezone.utc)).isoformat()
        self._conn.executemany(
            "UPDATE memories SET last_accessed = ? WHERE id = ?",
            [(timestamp, memory_id) for memory_id in memory_ids],
        )
        self._conn.commit()

    def _memory_rank(
        self,
        row: sqlite3.Row,
        *,
        query: str,
        source_index: int,
        total_candidates: int,
    ) -> float:
        signal_score = 1.0 - (source_index / (total_candidates + 1))
        confidence = max(0.0, min(1.0, float(row["confidence"])))
        recency_score = self._decayed_recency(row)
        scope_score = self._scope_score(str(row["scope"]), query)
        return (
            signal_score * 0.45
            + confidence * 0.35
            + recency_score * 0.15
            + scope_score * 0.05
        )

    def _decayed_recency(self, row: sqlite3.Row, now: datetime | None = None) -> float:
        reference = now or datetime.now(tz=timezone.utc)
        last_accessed = datetime.fromisoformat(row["last_accessed"])
        hours_since_access = max(
            0.0, (reference - last_accessed).total_seconds() / 3600.0
        )
        decay = math.exp(-hours_since_access / 168.0)
        return max(0.05, min(1.0, float(row["recency_weight"]) * decay))

    @staticmethod
    def _scope_score(scope: str, query: str) -> float:
        normalized_query = query.lower()
        if scope == "global":
            return 1.0
        if scope == "session":
            return 0.85
        if scope.startswith("project:"):
            project_name = scope.split(":", 1)[1]
            if project_name and project_name in normalized_query:
                return 1.15
            return 0.7
        return 0.8

    # ------------------------------------------------------------------
    # Async embedding fire-and-forget
    # ------------------------------------------------------------------

    async def _embed_and_store(self, memory_id: str, content: str) -> None:
        if not self._vec_available:
            return
        try:
            embedding_bytes = await self._embedding.embed(content)
            await asyncio.to_thread(
                self._sync_store_embedding, memory_id, embedding_bytes
            )
        except Exception as exc:
            logger.warning("Failed to embed memory %s: %s", memory_id, exc)

    async def _ensure_open(self) -> None:
        if self._conn is None:
            await self.open()
