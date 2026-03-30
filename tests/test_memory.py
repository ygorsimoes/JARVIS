"""Tests for the J.A.R.V.I.S. Memory System.

All tests run without ML dependencies (no MLX, no sentence-transformers).
The EmbeddingProvider falls back to the deterministic stub backend.
The MemoryStore uses :memory: SQLite — no filesystem required.
"""
from __future__ import annotations

import asyncio
import pytest

from jarvis.memory.embedding import EmbeddingProvider, EMBEDDING_DIM, _serialize_f32
from jarvis.memory.provenance import ProvenanceEnricher
from jarvis.memory.relevance import RelevanceClassifier, RelevanceDecision
from jarvis.memory.store import MemoryStore
from jarvis.adapters.memory.sqlite_vec import SQLiteVecMemoryAdapter
from jarvis.models.memory import MemoryCategory, MemorySource


# ---------------------------------------------------------------------------
# EmbeddingProvider
# ---------------------------------------------------------------------------


class TestEmbeddingProvider:
    def test_stub_backend_produces_correct_dim(self):
        provider = EmbeddingProvider(preferred_backend="stub")
        assert provider.dim == EMBEDDING_DIM

    @pytest.mark.asyncio
    async def test_embed_returns_bytes(self):
        provider = EmbeddingProvider(preferred_backend="stub")
        result = await provider.embed("hello world")
        assert isinstance(result, bytes)
        # 384 floats × 4 bytes each
        assert len(result) == EMBEDDING_DIM * 4

    @pytest.mark.asyncio
    async def test_embed_deterministic(self):
        provider = EmbeddingProvider(preferred_backend="stub")
        a = await provider.embed("test phrase")
        b = await provider.embed("test phrase")
        assert a == b

    @pytest.mark.asyncio
    async def test_embed_different_texts_differ(self):
        provider = EmbeddingProvider(preferred_backend="stub")
        a = await provider.embed("hello")
        b = await provider.embed("world")
        assert a != b

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        provider = EmbeddingProvider(preferred_backend="stub")
        results = await provider.embed_batch(["foo", "bar", "baz"])
        assert len(results) == 3
        for r in results:
            assert len(r) == EMBEDDING_DIM * 4


def test_serialize_f32():
    vec = [0.1, 0.2, 0.3, 0.4]
    blob = _serialize_f32(vec)
    assert len(blob) == len(vec) * 4
    import struct
    unpacked = struct.unpack("4f", blob)
    assert abs(unpacked[0] - 0.1) < 1e-6


# ---------------------------------------------------------------------------
# ProvenanceEnricher
# ---------------------------------------------------------------------------


class TestProvenanceEnricher:
    def setup_method(self):
        self.enricher = ProvenanceEnricher()

    def test_explicit_pattern_detected(self):
        result = self.enricher.enrich("meu nome é Ygor")
        assert result.source == MemorySource.EXPLICIT
        assert result.confidence >= 0.8

    def test_hedge_reduces_confidence(self):
        result = self.enricher.enrich("talvez eu goste de Python")
        assert result.confidence < 0.75

    def test_inferred_forced(self):
        result = self.enricher.enrich("Python é legal", inferred=True)
        assert result.source == MemorySource.INFERRED

    def test_project_scope_detected(self):
        result = self.enricher.enrich("o projeto jarvis usa mlx")
        assert result.scope == "project:jarvis"

    def test_default_scope_global(self):
        result = self.enricher.enrich("eu gosto de café")
        assert result.scope == "global"

    def test_recency_weight_default(self):
        result = self.enricher.enrich("qualquer coisa")
        assert result.recency_weight == 1.0


# ---------------------------------------------------------------------------
# RelevanceClassifier
# ---------------------------------------------------------------------------


class TestRelevanceClassifier:
    def setup_method(self):
        self.clf = RelevanceClassifier()

    def test_personal_fact_persists(self):
        result = self.clf.classify("meu nome é Ygor", "Olá Ygor!")
        assert result.decision == RelevanceDecision.PERSIST
        assert result.score >= 0.5

    def test_preference_persists(self):
        result = self.clf.classify("eu prefiro Python a JavaScript", "bom saber!")
        assert result.decision == RelevanceDecision.PERSIST

    def test_smalltalk_skipped(self):
        result = self.clf.classify("olá", "Olá! Como posso ajudar?")
        assert result.decision == RelevanceDecision.SKIP

    def test_too_short_skipped(self):
        result = self.clf.classify("ok", "entendido")
        assert result.decision == RelevanceDecision.SKIP

    def test_score_in_range(self):
        result = self.clf.classify("eu trabalho na empresa X", "entendido")
        assert 0.0 <= result.score <= 1.0

    def test_obrigado_skipped(self):
        result = self.clf.classify("obrigado!", "De nada!")
        assert result.decision == RelevanceDecision.SKIP


# ---------------------------------------------------------------------------
# MemoryStore (in-memory SQLite, no ML deps)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_store_lifecycle():
    store = MemoryStore(
        db_path=":memory:",
        embedding_provider=EmbeddingProvider(preferred_backend="stub"),
    )
    await store.open()
    memory = await store.save(
        content="meu nome é Ygor",
        category=MemoryCategory.PROFILE,
        source=MemorySource.EXPLICIT,
        confidence=0.9,
    )
    assert memory.content == "meu nome é Ygor"
    assert memory.category == MemoryCategory.PROFILE
    await store.close()


@pytest.mark.asyncio
async def test_memory_store_fts_search():
    store = MemoryStore(
        db_path=":memory:",
        embedding_provider=EmbeddingProvider(preferred_backend="stub"),
    )
    await store.open()
    await store.save(
        content="meu nome é Ygor",
        category=MemoryCategory.PROFILE,
        source=MemorySource.EXPLICIT,
        confidence=0.9,
    )
    await store.save(
        content="eu moro em Fortaleza",
        category=MemoryCategory.PROFILE,
        source=MemorySource.EXPLICIT,
        confidence=0.85,
    )
    # FTS5 search
    results = await store.search_fts("Ygor", top_k=5)
    assert len(results) >= 1
    assert any("Ygor" in r.content for r in results)
    await store.close()


@pytest.mark.asyncio
async def test_memory_store_list_all():
    store = MemoryStore(
        db_path=":memory:",
        embedding_provider=EmbeddingProvider(preferred_backend="stub"),
    )
    await store.open()
    for i in range(3):
        await store.save(
            content="mem %d" % i,
            category=MemoryCategory.EPISODIC,
            source=MemorySource.INFERRED,
            confidence=0.6,
        )
    all_mems = await store.list_all()
    assert len(all_mems) == 3
    await store.close()


@pytest.mark.asyncio
async def test_memory_store_search_empty_query_returns_empty():
    store = MemoryStore(
        db_path=":memory:",
        embedding_provider=EmbeddingProvider(preferred_backend="stub"),
    )
    await store.open()
    results = await store.search("", top_k=5)
    assert results == []
    await store.close()


# ---------------------------------------------------------------------------
# SQLiteVecMemoryAdapter (high-level)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_maybe_persist_relevant():
    adapter = SQLiteVecMemoryAdapter(db_path=":memory:", embedding_backend="stub")
    await adapter.open()
    memory = await adapter.maybe_persist_turn(
        user_text="meu nome é Ygor",
        assistant_text="Olá Ygor, como posso ajudar?",
    )
    assert memory is not None
    assert "Ygor" in memory.content
    await adapter.close()


@pytest.mark.asyncio
async def test_adapter_maybe_persist_smalltalk_skipped():
    adapter = SQLiteVecMemoryAdapter(db_path=":memory:", embedding_backend="stub")
    await adapter.open()
    memory = await adapter.maybe_persist_turn(
        user_text="olá",
        assistant_text="Olá! Como posso ajudar?",
    )
    assert memory is None
    await adapter.close()


@pytest.mark.asyncio
async def test_adapter_search_after_persist():
    adapter = SQLiteVecMemoryAdapter(db_path=":memory:", embedding_backend="stub")
    await adapter.open()
    await adapter.maybe_persist_turn(
        user_text="eu prefiro usar o terminal no macOS",
        assistant_text="Anotado.",
    )
    results = await adapter.search("terminal macOS", top_k=5)
    assert len(results) >= 1
    await adapter.close()


@pytest.mark.asyncio
async def test_adapter_search_empty_store():
    adapter = SQLiteVecMemoryAdapter(db_path=":memory:", embedding_backend="stub")
    await adapter.open()
    results = await adapter.search("qualquer coisa", top_k=5)
    assert results == []
    await adapter.close()


def test_adapter_should_persist_true():
    adapter = SQLiteVecMemoryAdapter(db_path=":memory:", embedding_backend="stub")
    assert adapter.should_persist({"user_text": "meu nome é Ygor", "assistant_text": ""})


def test_adapter_should_persist_false_smalltalk():
    adapter = SQLiteVecMemoryAdapter(db_path=":memory:", embedding_backend="stub")
    assert not adapter.should_persist({"user_text": "olá", "assistant_text": "Olá!"})
