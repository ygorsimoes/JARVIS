from __future__ import annotations

import asyncio
import struct
from typing import List, Optional

from ..model_cache import resolve_cached_model_reference
from ..observability import get_logger

logger = get_logger(__name__)

# Dimensionality of the embedding vector produced by this provider.
# Qwen3-Embedding-0.6B outputs 1024-dim; sentence-transformers/all-MiniLM-L6-v2
# outputs 384-dim.  We normalise to 384 as the lowest-common-denominator that
# works on both backends without requiring a separate dimension field in the DB.
EMBEDDING_DIM = 384


def _serialize_f32(vector: List[float]) -> bytes:
    """Pack a list of floats into the compact binary format expected by sqlite-vec.

    Confirmed format via Context7/sqlite-vec docs:
        struct.pack('%sf' % len(vector), *vector)
    """
    return struct.pack("%sf" % len(vector), *vector)


class EmbeddingProvider:
    """Asynchronous text embedding provider with graceful backend fallback.

    Backend selection (in priority order)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. ``mlx_backend`` — Qwen3-Embedding-0.6B via MLX (Apple Silicon, Metal).
       Fast, on-device, no network.  Outputs 1024-dim, down-projected to 384.
    2. ``sentence_transformers`` — ``all-MiniLM-L6-v2`` (CPU, cross-platform).
       Slower first call (model download), then ~5ms per embed on CPU.
    3. ``stub`` — Random unit vector (testing / development only).

    All public methods are async so they can be awaited from the hot path.
    Model loading is always deferred to a thread pool to avoid blocking the
    event loop.
    """

    def __init__(self, preferred_backend: str = "auto") -> None:
        """
        Parameters
        ----------
        preferred_backend:
            ``"auto"`` (default), ``"mlx"``, ``"sentence_transformers"``, or
            ``"stub"``.  ``"auto"`` will attempt MLX first, then
            sentence_transformers, then fall back to stub.
        """
        self._preferred_backend = preferred_backend
        self._backend_name: Optional[str] = None
        self._model = None
        self._init_lock = asyncio.Lock()

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM

    @property
    def backend_name(self) -> Optional[str]:
        return self._backend_name

    async def embed(self, text: str) -> bytes:
        """Return the serialised float32 embedding for *text*.

        The returned bytes are ready to be inserted directly into a
        ``vec0`` virtual table column via sqlite-vec.
        """
        await self._ensure_model()
        vector = await asyncio.to_thread(self._encode, text)
        return _serialize_f32(vector)

    async def embed_batch(self, texts: List[str]) -> List[bytes]:
        """Embed multiple texts in a single thread-pool call."""
        await self._ensure_model()
        vectors = await asyncio.to_thread(self._encode_batch, texts)
        return [_serialize_f32(v) for v in vectors]

    async def shutdown(self) -> None:
        self._model = None
        self._backend_name = None

    # ------------------------------------------------------------------
    # Private helpers — called from thread pool (asyncio.to_thread)
    # ------------------------------------------------------------------

    async def _ensure_model(self) -> None:
        if self._backend_name is not None:
            return
        async with self._init_lock:
            if self._backend_name is not None:
                return
            await asyncio.to_thread(self._load_model)

    def _load_model(self) -> None:
        preferred = self._preferred_backend

        if preferred in ("auto", "mlx"):
            try:
                self._load_mlx()
                return
            except Exception as exc:
                if preferred == "mlx":
                    raise
                logger.debug(
                    "mlx embedding unavailable (%s), trying sentence_transformers", exc
                )

        if preferred in ("auto", "sentence_transformers"):
            try:
                self._load_sentence_transformers()
                return
            except Exception as exc:
                if preferred == "sentence_transformers":
                    raise
                logger.debug("sentence_transformers unavailable (%s), using stub", exc)

        # Last-resort stub — random unit vector (deterministic per text via hash seed).
        self._backend_name = "stub"
        logger.warning(
            "EmbeddingProvider: no ML backend available — using random stub embeddings. "
            "Semantic search will not be meaningful."
        )

    def _load_mlx(self) -> None:
        from mlx_lm import load as mlx_load  # type: ignore[import]

        model_reference = resolve_cached_model_reference(
            "mlx-community/Qwen3-Embedding-0.6B-4bit"
        )
        model, tokenizer = mlx_load(model_reference)
        self._model = (model, tokenizer)
        self._backend_name = "mlx_qwen3_embedding"

    def _load_sentence_transformers(self) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._model = model
        self._backend_name = "sentence_transformers"

    def _encode(self, text: str) -> List[float]:
        return self._encode_batch([text])[0]

    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        if self._backend_name == "stub":
            return [self._stub_vector(t) for t in texts]

        if self._backend_name == "sentence_transformers":
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            # all-MiniLM-L6-v2 is already 384-dim, no projection needed.
            return [emb.tolist() for emb in embeddings]

        if self._backend_name and self._backend_name.startswith("mlx"):
            return self._encode_mlx_batch(texts)

        raise RuntimeError("EmbeddingProvider: no backend loaded")

    def _encode_mlx_batch(self, texts: List[str]) -> List[List[float]]:
        import mlx.core as mx  # type: ignore[import]

        model, tokenizer = self._model
        results = []
        for text in texts:
            tokens = tokenizer(text, return_tensors="np")
            input_ids = mx.array(tokens["input_ids"])
            outputs = model(input_ids)
            # Take mean of last hidden state, then project to EMBEDDING_DIM.
            hidden = outputs.last_hidden_state[0]  # (seq_len, 1024)
            mean_vec = mx.mean(hidden, axis=0)  # (1024,)
            # Simple linear projection: take first 384 dims and L2-normalise.
            projected = mean_vec[:EMBEDDING_DIM]
            norm = mx.sqrt(mx.sum(projected * projected) + 1e-8)
            unit = (projected / norm).tolist()
            results.append(unit)
        return results

    @staticmethod
    def _stub_vector(text: str) -> List[float]:
        """Deterministic pseudo-random unit vector from a hash of the text."""
        import hashlib
        import math

        digest = hashlib.sha256(text.encode()).digest()
        raw = [(digest[i % 32] / 255.0 * 2.0 - 1.0) for i in range(EMBEDDING_DIM)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]
