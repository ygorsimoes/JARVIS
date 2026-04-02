from __future__ import annotations

import asyncio

from ..models.conversation import RouteTarget
from ..models.memory import Memory
from ..observability import get_logger

logger = get_logger(__name__)


class MemorySystem:
    def __init__(
        self,
        adapter,
        hot_path_top_k: int = 3,
        deliberative_top_k: int = 5,
    ) -> None:
        self._adapter = adapter
        self._hot_path_top_k = hot_path_top_k
        self._deliberative_top_k = deliberative_top_k

    async def recall(
        self,
        query: str,
        route_target: RouteTarget,
        top_k: int | None = None,
    ) -> list[Memory]:
        if not query.strip() or route_target == RouteTarget.DIRECT_TOOL:
            return []

        try:
            if route_target == RouteTarget.HOT_PATH:
                search = getattr(self._adapter, "search_fts", None)
                if search is None:
                    search = self._adapter.search
                return await search(query, top_k=top_k or self._hot_path_top_k)

            semantic_search = getattr(self._adapter, "search_semantic", None)
            lexical_search = getattr(self._adapter, "search_fts", None)
            if semantic_search is None or lexical_search is None:
                search = semantic_search or lexical_search or self._adapter.search
                return await search(query, top_k=top_k or self._deliberative_top_k)

            limit = top_k or self._deliberative_top_k
            lexical_results, semantic_results = await asyncio.gather(
                lexical_search(query, top_k=limit),
                semantic_search(query, top_k=limit),
            )
            merged: list[Memory] = []
            seen: set[str] = set()
            for memory in list(lexical_results) + list(semantic_results):
                key = "%s|%s|%s" % (memory.content, memory.scope, memory.category.value)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(memory)
            return merged[:limit]
        except Exception as exc:
            logger.warning(
                "Memory recall failed for route %s: %s", route_target.value, exc
            )
            return []

    async def maybe_persist_turn(self, user_text: str, assistant_text: str):
        return await self._adapter.maybe_persist_turn(user_text, assistant_text)
