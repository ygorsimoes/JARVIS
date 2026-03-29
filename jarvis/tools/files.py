from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterable

from .security import ensure_path_within_roots, normalize_roots


class FilesTool:
    def __init__(self, allowed_roots: Iterable[str]) -> None:
        self._allowed_roots = normalize_roots(allowed_roots)

    async def list(self, path: str) -> dict:
        target = ensure_path_within_roots(path, self._allowed_roots)
        entries = await asyncio.to_thread(lambda: sorted(item.name for item in target.iterdir()))
        return {"path": str(target), "entries": entries}

    async def read(self, path: str) -> dict:
        target = ensure_path_within_roots(path, self._allowed_roots)
        content = await asyncio.to_thread(target.read_text)
        return {"path": str(target), "content": content}
