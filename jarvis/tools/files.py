from __future__ import annotations

import asyncio
import shutil
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

    async def move(self, source_path: str, destination_path: str) -> dict:
        src = ensure_path_within_roots(source_path, self._allowed_roots)
        dst = ensure_path_within_roots(destination_path, self._allowed_roots)
        await asyncio.to_thread(shutil.move, str(src), str(dst))
        return {"source": str(src), "destination": str(dst), "status": "success"}
