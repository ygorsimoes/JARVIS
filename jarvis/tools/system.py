from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Iterable


class ToolExecutionError(RuntimeError):
    pass


class SystemTool:
    def __init__(self, allowed_apps: Iterable[str] | None = None) -> None:
        self._allowed_apps = {item.strip().lower(): item.strip() for item in (allowed_apps or []) if item.strip()}

    async def get_time(self) -> dict:
        now = datetime.now().strftime("%H:%M")
        return {"time": now}

    async def open_app(self, app_name: str) -> dict:
        normalized = app_name.strip().lower()
        if self._allowed_apps and normalized not in self._allowed_apps:
            raise ToolExecutionError("app %s is not in the allowlist" % app_name)

        canonical_name = self._allowed_apps.get(normalized, app_name.strip())
        process = await asyncio.create_subprocess_exec("open", "-a", canonical_name)
        return_code = await process.wait()
        if return_code != 0:
            raise ToolExecutionError("failed to open app %s" % canonical_name)
        return {"app_name": canonical_name, "pid": process.pid}
