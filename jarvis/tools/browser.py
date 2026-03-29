from __future__ import annotations

import asyncio
import urllib.parse
import webbrowser


class BrowserTool:
    async def search(self, query: str) -> dict:
        url = "https://duckduckgo.com/?q=%s" % urllib.parse.quote_plus(query)
        await asyncio.to_thread(webbrowser.open, url)
        return {"query": query, "url": url}
