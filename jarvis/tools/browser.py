from __future__ import annotations

import asyncio
import urllib.parse
import urllib.request
import webbrowser


class BrowserTool:
    async def search(self, query: str) -> dict:
        url = "https://duckduckgo.com/?q=%s" % urllib.parse.quote_plus(query)
        await asyncio.to_thread(webbrowser.open, url)
        return {"query": query, "url": url}

    async def fetch_url(self, url: str) -> dict:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                content = response.read().decode('utf-8')
                
            return {
                "url": url,
                "status": "success",
                # Retorna os primeiros caracteres limpos se preferir lidar com html no prompt/routing
                "content_preview": content[:4000]
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc), "url": url}
