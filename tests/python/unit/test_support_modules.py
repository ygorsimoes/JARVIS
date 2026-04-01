import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from jarvis.tools.browser import BrowserTool
from jarvis.tools.files import FilesTool
from jarvis.tools.security import (
    build_trust_metadata,
    ensure_path_within_roots,
    normalize_roots,
    validate_http_url,
)
from jarvis.tools.timer import TimerTool, parse_duration_seconds


@pytest.mark.asyncio
class TestSupportModules:
    async def test_browser_tool_returns_search_url(self):
        with patch("jarvis.tools.browser.webbrowser.open") as open_browser:
            result = await BrowserTool().search("jarvis local ai")

        open_browser.assert_called_once()
        assert result["query"] == "jarvis local ai"
        assert "duckduckgo.com" in result["url"]

    async def test_browser_fetch_url_rejects_non_http_schemes(self):
        result = await BrowserTool().fetch_url("file:///etc/passwd")

        assert result["status"] == "error"
        assert "HTTP(s)" in result["error"]

    async def test_files_tool_lists_and_reads_only_allowed_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            allowed_file = root / "notes.txt"
            allowed_file.write_text("conteudo")
            other_root = root.parent / "forbidden.txt"
            tool = FilesTool([str(root)])

            listing = await tool.list(str(root))
            content = await tool.read(str(allowed_file))

            assert listing["entries"] == ["notes.txt"]
            assert content["content"] == "conteudo"

            with pytest.raises(PermissionError):
                await tool.read(str(other_root))

    async def test_timer_tool_lifecycle_and_duration_parser(self):
        tool = TimerTool()

        started = await tool.start(90, label="Cafe")
        timers = await tool.list()
        cancelled = await tool.cancel(started["timer_id"])

        assert started["label"] == "Cafe"
        assert len(timers) == 1
        assert cancelled == {"timer_id": started["timer_id"], "cancelled": True}
        assert parse_duration_seconds("1 hora 30 minutos") == 5400
        assert parse_duration_seconds("15 min") == 900
        assert parse_duration_seconds("45 segundos") == 45
        assert parse_duration_seconds("sem tempo") is None

    async def test_timer_cancel_unknown_returns_false(self):
        tool = TimerTool()

        cancelled = await tool.cancel("missing-timer")

        assert cancelled == {"timer_id": "missing-timer", "cancelled": False}


class TestSecurityHelpers:
    def test_normalize_roots_and_enforce_bounds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            child = root / "subdir" / "arquivo.txt"
            child.parent.mkdir()
            child.write_text("ok")

            roots = normalize_roots([str(root), ""])
            resolved = ensure_path_within_roots(str(child), roots)

            assert roots == [root.resolve()]
            assert resolved == child.resolve()

    def test_http_url_validation_and_trust_metadata(self):
        assert (
            validate_http_url("https://example.com/docs") == "https://example.com/docs"
        )

        with pytest.raises(ValueError):
            validate_http_url("mailto:test@example.com")

        assert build_trust_metadata(source="external_url", trusted=False) == {
            "source": "external_url",
            "trusted": False,
        }
