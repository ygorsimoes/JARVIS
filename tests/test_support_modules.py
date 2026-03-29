import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jarvis.tools.browser import BrowserTool
from jarvis.tools.files import FilesTool
from jarvis.tools.security import ensure_path_within_roots, normalize_roots
from jarvis.tools.timer import TimerTool, parse_duration_seconds


class SupportModulesTests(unittest.IsolatedAsyncioTestCase):
    async def test_browser_tool_returns_search_url(self):
        with patch("jarvis.tools.browser.webbrowser.open") as open_browser:
            result = await BrowserTool().search("jarvis local ai")

        open_browser.assert_called_once()
        self.assertEqual(result["query"], "jarvis local ai")
        self.assertIn("duckduckgo.com", result["url"])

    async def test_files_tool_lists_and_reads_only_allowed_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            allowed_file = root / "notes.txt"
            allowed_file.write_text("conteudo")
            other_root = root.parent / "forbidden.txt"
            tool = FilesTool([str(root)])

            listing = await tool.list(str(root))
            content = await tool.read(str(allowed_file))

            self.assertEqual(listing["entries"], ["notes.txt"])
            self.assertEqual(content["content"], "conteudo")

            with self.assertRaises(PermissionError):
                await tool.read(str(other_root))

    async def test_timer_tool_lifecycle_and_duration_parser(self):
        tool = TimerTool()

        started = await tool.start(90, label="Cafe")
        timers = await tool.list()
        cancelled = await tool.cancel(started["timer_id"])

        self.assertEqual(started["label"], "Cafe")
        self.assertEqual(len(timers), 1)
        self.assertEqual(
            cancelled, {"timer_id": started["timer_id"], "cancelled": True}
        )
        self.assertEqual(parse_duration_seconds("1 hora 30 minutos"), 5400)
        self.assertEqual(parse_duration_seconds("15 min"), 900)
        self.assertEqual(parse_duration_seconds("45 segundos"), 45)
        self.assertIsNone(parse_duration_seconds("sem tempo"))


class SecurityHelpersTests(unittest.TestCase):
    def test_normalize_roots_and_enforce_bounds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            child = root / "subdir" / "arquivo.txt"
            child.parent.mkdir()
            child.write_text("ok")

            roots = normalize_roots([str(root), ""])
            resolved = ensure_path_within_roots(str(child), roots)

            self.assertEqual(roots, [root.resolve()])
            self.assertEqual(resolved, child.resolve())


if __name__ == "__main__":
    unittest.main()
