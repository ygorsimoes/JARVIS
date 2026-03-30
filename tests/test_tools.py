import unittest
from unittest.mock import ANY, AsyncMock, patch

from jarvis.config import JarvisConfig
from jarvis.tools import ToolRegistry, ToolValidationError, build_default_registry
from jarvis.tools.shell import ShellTool
from jarvis.tools.system import SystemTool, ToolExecutionError


class ToolRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_registry_rejects_invalid_payload(self):
        registry = ToolRegistry()

        async def handler(duration_seconds: int) -> dict:
            return {"duration_seconds": duration_seconds}

        registry.register(
            "timer.start",
            "Cria timer",
            handler,
            input_schema={
                "type": "object",
                "properties": {"duration_seconds": {"type": "integer", "minimum": 1}},
                "required": ["duration_seconds"],
                "additionalProperties": False,
            },
        )

        with self.assertRaises(ToolValidationError):
            await registry.execute("timer.start", duration_seconds="dez")

        with self.assertRaises(ToolValidationError):
            await registry.execute("timer.start", duration_seconds=0)

    async def test_system_tool_enforces_allowlist(self):
        tool = SystemTool(allowed_apps=["Safari"])
        with self.assertRaises(ToolExecutionError):
            await tool.open_app("Spotify")

    async def test_system_tool_opens_allowed_app(self):
        process = AsyncMock()
        process.wait.return_value = 0
        process.pid = 42
        tool = SystemTool(allowed_apps=["Safari"])

        with patch(
            "jarvis.tools.system.asyncio.create_subprocess_exec", return_value=process
        ) as spawn:
            result = await tool.open_app("Safari")

        spawn.assert_awaited_once_with("open", "-a", "Safari")
        self.assertEqual(result, {"app_name": "Safari", "pid": 42})

    async def test_system_tool_raises_when_open_command_fails(self):
        process = AsyncMock()
        process.wait.return_value = 1
        process.pid = 99
        tool = SystemTool(allowed_apps=["Safari"])

        with patch(
            "jarvis.tools.system.asyncio.create_subprocess_exec", return_value=process
        ):
            with self.assertRaises(ToolExecutionError):
                await tool.open_app("Safari")

    async def test_default_registry_uses_system_allowlist(self):
        registry = build_default_registry(JarvisConfig(system_allowed_apps=["Safari"]))
        with self.assertRaises(ToolValidationError):
            await registry.execute("system.get_time", unexpected=True)

    async def test_system_tool_denies_all_when_allowlist_is_empty(self):
        tool = SystemTool(allowed_apps=[])
        with self.assertRaises(ToolExecutionError):
            await tool.open_app("Safari")

    async def test_shell_tool_uses_exec_and_blocks_shell_operators(self):
        tool = ShellTool(allowed_commands=["pwd"])
        process = AsyncMock()
        process.communicate.return_value = (b"/tmp\n", b"")
        process.returncode = 0

        with patch(
            "jarvis.tools.shell.asyncio.create_subprocess_exec", return_value=process
        ) as spawn:
            result = await tool.execute("pwd")

        spawn.assert_awaited_once_with(
            "pwd",
            stdout=ANY,
            stderr=ANY,
        )
        self.assertEqual(result["status"], "success")

        blocked = await tool.execute("pwd; whoami")
        self.assertEqual(blocked["status"], "error")
        self.assertIn("Operadores de shell", blocked["error"])


if __name__ == "__main__":
    unittest.main()
