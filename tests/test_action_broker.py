import unittest

from jarvis.core.action_broker import ActionBroker, ActionRequest
from jarvis.core.capability_broker import (
    Capability,
    CapabilityBroker,
    CapabilityDeniedError,
)
from jarvis.models.actions import BrowserSearchAction
from jarvis.tools import ToolRegistry


class ActionBrokerTests(unittest.IsolatedAsyncioTestCase):
    async def test_executes_registered_tool_from_action(self):
        registry = ToolRegistry()

        async def handler(query: str) -> dict:
            return {"query": query, "url": "https://example.com"}

        registry.register(
            "browser.search",
            "Busca na web",
            handler,
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        )
        broker = ActionBroker(
            CapabilityBroker([Capability("browser.search", enabled=True)]),
            registry,
        )

        result = await broker.execute(
            ActionRequest.from_action(BrowserSearchAction(query="jarvis"))
        )

        self.assertEqual(result.tool_name, "browser.search")
        self.assertEqual(result.output["query"], "jarvis")

    async def test_denies_unregistered_or_disabled_capability(self):
        registry = ToolRegistry()
        broker = ActionBroker(
            CapabilityBroker([Capability("browser.search", enabled=False)]),
            registry,
        )

        with self.assertRaises(CapabilityDeniedError):
            await broker.execute(
                ActionRequest(tool_name="browser.search", arguments={})
            )


if __name__ == "__main__":
    unittest.main()
