import unittest

from jarvis.core.action_broker import ActionAuditEntry, ActionBroker, ActionRequest
from jarvis.core.capability_broker import (
    Capability,
    CapabilityBroker,
    CapabilityDeniedError,
    ConfirmationRequiredError,
    RiskLevel,
)
from jarvis.models.actions import BrowserSearchAction, SetVolumeAction
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

    async def test_describe_available_tools_filters_disabled_capabilities(self):
        registry = ToolRegistry()

        async def search_handler(query: str) -> dict:
            return {"query": query}

        async def shell_handler(command: str) -> dict:
            return {"command": command}

        registry.register("browser.search", "Busca na web", search_handler)
        registry.register("shell.execute", "Comando shell", shell_handler)
        broker = ActionBroker(
            CapabilityBroker(
                [
                    Capability("browser.search", enabled=True),
                    Capability("shell.execute", enabled=False),
                ]
            ),
            registry,
        )

        described = broker.describe_available_tools()

        self.assertEqual([tool["name"] for tool in described], ["browser.search"])

    async def test_execute_records_audit_and_side_effects(self):
        registry = ToolRegistry()

        async def handler(level: int) -> dict:
            return {"volume": level}

        registry.register(
            "system.set_volume",
            "Ajusta volume",
            handler,
            input_schema={
                "type": "object",
                "properties": {"level": {"type": "integer", "minimum": 0}},
                "required": ["level"],
                "additionalProperties": False,
            },
        )
        broker = ActionBroker(
            CapabilityBroker(
                [
                    Capability(
                        "system.set_volume",
                        enabled=True,
                        requires_confirmation=True,
                        side_effects=["system_volume"],
                    )
                ]
            ),
            registry,
        )

        result = await broker.execute(
            ActionRequest.from_action(
                SetVolumeAction(level=30),
                confirmed=True,
                source="llm_tool",
            )
        )

        self.assertEqual(result.side_effects, ["system_volume"])
        self.assertTrue(result.confirmed)
        self.assertTrue(result.audit_logged)
        self.assertEqual(
            broker.audit_log,
            [
                ActionAuditEntry(
                    tool_name="system.set_volume",
                    scope="global",
                    source="llm_tool",
                    status="executed",
                    confirmed=True,
                    side_effects=["system_volume"],
                    created_at=broker.audit_log[0].created_at,
                )
            ],
        )

    async def test_describe_available_tools_includes_capability_metadata(self):
        registry = ToolRegistry()

        async def handler(level: int) -> dict:
            return {"volume": level}

        registry.register("system.set_volume", "Ajusta volume", handler)
        broker = ActionBroker(
            CapabilityBroker(
                [
                    Capability(
                        "system.set_volume",
                        enabled=True,
                        scope=("project", "workspace"),
                        requires_confirmation=True,
                        risk_level=RiskLevel.WRITE_SAFE,
                        side_effects=["system_volume"],
                    )
                ]
            ),
            registry,
        )

        described = broker.describe_available_tools(scope="project")

        self.assertEqual(
            described,
            [
                {
                    "name": "system.set_volume",
                    "description": "Ajusta volume",
                    "input_schema": None,
                    "scopes": ["project", "workspace"],
                    "risk_level": "write_safe",
                    "requires_confirmation": True,
                    "side_effects": ["system_volume"],
                }
            ],
        )

    async def test_execute_raises_structured_confirmation_error(self):
        registry = ToolRegistry()

        async def handler(level: int) -> dict:
            return {"volume": level}

        registry.register("system.set_volume", "Ajusta volume", handler)
        broker = ActionBroker(
            CapabilityBroker(
                [
                    Capability(
                        "system.set_volume",
                        enabled=True,
                        requires_confirmation=True,
                        side_effects=["system_volume"],
                    )
                ]
            ),
            registry,
        )

        with self.assertRaises(ConfirmationRequiredError) as context:
            await broker.execute(
                ActionRequest.from_action(SetVolumeAction(level=20), source="llm_tool")
            )

        self.assertEqual(context.exception.tool_name, "system.set_volume")
        self.assertEqual(context.exception.side_effects, ["system_volume"])


if __name__ == "__main__":
    unittest.main()
