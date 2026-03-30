import unittest

from jarvis.core.capability_broker import (
    Capability,
    CapabilityBroker,
    CapabilityDeniedError,
    ConfirmationRequiredError,
    RiskLevel,
)


class CapabilityBrokerTests(unittest.TestCase):
    def test_denies_disabled_capability(self):
        broker = CapabilityBroker([Capability("files.delete", enabled=False)])
        with self.assertRaises(CapabilityDeniedError):
            broker.authorize("files.delete")

    def test_requires_confirmation_for_destructive_tool(self):
        broker = CapabilityBroker(
            [Capability("files.delete", enabled=True, risk_level=RiskLevel.DESTRUCTIVE)]
        )
        with self.assertRaises(ConfirmationRequiredError):
            broker.authorize("files.delete")

        capability = broker.authorize("files.delete", confirmed=True)
        self.assertEqual(capability.tool_name, "files.delete")

    def test_lists_enabled_capabilities_for_scope(self):
        broker = CapabilityBroker(
            [
                Capability("browser.search", enabled=True),
                Capability("files.read", enabled=True, scope="project"),
                Capability("files.move", enabled=False, scope="project"),
            ]
        )

        global_tools = {cap.tool_name for cap in broker.list_enabled()}
        project_tools = {cap.tool_name for cap in broker.list_enabled(scope="project")}

        self.assertEqual(global_tools, {"browser.search"})
        self.assertEqual(project_tools, {"browser.search", "files.read"})


if __name__ == "__main__":
    unittest.main()
