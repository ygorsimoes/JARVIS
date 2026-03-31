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

    def test_supports_multiple_scopes_and_respects_audit_log_flag(self):
        broker = CapabilityBroker(
            [
                Capability(
                    "files.move",
                    enabled=True,
                    scope=("project", "workspace"),
                    audit_log=False,
                )
            ]
        )

        capability = broker.authorize("files.move", scope="project", confirmed=False)

        self.assertEqual(capability.scopes, ("project", "workspace"))
        self.assertEqual(broker.audit_log, [])

        with self.assertRaises(CapabilityDeniedError):
            broker.authorize("files.move", scope="global")

        self.assertEqual(broker.audit_log, [])

    def test_confirmation_error_exposes_payload(self):
        broker = CapabilityBroker(
            [
                Capability(
                    "system.set_volume",
                    enabled=True,
                    requires_confirmation=True,
                    side_effects=["system_volume"],
                )
            ]
        )

        with self.assertRaises(ConfirmationRequiredError) as context:
            broker.authorize("system.set_volume")

        self.assertEqual(
            context.exception.to_payload(),
            {
                "status": "confirmation_required",
                "tool_name": "system.set_volume",
                "scope": "global",
                "side_effects": ["system_volume"],
                "message": "tool system.set_volume requires explicit confirmation",
            },
        )


if __name__ == "__main__":
    unittest.main()
