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


if __name__ == "__main__":
    unittest.main()
