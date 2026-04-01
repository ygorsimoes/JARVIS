import pytest

from jarvis.core.capability_broker import (
    Capability,
    CapabilityBroker,
    CapabilityDeniedError,
    ConfirmationRequiredError,
    RiskLevel,
)


class TestCapabilityBroker:
    def test_denies_disabled_capability(self):
        broker = CapabilityBroker([Capability("files.delete", enabled=False)])
        with pytest.raises(CapabilityDeniedError):
            broker.authorize("files.delete")

    def test_requires_confirmation_for_destructive_tool(self):
        broker = CapabilityBroker(
            [Capability("files.delete", enabled=True, risk_level=RiskLevel.DESTRUCTIVE)]
        )
        with pytest.raises(ConfirmationRequiredError):
            broker.authorize("files.delete")

        capability = broker.authorize("files.delete", confirmed=True)
        assert capability.tool_name == "files.delete"

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

        assert global_tools == {"browser.search"}
        assert project_tools == {"browser.search", "files.read"}

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

        assert capability.scopes == ("project", "workspace")
        assert broker.audit_log == []

        with pytest.raises(CapabilityDeniedError):
            broker.authorize("files.move", scope="global")

        assert broker.audit_log == []

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

        with pytest.raises(ConfirmationRequiredError) as context:
            broker.authorize("system.set_volume")

        assert context.value.to_payload() == {
            "status": "confirmation_required",
            "tool_name": "system.set_volume",
            "scope": "global",
            "side_effects": ["system_volume"],
            "message": "tool system.set_volume requires explicit confirmation",
        }
