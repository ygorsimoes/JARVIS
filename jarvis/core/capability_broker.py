from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Iterable, Optional


class RiskLevel(str, Enum):
    READ_ONLY = "read_only"
    WRITE_SAFE = "write_safe"
    DESTRUCTIVE = "destructive"


class CapabilityError(RuntimeError):
    pass


class CapabilityDeniedError(CapabilityError):
    pass


class ConfirmationRequiredError(CapabilityError):
    pass


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass
class Capability:
    tool_name: str
    enabled: bool
    scope: str = "global"
    risk_level: RiskLevel = RiskLevel.READ_ONLY
    requires_confirmation: bool = False
    side_effects: list[str] = field(default_factory=list)
    audit_log: bool = True


@dataclass
class CapabilityAuditEntry:
    tool_name: str
    scope: str
    allowed: bool
    reason: str
    created_at: datetime = field(default_factory=utc_now)


class CapabilityBroker:
    def __init__(self, capabilities: Optional[Iterable[Capability]] = None) -> None:
        self._capabilities: Dict[str, Capability] = {}
        self._audit_log: list[CapabilityAuditEntry] = []
        for capability in capabilities or ():
            self.register(capability)

    def register(self, capability: Capability) -> None:
        self._capabilities[capability.tool_name] = capability

    def get(self, tool_name: str) -> Capability:
        capability = self._capabilities.get(tool_name)
        if capability is None:
            self._record(tool_name, "global", False, "not registered")
            raise CapabilityDeniedError("capability %s is not registered" % tool_name)
        return capability

    def authorize(self, tool_name: str, scope: str = "global", confirmed: bool = False) -> Capability:
        capability = self.get(tool_name)
        if not capability.enabled:
            self._record(tool_name, scope, False, "disabled")
            raise CapabilityDeniedError("capability %s is disabled" % tool_name)
        if capability.scope not in {"global", scope}:
            self._record(tool_name, scope, False, "scope mismatch")
            raise CapabilityDeniedError("capability %s is not enabled for scope %s" % (tool_name, scope))
        if capability.requires_confirmation or capability.risk_level == RiskLevel.DESTRUCTIVE:
            if not confirmed:
                self._record(tool_name, scope, False, "confirmation required")
                raise ConfirmationRequiredError("tool %s requires explicit confirmation" % tool_name)
        self._record(tool_name, scope, True, "authorized")
        return capability

    def list_enabled(self) -> list[Capability]:
        return [capability for capability in self._capabilities.values() if capability.enabled]

    @property
    def audit_log(self) -> list[CapabilityAuditEntry]:
        return list(self._audit_log)

    def _record(self, tool_name: str, scope: str, allowed: bool, reason: str) -> None:
        self._audit_log.append(
            CapabilityAuditEntry(
                tool_name=tool_name,
                scope=scope,
                allowed=allowed,
                reason=reason,
            )
        )
