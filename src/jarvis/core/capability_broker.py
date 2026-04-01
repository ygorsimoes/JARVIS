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
    def __init__(self, tool_name: str, reason: str) -> None:
        self.tool_name = tool_name
        self.reason = reason
        super().__init__("capability %s denied: %s" % (tool_name, reason))


class ConfirmationRequiredError(CapabilityError):
    def __init__(
        self,
        tool_name: str,
        scope: str,
        side_effects: Iterable[str] | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.scope = scope
        self.side_effects = list(side_effects or [])
        super().__init__("tool %s requires explicit confirmation" % tool_name)

    def to_payload(self) -> dict[str, object]:
        return {
            "status": "confirmation_required",
            "tool_name": self.tool_name,
            "scope": self.scope,
            "side_effects": list(self.side_effects),
            "message": str(self),
        }


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass
class Capability:
    tool_name: str
    enabled: bool
    scope: str | Iterable[str] = "global"
    risk_level: RiskLevel = RiskLevel.READ_ONLY
    requires_confirmation: bool = False
    side_effects: list[str] = field(default_factory=list)
    audit_log: bool = True

    def __post_init__(self) -> None:
        self.scope = _normalize_scopes(self.scope)

    @property
    def scopes(self) -> tuple[str, ...]:
        return self.scope  # type: ignore[return-value]

    def allows_scope(self, scope: str) -> bool:
        return "global" in self.scopes or scope in self.scopes


@dataclass
class CapabilityAuditEntry:
    tool_name: str
    scope: str
    allowed: bool
    reason: str
    confirmed: bool = False
    side_effects: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)


def _normalize_scopes(scope: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(scope, str):
        values = [scope]
    else:
        values = list(scope)
    normalized = [value.strip() for value in values if value and value.strip()]
    return tuple(dict.fromkeys(normalized or ["global"]))


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
            raise CapabilityDeniedError(tool_name, "not registered")
        return capability

    def authorize(
        self, tool_name: str, scope: str = "global", confirmed: bool = False
    ) -> Capability:
        capability = self.get(tool_name)
        if not capability.enabled:
            self._record(
                tool_name,
                scope,
                False,
                "disabled",
                confirmed=confirmed,
                capability=capability,
            )
            raise CapabilityDeniedError(tool_name, "disabled")
        if not capability.allows_scope(scope):
            self._record(
                tool_name,
                scope,
                False,
                "scope mismatch",
                confirmed=confirmed,
                capability=capability,
            )
            raise CapabilityDeniedError(tool_name, "scope mismatch for %s" % scope)
        if (
            capability.requires_confirmation
            or capability.risk_level == RiskLevel.DESTRUCTIVE
        ):
            if not confirmed:
                self._record(
                    tool_name,
                    scope,
                    False,
                    "confirmation required",
                    confirmed=confirmed,
                    capability=capability,
                )
                raise ConfirmationRequiredError(
                    tool_name=tool_name,
                    scope=scope,
                    side_effects=capability.side_effects,
                )
        self._record(
            tool_name,
            scope,
            True,
            "authorized",
            confirmed=confirmed,
            capability=capability,
        )
        return capability

    def list_enabled(self, scope: str = "global") -> list[Capability]:
        return [
            capability
            for capability in self._capabilities.values()
            if capability.enabled and capability.allows_scope(scope)
        ]

    def enabled_tool_names(self, scope: str = "global") -> set[str]:
        return {capability.tool_name for capability in self.list_enabled(scope=scope)}

    @property
    def audit_log(self) -> list[CapabilityAuditEntry]:
        return list(self._audit_log)

    def _record(
        self,
        tool_name: str,
        scope: str,
        allowed: bool,
        reason: str,
        *,
        confirmed: bool = False,
        capability: Capability | None = None,
    ) -> None:
        if capability is not None and not capability.audit_log:
            return
        self._audit_log.append(
            CapabilityAuditEntry(
                tool_name=tool_name,
                scope=scope,
                allowed=allowed,
                reason=reason,
                confirmed=confirmed,
                side_effects=list(capability.side_effects if capability else []),
            )
        )
