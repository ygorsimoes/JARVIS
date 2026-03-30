from __future__ import annotations

from dataclasses import dataclass

from ..config import JarvisConfig
from ..models.conversation import RouteDecision, RouteTarget


@dataclass(frozen=True)
class LLMExecutionPlan:
    adapter: object
    backend_name: str
    requested_target: RouteTarget
    effective_target: RouteTarget
    reason: str


class PolicyEngine:
    def __init__(
        self, config: JarvisConfig, enable_native_backends: bool = False
    ) -> None:
        self._config = config
        self._enable_native_backends = enable_native_backends

    def requires_resource_governor(self) -> bool:
        if not self._enable_native_backends:
            return False
        return any(
            backend.startswith("mlx")
            for backend in (
                self._config.llm_hot_path,
                self._config.llm_deliberative,
                self._config.tts_backend,
            )
        )

    def select_llm(
        self,
        route: RouteDecision,
        hot_path_adapter: object,
        hot_path_backend_name: str,
        deliberative_adapter: object,
        deliberative_backend_name: str,
    ) -> LLMExecutionPlan:
        if route.target == RouteTarget.DIRECT_TOOL:
            raise ValueError("direct tool routes do not need an llm execution plan")

        if (
            route.target == RouteTarget.HOT_PATH
            and self._enable_native_backends
            and hot_path_backend_name == "fake"
            and deliberative_backend_name != "fake"
        ):
            return LLMExecutionPlan(
                adapter=deliberative_adapter,
                backend_name=deliberative_backend_name,
                requested_target=route.target,
                effective_target=RouteTarget.DELIBERATIVE,
                reason="hot path placeholder escalated to local deliberative backend",
            )

        if route.target == RouteTarget.HOT_PATH:
            return LLMExecutionPlan(
                adapter=hot_path_adapter,
                backend_name=hot_path_backend_name,
                requested_target=route.target,
                effective_target=RouteTarget.HOT_PATH,
                reason="hot path backend selected",
            )

        return LLMExecutionPlan(
            adapter=deliberative_adapter,
            backend_name=deliberative_backend_name,
            requested_target=route.target,
            effective_target=RouteTarget.DELIBERATIVE,
            reason="deliberative backend selected",
        )
