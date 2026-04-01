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


@dataclass(frozen=True)
class _BackendCandidate:
    adapter: object
    backend_name: str
    target: RouteTarget


class PolicyEngine:
    def __init__(
        self, config: JarvisConfig, enable_native_backends: bool = True
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

    async def select_llm(
        self,
        route: RouteDecision,
        hot_path_adapter: object,
        hot_path_backend_name: str,
        deliberative_adapter: object,
        deliberative_backend_name: str,
        fallback_adapter: object,
        fallback_backend_name: str,
    ) -> LLMExecutionPlan:
        if route.target == RouteTarget.DIRECT_TOOL:
            raise ValueError("direct tool routes do not need an llm execution plan")

        candidates = self._ordered_candidates(
            route=route,
            hot_path_adapter=hot_path_adapter,
            hot_path_backend_name=hot_path_backend_name,
            deliberative_adapter=deliberative_adapter,
            deliberative_backend_name=deliberative_backend_name,
            fallback_adapter=fallback_adapter,
            fallback_backend_name=fallback_backend_name,
        )

        for i, candidate in enumerate(candidates):
            if await self._candidate_is_available(candidate):
                return LLMExecutionPlan(
                    adapter=candidate.adapter,
                    backend_name=candidate.backend_name,
                    requested_target=route.target,
                    effective_target=candidate.target,
                    reason=self._selection_reason(
                        route.target, candidate, used_fallback=(i > 0)
                    ),
                )

        raise RuntimeError(
            "nenhum backend llm local disponivel para %s" % route.target.value
        )

    def _ordered_candidates(
        self,
        route: RouteDecision,
        hot_path_adapter: object,
        hot_path_backend_name: str,
        deliberative_adapter: object,
        deliberative_backend_name: str,
        fallback_adapter: object,
        fallback_backend_name: str,
    ) -> list[_BackendCandidate]:
        hot_candidate = _BackendCandidate(
            adapter=hot_path_adapter,
            backend_name=hot_path_backend_name,
            target=RouteTarget.HOT_PATH,
        )
        deliberative_candidate = _BackendCandidate(
            adapter=deliberative_adapter,
            backend_name=deliberative_backend_name,
            target=RouteTarget.DELIBERATIVE,
        )
        cloud_candidate = _BackendCandidate(
            adapter=fallback_adapter,
            backend_name=fallback_backend_name,
            target=RouteTarget.DELIBERATIVE,
        )

        if route.target == RouteTarget.HOT_PATH:
            candidates = [hot_candidate, deliberative_candidate, cloud_candidate]
        else:
            candidates = [deliberative_candidate, hot_candidate, cloud_candidate]

        if not self._enable_native_backends:
            return [candidates[0]]

        # Mover backends fake para o fim
        real_candidates = [c for c in candidates if c.backend_name != "fake"]
        fake_candidates = [c for c in candidates if c.backend_name == "fake"]
        return real_candidates + fake_candidates

    async def _candidate_is_available(self, candidate: _BackendCandidate) -> bool:
        healthcheck = getattr(candidate.adapter, "healthcheck", None)
        if healthcheck is None:
            return True
        return bool(await healthcheck())

    @staticmethod
    def _selection_reason(
        requested_target: RouteTarget,
        candidate: _BackendCandidate,
        *,
        used_fallback: bool,
    ) -> str:
        if not used_fallback and requested_target == candidate.target:
            return "%s backend selected" % candidate.target.value
        if candidate.backend_name == "fake":
            return (
                "fallback to fake backend because no real local backend was available"
            )
        return "%s route fell back to %s backend" % (
            requested_target.value,
            candidate.backend_name,
        )
