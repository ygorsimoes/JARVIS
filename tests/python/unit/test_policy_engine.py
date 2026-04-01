import pytest

from jarvis.config import JarvisConfig
from jarvis.core.policy_engine import PolicyEngine
from jarvis.models.conversation import RouteDecision, RouteTarget


class _HealthcheckAdapter:
    def __init__(self, available: bool) -> None:
        self.available = available

    async def healthcheck(self) -> bool:
        return self.available


@pytest.mark.asyncio
class TestPolicyEngine:
    async def test_requires_resource_governor_for_native_mlx_backends(self):
        engine = PolicyEngine(JarvisConfig(), enable_native_backends=True)

        assert engine.requires_resource_governor()

    async def test_skips_resource_governor_when_no_native_mlx_backend_is_enabled(self):
        engine = PolicyEngine(
            JarvisConfig(
                llm_hot_path="foundation_models",
                llm_hot_path_fallback="anthropic",
                llm_hot_path_fallback_model="claude-3-7-sonnet-20250219",
                llm_deliberative="fake",
                tts_backend="noop",
            ),
            enable_native_backends=True,
        )

        assert not engine.requires_resource_governor()

    async def test_prefers_foundation_models_when_hot_path_is_available(self):
        route = RouteDecision(target=RouteTarget.HOT_PATH, reason="pedido curto")
        hot_path_adapter = object()
        fallback_adapter = object()
        deliberative_adapter = object()
        engine = PolicyEngine(JarvisConfig(), enable_native_backends=True)

        plan = await engine.select_llm(
            route=route,
            hot_path_adapter=hot_path_adapter,
            hot_path_backend_name="foundation_models",
            deliberative_adapter=deliberative_adapter,
            deliberative_backend_name="mlx_lm",
            fallback_adapter=fallback_adapter,
            fallback_backend_name="mlx_lm_fallback",
        )

        assert plan.adapter is hot_path_adapter
        assert plan.backend_name == "foundation_models"
        assert plan.requested_target == RouteTarget.HOT_PATH
        assert plan.effective_target == RouteTarget.HOT_PATH

    async def test_keeps_hot_path_selection_when_native_backends_are_disabled(self):
        route = RouteDecision(target=RouteTarget.HOT_PATH, reason="pedido curto")
        hot_path_adapter = object()
        deliberative_adapter = object()
        engine = PolicyEngine(JarvisConfig(), enable_native_backends=False)

        plan = await engine.select_llm(
            route=route,
            hot_path_adapter=hot_path_adapter,
            hot_path_backend_name="fake",
            deliberative_adapter=deliberative_adapter,
            deliberative_backend_name="mlx_lm",
            fallback_adapter=object(),
            fallback_backend_name="mlx_lm_fallback",
        )

        assert plan.adapter is hot_path_adapter
        assert plan.backend_name == "fake"
        assert plan.effective_target == RouteTarget.HOT_PATH

    async def test_falls_back_to_local_hot_path_fallback_when_foundation_models_is_unavailable(
        self,
    ):
        route = RouteDecision(target=RouteTarget.HOT_PATH, reason="pedido curto")
        engine = PolicyEngine(JarvisConfig(), enable_native_backends=True)

        plan = await engine.select_llm(
            route=route,
            hot_path_adapter=_HealthcheckAdapter(available=False),
            hot_path_backend_name="foundation_models",
            deliberative_adapter=object(),
            deliberative_backend_name="mlx_lm",
            fallback_adapter=object(),
            fallback_backend_name="mlx_lm_fallback",
        )

        assert plan.backend_name == "mlx_lm_fallback"
        assert plan.effective_target == RouteTarget.HOT_PATH

    async def test_falls_back_to_deliberative_when_hot_path_and_local_fallback_are_unavailable(
        self,
    ):
        route = RouteDecision(target=RouteTarget.HOT_PATH, reason="pedido curto")
        engine = PolicyEngine(JarvisConfig(), enable_native_backends=True)

        plan = await engine.select_llm(
            route=route,
            hot_path_adapter=_HealthcheckAdapter(available=False),
            hot_path_backend_name="foundation_models",
            deliberative_adapter=object(),
            deliberative_backend_name="mlx_lm",
            fallback_adapter=_HealthcheckAdapter(available=False),
            fallback_backend_name="mlx_lm_fallback",
        )

        assert plan.backend_name == "mlx_lm"
        assert plan.effective_target == RouteTarget.DELIBERATIVE

    async def test_falls_back_to_cloud_when_all_local_backends_are_unavailable(self):
        route = RouteDecision(target=RouteTarget.HOT_PATH, reason="pedido curto")
        engine = PolicyEngine(JarvisConfig(), enable_native_backends=True)

        plan = await engine.select_llm(
            route=route,
            hot_path_adapter=_HealthcheckAdapter(available=False),
            hot_path_backend_name="foundation_models",
            deliberative_adapter=_HealthcheckAdapter(available=False),
            deliberative_backend_name="mlx_lm",
            fallback_adapter=object(),
            fallback_backend_name="anthropic",
        )

        assert plan.backend_name == "anthropic"
        assert plan.effective_target == RouteTarget.DELIBERATIVE

    async def test_raises_when_no_backend_is_available(self):
        route = RouteDecision(target=RouteTarget.HOT_PATH, reason="pedido curto")
        engine = PolicyEngine(JarvisConfig(), enable_native_backends=True)

        with pytest.raises(RuntimeError):
            await engine.select_llm(
                route=route,
                hot_path_adapter=_HealthcheckAdapter(available=False),
                hot_path_backend_name="foundation_models",
                deliberative_adapter=_HealthcheckAdapter(available=False),
                deliberative_backend_name="mlx_lm",
                fallback_adapter=_HealthcheckAdapter(available=False),
                fallback_backend_name="mlx_lm_fallback",
            )
