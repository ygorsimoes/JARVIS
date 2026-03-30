import unittest

from jarvis.config import JarvisConfig
from jarvis.core.policy_engine import PolicyEngine
from jarvis.models.conversation import RouteDecision, RouteTarget


class _HealthcheckAdapter:
    def __init__(self, available: bool) -> None:
        self.available = available

    async def healthcheck(self) -> bool:
        return self.available


class PolicyEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_requires_resource_governor_for_native_mlx_backends(self):
        engine = PolicyEngine(JarvisConfig(), enable_native_backends=True)

        self.assertTrue(engine.requires_resource_governor())

    async def test_skips_resource_governor_when_no_native_mlx_backend_is_enabled(self):
        engine = PolicyEngine(
            JarvisConfig(
                llm_hot_path="foundation_models",
                llm_deliberative="fake",
                tts_backend="noop",
            ),
            enable_native_backends=True,
        )

        self.assertFalse(engine.requires_resource_governor())

    async def test_escalates_fake_hot_path_to_local_deliberative_backend(self):
        route = RouteDecision(target=RouteTarget.HOT_PATH, reason="pedido curto")
        hot_path_adapter = object()
        deliberative_adapter = object()
        engine = PolicyEngine(JarvisConfig(), enable_native_backends=True)

        plan = await engine.select_llm(
            route=route,
            hot_path_adapter=hot_path_adapter,
            hot_path_backend_name="fake",
            deliberative_adapter=deliberative_adapter,
            deliberative_backend_name="mlx_lm",
            fallback_adapter=object(),
            fallback_backend_name="anthropic",
        )

        self.assertIs(plan.adapter, deliberative_adapter)
        self.assertEqual(plan.backend_name, "mlx_lm")
        self.assertEqual(plan.requested_target, RouteTarget.HOT_PATH)
        self.assertEqual(plan.effective_target, RouteTarget.DELIBERATIVE)

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
            fallback_backend_name="anthropic",
        )

        self.assertIs(plan.adapter, hot_path_adapter)
        self.assertEqual(plan.backend_name, "fake")
        self.assertEqual(plan.effective_target, RouteTarget.HOT_PATH)

    async def test_falls_back_to_deliberative_when_hot_path_healthcheck_fails(self):
        route = RouteDecision(target=RouteTarget.HOT_PATH, reason="pedido curto")
        engine = PolicyEngine(JarvisConfig(), enable_native_backends=True)

        plan = await engine.select_llm(
            route=route,
            hot_path_adapter=_HealthcheckAdapter(available=False),
            hot_path_backend_name="foundation_models",
            deliberative_adapter=object(),
            deliberative_backend_name="mlx_lm",
            fallback_adapter=object(),
            fallback_backend_name="anthropic",
        )

        self.assertEqual(plan.backend_name, "mlx_lm")
        self.assertEqual(plan.effective_target, RouteTarget.DELIBERATIVE)

    async def test_falls_back_to_cloud_when_both_local_healthchecks_fail(self):
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

        self.assertEqual(plan.backend_name, "anthropic")
        self.assertEqual(plan.effective_target, RouteTarget.DELIBERATIVE)

    async def test_raises_when_no_local_backend_is_available(self):
        route = RouteDecision(target=RouteTarget.HOT_PATH, reason="pedido curto")
        engine = PolicyEngine(JarvisConfig(), enable_native_backends=True)

        with self.assertRaises(RuntimeError):
            await engine.select_llm(
                route=route,
                hot_path_adapter=_HealthcheckAdapter(available=False),
                hot_path_backend_name="foundation_models",
                deliberative_adapter=_HealthcheckAdapter(available=False),
                deliberative_backend_name="mlx_lm",
                fallback_adapter=_HealthcheckAdapter(available=False),
                fallback_backend_name="anthropic",
            )


if __name__ == "__main__":
    unittest.main()
