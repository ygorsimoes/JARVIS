import unittest

from jarvis.config import JarvisConfig
from jarvis.core.policy_engine import PolicyEngine
from jarvis.models.conversation import RouteDecision, RouteTarget


class PolicyEngineTests(unittest.TestCase):
    def test_requires_resource_governor_for_native_mlx_backends(self):
        engine = PolicyEngine(JarvisConfig(), enable_native_backends=True)

        self.assertTrue(engine.requires_resource_governor())

    def test_skips_resource_governor_when_no_native_mlx_backend_is_enabled(self):
        engine = PolicyEngine(
            JarvisConfig(
                llm_hot_path="foundation_models",
                llm_deliberative="fake",
                tts_backend="noop",
            ),
            enable_native_backends=True,
        )

        self.assertFalse(engine.requires_resource_governor())

    def test_escalates_fake_hot_path_to_local_deliberative_backend(self):
        route = RouteDecision(target=RouteTarget.HOT_PATH, reason="pedido curto")
        hot_path_adapter = object()
        deliberative_adapter = object()
        engine = PolicyEngine(JarvisConfig(), enable_native_backends=True)

        plan = engine.select_llm(
            route=route,
            hot_path_adapter=hot_path_adapter,
            hot_path_backend_name="fake",
            deliberative_adapter=deliberative_adapter,
            deliberative_backend_name="mlx_lm",
        )

        self.assertIs(plan.adapter, deliberative_adapter)
        self.assertEqual(plan.backend_name, "mlx_lm")
        self.assertEqual(plan.requested_target, RouteTarget.HOT_PATH)
        self.assertEqual(plan.effective_target, RouteTarget.DELIBERATIVE)

    def test_keeps_hot_path_selection_when_native_backends_are_disabled(self):
        route = RouteDecision(target=RouteTarget.HOT_PATH, reason="pedido curto")
        hot_path_adapter = object()
        deliberative_adapter = object()
        engine = PolicyEngine(JarvisConfig(), enable_native_backends=False)

        plan = engine.select_llm(
            route=route,
            hot_path_adapter=hot_path_adapter,
            hot_path_backend_name="fake",
            deliberative_adapter=deliberative_adapter,
            deliberative_backend_name="mlx_lm",
        )

        self.assertIs(plan.adapter, hot_path_adapter)
        self.assertEqual(plan.backend_name, "fake")
        self.assertEqual(plan.effective_target, RouteTarget.HOT_PATH)


if __name__ == "__main__":
    unittest.main()
