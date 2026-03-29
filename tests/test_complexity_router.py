import unittest

from jarvis.core.complexity_router import ComplexityRouter
from jarvis.models.conversation import RouteTarget


class ComplexityRouterTests(unittest.TestCase):
    def setUp(self):
        self.router = ComplexityRouter()

    def test_routes_time_question_to_direct_tool(self):
        decision = self.router.route("Que horas sao agora?")
        self.assertEqual(decision.target, RouteTarget.DIRECT_TOOL)
        self.assertEqual(decision.tool_name, "system.get_time")

    def test_routes_timer_to_direct_tool(self):
        decision = self.router.route("Define um timer de 20 minutos")
        self.assertEqual(decision.target, RouteTarget.DIRECT_TOOL)
        self.assertEqual(decision.tool_name, "timer.start")

    def test_routes_reasoning_to_deliberative(self):
        decision = self.router.route("Analisa esse erro e explica por que ele acontece")
        self.assertEqual(decision.target, RouteTarget.DELIBERATIVE)

    def test_routes_short_prompt_to_hot_path(self):
        decision = self.router.route("Me lembra do resumo")
        self.assertEqual(decision.target, RouteTarget.HOT_PATH)


if __name__ == "__main__":
    unittest.main()
