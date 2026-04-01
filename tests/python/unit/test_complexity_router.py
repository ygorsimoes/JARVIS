from jarvis.core.complexity_router import ComplexityRouter
from jarvis.models.conversation import RouteTarget


class TestComplexityRouter:
    def setup_method(self):
        self.router = ComplexityRouter()

    def test_routes_time_question_to_direct_tool(self):
        decision = self.router.route("Que horas sao agora?")
        assert decision.target == RouteTarget.DIRECT_TOOL
        assert decision.tool_name == "system.get_time"

    def test_routes_timer_to_direct_tool(self):
        decision = self.router.route("Define um timer de 20 minutos")
        assert decision.target == RouteTarget.DIRECT_TOOL
        assert decision.tool_name == "timer.start"

    def test_routes_reasoning_to_deliberative(self):
        decision = self.router.route("Analisa esse erro e explica por que ele acontece")
        assert decision.target == RouteTarget.DELIBERATIVE

    def test_routes_short_prompt_to_hot_path(self):
        decision = self.router.route("Me lembra do resumo")
        assert decision.target == RouteTarget.HOT_PATH

    def test_routes_multiple_direct_actions_to_deliberative(self):
        decision = self.router.route("Abre o Safari e pesquise arquitetura hexagonal")
        assert decision.target == RouteTarget.DELIBERATIVE

    def test_routes_memory_heavy_prompt_to_deliberative(self):
        decision = self.router.route("Me lembra do contexto", recalled_memories=2)
        assert decision.target == RouteTarget.DELIBERATIVE

    def test_routes_tool_chain_depth_to_deliberative(self):
        decision = self.router.route("Abre o Safari", tool_chain_depth=2)
        assert decision.target == RouteTarget.DELIBERATIVE
