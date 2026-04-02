from jarvis.core.dialogue_manager import DialogueManager
from jarvis.models.conversation import Role, RouteTarget
from jarvis.models.memory import Memory, MemoryCategory, MemorySource


class TestDialogueManager:
    def test_record_turn_respects_working_memory_window(self):
        manager = DialogueManager(system_prompt="Sistema", working_memory_turns=2)

        manager.record_turn(Role.USER, "primeiro")
        manager.record_turn(Role.ASSISTANT, "segundo")
        manager.record_turn(Role.USER, "terceiro")

        turns = manager.working_memory
        assert [turn.content for turn in turns] == ["segundo", "terceiro"]

    def test_compose_messages_includes_memories_and_turn_metadata(self):
        manager = DialogueManager(system_prompt="Sistema", working_memory_turns=3)
        manager.record_turn(Role.USER, "Oi", metadata={"source": "microphone"})
        manager.record_turn(Role.ASSISTANT, "Como posso ajudar?")
        memories = [
            Memory(
                content="Usuario gosta de respostas objetivas",
                category=MemoryCategory.PROFILE,
                source=MemorySource.INFERRED,
                confidence=0.9,
                recency_weight=0.7,
                scope="global",
            ),
            Memory(
                content="Projeto jarvis usa mlx e sqlite vec para memoria local com busca hibrida e contexto compacto",
                category=MemoryCategory.PROCEDURAL,
                source=MemorySource.EXPLICIT,
                confidence=0.8,
                recency_weight=0.6,
                scope="project:jarvis",
            ),
            Memory(
                content="Esta memoria extra nao deve aparecer no hot path",
                category=MemoryCategory.EPISODIC,
                source=MemorySource.INFERRED,
                confidence=0.5,
                recency_weight=0.5,
                scope="global",
            ),
        ]

        messages = manager.compose_messages(
            "Que horas sao?",
            recalled_memories=memories,
            route_target=RouteTarget.HOT_PATH,
        )

        assert messages[0].role == Role.SYSTEM
        assert messages[0].content == "Sistema"
        assert messages[1].role == Role.SYSTEM
        assert "Usuario gosta de respostas objetivas" in messages[1].content
        assert "project:jarvis" in messages[1].content
        assert "origem inferred" in messages[1].content
        assert "nao deve aparecer" not in messages[1].content
        assert messages[2].metadata == {"source": "microphone"}
        assert messages[-1].role == Role.USER
        assert messages[-1].content == "Que horas sao?"
