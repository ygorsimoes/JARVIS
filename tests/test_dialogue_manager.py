import unittest

from jarvis.core.dialogue_manager import DialogueManager
from jarvis.models.conversation import Role
from jarvis.models.memory import Memory, MemoryCategory, MemorySource


class DialogueManagerTests(unittest.TestCase):
    def test_record_turn_respects_working_memory_window(self):
        manager = DialogueManager(system_prompt="Sistema", working_memory_turns=2)

        manager.record_turn(Role.USER, "primeiro")
        manager.record_turn(Role.ASSISTANT, "segundo")
        manager.record_turn(Role.USER, "terceiro")

        turns = manager.working_memory
        self.assertEqual([turn.content for turn in turns], ["segundo", "terceiro"])

    def test_compose_messages_includes_memories_and_turn_metadata(self):
        manager = DialogueManager(system_prompt="Sistema", working_memory_turns=3)
        manager.record_turn(Role.USER, "Oi", metadata={"source": "microphone"})
        manager.record_turn(Role.ASSISTANT, "Como posso ajudar?")
        memory = Memory(
            content="Usuario gosta de respostas objetivas",
            category=MemoryCategory.PROFILE,
            source=MemorySource.INFERRED,
            confidence=0.9,
            recency_weight=0.7,
            scope="global",
        )

        messages = manager.compose_messages(
            "Que horas sao?", recalled_memories=[memory]
        )

        self.assertEqual(messages[0].role, Role.SYSTEM)
        self.assertEqual(messages[0].content, "Sistema")
        self.assertEqual(messages[1].role, Role.SYSTEM)
        self.assertIn("Usuario gosta de respostas objetivas", messages[1].content)
        self.assertEqual(messages[2].metadata, {"source": "microphone"})
        self.assertEqual(messages[-1].role, Role.USER)
        self.assertEqual(messages[-1].content, "Que horas sao?")


if __name__ == "__main__":
    unittest.main()
