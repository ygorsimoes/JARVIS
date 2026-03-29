from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, List, Optional

from ..models.conversation import Message, Role, Turn
from ..models.memory import Memory
from ..prompts import MEMORY_PREAMBLE


class DialogueManager:
    def __init__(self, system_prompt: str, working_memory_turns: int = 12) -> None:
        self._system_prompt = system_prompt
        self._working_memory: Deque[Turn] = deque(maxlen=working_memory_turns)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def working_memory(self) -> List[Turn]:
        return list(self._working_memory)

    def record_turn(self, role: Role, content: str, metadata: Optional[dict] = None) -> Turn:
        turn = Turn(role=role, content=content, metadata=metadata or {})
        self._working_memory.append(turn)
        return turn

    def compose_messages(self, user_text: str, recalled_memories: Optional[Iterable[Memory]] = None) -> List[Message]:
        messages: List[Message] = [Message(role=Role.SYSTEM, content=self._system_prompt)]

        memory_block = self._format_memory_block(recalled_memories or [])
        if memory_block:
            messages.append(Message(role=Role.SYSTEM, content=memory_block))

        for turn in self._working_memory:
            messages.append(Message(role=turn.role, content=turn.content, metadata=dict(turn.metadata)))

        messages.append(Message(role=Role.USER, content=user_text))
        return messages

    def _format_memory_block(self, recalled_memories: Iterable[Memory]) -> str:
        lines = []
        for memory in recalled_memories:
            lines.append(
                "- [%s/%s %.2f] %s"
                % (memory.category.value, memory.source.value, memory.confidence, memory.content)
            )
        if not lines:
            return ""
        return "%s\n%s" % (MEMORY_PREAMBLE, "\n".join(lines))
