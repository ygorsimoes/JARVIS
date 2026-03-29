from __future__ import annotations

import asyncio


class PushToTalkActivationAdapter:
    def __init__(self, prompt: str = "Pressione ENTER para iniciar um turno de voz...") -> None:
        self.prompt = prompt

    async def listen(self) -> bool:
        await asyncio.to_thread(input, self.prompt)
        return True
