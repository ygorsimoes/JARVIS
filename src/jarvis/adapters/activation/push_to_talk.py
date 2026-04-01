from __future__ import annotations

import asyncio
from typing import Optional


class PushToTalkActivationAdapter:
    def __init__(
        self,
        prompt: str = "Pressione ENTER para iniciar um turno de voz...",
        backend: str = "push_to_talk",
        hotkey: str = "<ctrl>+<alt>+space",
        terminal_fallback: bool = True,
    ) -> None:
        self.prompt = prompt
        self.backend = backend
        self.hotkey = hotkey
        self.terminal_fallback = terminal_fallback
        self._listener = None
        self._activation_queue: asyncio.Queue[bool] | None = None
        self._hotkey_loop: asyncio.AbstractEventLoop | None = None
        self._hotkey_disabled_reason: Optional[str] = None

    async def listen(self) -> bool:
        if self._prefers_hotkey():
            activated = await self._listen_with_hotkey()
            if activated is not None:
                return activated
        return await self._listen_with_terminal()

    async def shutdown(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self._activation_queue = None
        self._hotkey_loop = None

    def _prefers_hotkey(self) -> bool:
        return self.backend != "push_to_talk_terminal"

    async def _listen_with_hotkey(self) -> Optional[bool]:
        try:
            self._ensure_hotkey_listener()
        except Exception as exc:
            self._hotkey_disabled_reason = str(exc)
            if not self.terminal_fallback:
                raise RuntimeError(
                    "global push-to-talk hotkey is unavailable: %s" % exc
                ) from exc
            return None

        assert self._activation_queue is not None
        return await self._activation_queue.get()

    async def _listen_with_terminal(self) -> bool:
        await asyncio.to_thread(input, self.prompt)
        return True

    def _ensure_hotkey_listener(self) -> None:
        if self._listener is not None:
            return

        loop = asyncio.get_running_loop()
        self._hotkey_loop = loop
        if self._activation_queue is None:
            self._activation_queue = asyncio.Queue(maxsize=1)

        try:
            from pynput import keyboard
        except ImportError as exc:
            raise RuntimeError(
                "pynput is not installed. Install the 'audio' dependency extra to enable global push-to-talk."
            ) from exc

        hotkey = keyboard.HotKey(
            keyboard.HotKey.parse(self.hotkey),
            lambda: self._notify_hotkey(loop),
        )
        listener_holder = {}

        def for_canonical(callback):
            def handler(key):
                listener = listener_holder["listener"]
                return callback(listener.canonical(key))

            return handler

        listener = keyboard.Listener(
            on_press=for_canonical(hotkey.press),
            on_release=for_canonical(hotkey.release),
        )
        listener_holder["listener"] = listener
        listener.start()
        self._listener = listener

    def _notify_hotkey(self, loop: asyncio.AbstractEventLoop) -> None:
        if loop.is_closed():
            return
        loop.call_soon_threadsafe(self._emit_hotkey_activation)

    def _emit_hotkey_activation(self) -> None:
        if self._activation_queue is None or self._activation_queue.full():
            return
        self._activation_queue.put_nowait(True)
