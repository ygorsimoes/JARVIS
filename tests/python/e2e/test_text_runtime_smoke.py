from __future__ import annotations

import pytest

from jarvis.config import JarvisConfig
from jarvis.models.conversation import RouteTarget
from jarvis.runtime import JarvisRuntime


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_text_runtime_smoke_route_and_response() -> None:
    runtime = JarvisRuntime.from_config(
        JarvisConfig(allowed_file_roots=["/tmp"]),
        enable_native_backends=False,
    )

    try:
        response = await runtime.respond_text("Que horas sao agora?")
    finally:
        await runtime.shutdown()

    assert response.route.target == RouteTarget.DIRECT_TOOL
    assert "Agora sao" in response.full_text
