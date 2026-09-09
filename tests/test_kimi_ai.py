"""Unit tests for Kimi structured visual planning."""

import asyncio
import json

import httpx
import pytest

from app.clients.kimi_ai import KimiVisualPlanningClient
from app.errors import StructuredAIProviderError


def test_kimi_planning_client_uses_json_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    async_client = httpx.AsyncClient

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        assert request.url == KimiVisualPlanningClient.ENDPOINT
        assert request.headers["Authorization"] == "Bearer private-kimi-key"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"visual_beats": []}'}}]},
            request=request,
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    result = asyncio.run(
        KimiVisualPlanningClient(api_key="private-kimi-key").generate("story")
    )

    assert result == '{"visual_beats": []}'
    assert captured["model"] == "kimi-k3"
    assert captured["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("mode", ["timeout", "http", "invalid-json", "shape"])
def test_kimi_planning_client_fails_safely(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    async_client = httpx.AsyncClient

    async def handler(request: httpx.Request) -> httpx.Response:
        if mode == "timeout":
            raise httpx.ReadTimeout("slow private-kimi-key", request=request)
        if mode == "http":
            return httpx.Response(401, request=request)
        if mode == "invalid-json":
            return httpx.Response(200, content=b"not-json", request=request)
        return httpx.Response(200, json={"choices": []}, request=request)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    with pytest.raises(StructuredAIProviderError) as raised:
        asyncio.run(
            KimiVisualPlanningClient(api_key="private-kimi-key").generate("story")
        )

    assert "private-kimi-key" not in str(raised.value)


def test_kimi_planning_client_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    with pytest.raises(StructuredAIProviderError, match="MOONSHOT_API_KEY"):
        asyncio.run(KimiVisualPlanningClient().generate("story"))
