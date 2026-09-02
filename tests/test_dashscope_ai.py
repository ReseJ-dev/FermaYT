"""Unit tests for production structured DashScope clients without HTTP traffic."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from app.clients.dashscope_ai import (
    DashScopeVisualPlanningClient,
    DashScopeVisualQAClient,
)
from app.errors import StructuredAIProviderError


def test_planning_client_sends_json_mode_and_returns_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    async_client = httpx.AsyncClient

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer private-key"
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
        DashScopeVisualPlanningClient(api_key="private-key").generate("whole story")
    )
    assert result == '{"visual_beats": []}'
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["model"] == "qwen-plus"


def test_visual_qa_client_embeds_candidate_and_references(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    captured: dict[str, object] = {}
    async_client = httpx.AsyncClient

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"result":"PASS"}'}}]},
            request=request,
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    result = asyncio.run(
        DashScopeVisualQAClient(api_key="private-key").evaluate(
            "judge frame", (str(first), str(second))
        )
    )
    content = captured["messages"][0]["content"]  # type: ignore[index]
    assert result == '{"result":"PASS"}'
    assert [item["type"] for item in content] == ["image_url", "image_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize("mode", ["timeout", "http", "invalid-json", "shape"])
def test_structured_provider_fails_safely(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    async_client = httpx.AsyncClient

    async def handler(request: httpx.Request) -> httpx.Response:
        if mode == "timeout":
            raise httpx.ReadTimeout("slow", request=request)
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
    with pytest.raises(StructuredAIProviderError):
        asyncio.run(
            DashScopeVisualPlanningClient(api_key="never-exposed").generate("story")
        )
