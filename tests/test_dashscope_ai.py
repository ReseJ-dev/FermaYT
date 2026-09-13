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
from app.provider_diagnostics import find_structured_ai_provider_diagnostic


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
    assert captured["max_tokens"] == 32_768


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
            DashScopeVisualPlanningClient(
                api_key="never-exposed",
                max_attempts=1,
            ).generate("story")
        )


@pytest.mark.parametrize(
    ("status", "category"),
    [
        (400, "PLANNING_BAD_REQUEST"),
        (401, "PLANNING_AUTH_ERROR"),
        (403, "PLANNING_AUTH_ERROR"),
        (404, "PLANNING_NOT_FOUND"),
        (429, "PLANNING_RATE_LIMIT"),
        (500, "PLANNING_PROVIDER_5XX"),
    ],
)
def test_planning_http_failures_have_safe_stable_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    category: str,
) -> None:
    async_client = httpx.AsyncClient

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            request=request,
            headers={"x-request-id": "request-123"},
            json={
                "code": "ProviderCode",
                "message": "safe failure; Authorization: Bearer private-token",
            },
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    with pytest.raises(StructuredAIProviderError) as error:
        asyncio.run(
            DashScopeVisualPlanningClient(
                api_key="never-exposed",
                max_attempts=1,
            ).generate("story")
        )

    diagnostic = find_structured_ai_provider_diagnostic(error.value)
    assert diagnostic is not None
    assert diagnostic.category == category
    assert diagnostic.http_status == status
    assert diagnostic.request_id == "request-123"
    assert "ProviderCode" in (diagnostic.provider_error or "")
    assert "private-token" not in (diagnostic.provider_error or "")


def test_planning_timeout_records_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    async_client = httpx.AsyncClient

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    with pytest.raises(StructuredAIProviderError) as error:
        asyncio.run(
            DashScopeVisualPlanningClient(
                api_key="key", timeout=42, max_attempts=1
            ).generate("story")
        )
    diagnostic = find_structured_ai_provider_diagnostic(error.value)
    assert diagnostic is not None
    assert diagnostic.category == "PLANNING_TIMEOUT"
    assert diagnostic.timeout_seconds == 42
    assert diagnostic.retry_exhausted is True


def test_planning_retries_transient_http_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async_client = httpx.AsyncClient
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(429, request=request, json={"message": "busy"})
        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"visual_beats":[]}'},
                    }
                ]
            },
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    result = asyncio.run(
        DashScopeVisualPlanningClient(
            api_key="key", max_attempts=3, retry_base_delay=0
        ).generate("story")
    )
    assert result == '{"visual_beats":[]}'
    assert calls == 3


def test_planning_does_not_retry_auth_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async_client = httpx.AsyncClient
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, request=request)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    with pytest.raises(StructuredAIProviderError):
        asyncio.run(
            DashScopeVisualPlanningClient(
                api_key="key", max_attempts=3, retry_base_delay=0
            ).generate("story")
        )
    assert calls == 1


@pytest.mark.parametrize(
    ("response", "category"),
    [
        (
            {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]},
            "PLANNING_EMPTY_RESPONSE",
        ),
        (
            {"choices": [{"finish_reason": "length", "message": {"content": '{"a":'}}]},
            "PLANNING_TRUNCATED_OUTPUT",
        ),
        (
            {"choices": [{"finish_reason": "stop", "message": {"content": "not"}}]},
            "PLANNING_INVALID_JSON",
        ),
        ({"choices": []}, "PLANNING_STRUCTURED_JSON_INCOMPATIBILITY"),
    ],
)
def test_planning_response_failures_are_distinguished(
    monkeypatch: pytest.MonkeyPatch,
    response: dict[str, object],
    category: str,
) -> None:
    async_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: async_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, request=request, json=response)
            ),
            **kwargs,
        ),
    )
    with pytest.raises(StructuredAIProviderError) as error:
        asyncio.run(
            DashScopeVisualPlanningClient(api_key="key").generate("story")
        )
    diagnostic = find_structured_ai_provider_diagnostic(error.value)
    assert diagnostic is not None
    assert diagnostic.category == category


def test_planning_malformed_http_json_is_distinguished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: async_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, request=request, content=b"not-json"
                )
            ),
            **kwargs,
        ),
    )
    with pytest.raises(StructuredAIProviderError) as error:
        asyncio.run(
            DashScopeVisualPlanningClient(api_key="key").generate("story")
        )
    diagnostic = find_structured_ai_provider_diagnostic(error.value)
    assert diagnostic is not None
    assert diagnostic.category == "PLANNING_INVALID_JSON"
    assert diagnostic.response_length == len(b"not-json")
    assert diagnostic.response_preview == "not-json"
