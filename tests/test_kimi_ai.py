"""Unit tests for Kimi structured visual planning."""

import asyncio
import json

import httpx
import pytest

from app.clients.kimi_ai import KimiVisualPlanningClient
from app.errors import StructuredAIProviderError
from app.provider_diagnostics import find_structured_ai_provider_diagnostic


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
    assert captured["max_completion_tokens"] == 32_768
    assert "max_tokens" not in captured
    assert captured["reasoning_effort"] == "low"


def test_kimi_k2_6_omits_unsupported_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    async_client = httpx.AsyncClient

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
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

    asyncio.run(
        KimiVisualPlanningClient(
            api_key="private-kimi-key",
            model="kimi-k2.6",
        ).generate("story")
    )

    assert captured["model"] == "kimi-k2.6"
    assert "reasoning_effort" not in captured


def test_kimi_preflight_checks_models_without_paid_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    async_client = httpx.AsyncClient

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        return httpx.Response(
            200,
            json={"data": [{"id": "kimi-k2.6"}]},
            request=request,
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    asyncio.run(
        KimiVisualPlanningClient(
            api_key="private-kimi-key",
            model="kimi-k2.6",
        ).preflight()
    )

    assert captured == {"method": "GET", "path": "/v1/models"}


def test_kimi_preflight_failure_preserves_payment_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async_client = httpx.AsyncClient

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            402,
            request=request,
            json={"error": {"message": "insufficient balance"}},
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    with pytest.raises(StructuredAIProviderError, match="preflight failed") as error:
        asyncio.run(
            KimiVisualPlanningClient(
                api_key="private-kimi-key",
                model="kimi-k2.6",
            ).preflight()
        )

    diagnostic = find_structured_ai_provider_diagnostic(error.value)
    assert diagnostic is not None
    assert diagnostic.category == "PLANNING_PAYMENT_REQUIRED"
    assert diagnostic.operation == "visual_planning_preflight"


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
            KimiVisualPlanningClient(
                api_key="private-kimi-key",
                max_attempts=1,
            ).generate("story")
        )

    assert "private-kimi-key" not in str(raised.value)


def test_kimi_planning_client_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    with pytest.raises(StructuredAIProviderError, match="MOONSHOT_API_KEY"):
        asyncio.run(KimiVisualPlanningClient().generate("story"))


def test_kimi_timeout_is_never_retried_even_with_legacy_retry_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async_client = httpx.AsyncClient
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("slow", request=request)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    with pytest.raises(StructuredAIProviderError):
        asyncio.run(
            KimiVisualPlanningClient(
                api_key="key", max_attempts=3, retry_base_delay=0
            ).generate("story")
        )
    assert calls == 1
