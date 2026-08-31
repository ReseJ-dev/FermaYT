"""Tests for text-to-speech API clients."""

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from app.clients.tts_api import QwenTTSApiClient, TTSApiClient
from app.errors import TTSGenerationError


ResponseHandler = Callable[[httpx.Request], httpx.Response]


def install_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: ResponseHandler,
) -> None:
    real_async_client = httpx.AsyncClient

    def create_mock_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", create_mock_client)


def configure_client(
    monkeypatch: pytest.MonkeyPatch,
    handler: ResponseHandler,
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-key")
    install_mock_transport(monkeypatch, handler)


def run_generate(text: str = "Welcome to the video") -> str:
    return asyncio.run(TTSApiClient().generate(text))


def test_generate_returns_audio_url(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == TTSApiClient.API_URL
        assert request.headers["Authorization"] == "Bearer secret-key"
        assert request.headers["Content-Type"] == "application/json"
        assert "X-DashScope-SSE" not in request.headers
        assert json.loads(request.content) == {
            "model": "qwen3-tts-flash",
            "input": {
                "text": "Welcome to the video",
                "voice": "Cherry",
                "language_type": "Russian",
            },
        }
        return httpx.Response(
            200,
            json={
                "code": "",
                "message": "",
                "output": {
                    "audio": {
                        "data": "",
                        "url": "https://example.com/audio.wav",
                    }
                },
            },
        )

    configure_client(monkeypatch, handler)

    assert run_generate() == "https://example.com/audio.wav"


def test_generate_uses_explicit_constructor_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://custom.example.com/tts"
        assert request.headers["Authorization"] == "Bearer explicit-key"
        assert json.loads(request.content) == {
            "model": "custom-model",
            "input": {
                "text": "Hello",
                "voice": "CustomVoice",
                "language_type": "English",
            },
        }
        return httpx.Response(
            200,
            json={
                "output": {
                    "audio": {"url": "https://example.com/custom.wav"}
                }
            },
        )

    monkeypatch.setenv("DASHSCOPE_API_KEY", "environment-key")
    install_mock_transport(monkeypatch, handler)
    client = QwenTTSApiClient(
        api_key="explicit-key",
        endpoint="https://custom.example.com/tts",
        model="custom-model",
        voice="CustomVoice",
        language="English",
    )

    result = asyncio.run(client.generate("Hello"))

    assert result == "https://example.com/custom.wav"


@pytest.mark.parametrize("text", ["", "   "])
def test_generate_rejects_blank_text(text: str) -> None:
    with pytest.raises(TTSGenerationError, match="Invalid TTS text"):
        run_generate(text)


def test_generate_rejects_missing_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(TTSGenerationError, match="DASHSCOPE_API_KEY"):
        run_generate()


def test_generate_rejects_explicit_empty_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "environment-key")
    client = QwenTTSApiClient(api_key="")

    with pytest.raises(TTSGenerationError, match="DASHSCOPE_API_KEY"):
        asyncio.run(client.generate("Welcome"))


def test_generate_handles_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    configure_client(monkeypatch, handler)

    with pytest.raises(TTSGenerationError, match="timed out"):
        run_generate()


@pytest.mark.parametrize("status_code", [400, 401, 429, 500])
def test_generate_handles_http_errors(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"message": "failure"})

    configure_client(monkeypatch, handler)

    with pytest.raises(TTSGenerationError, match=f"HTTP {status_code}"):
        run_generate()


def test_generate_handles_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    configure_client(monkeypatch, handler)

    with pytest.raises(TTSGenerationError, match="invalid JSON"):
        run_generate()


def test_generate_handles_api_error_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"code": "InvalidParameter", "message": "Bad text"},
        )

    configure_client(monkeypatch, handler)

    with pytest.raises(
        TTSGenerationError,
        match="InvalidParameter: Bad text",
    ):
        run_generate()


def test_generate_requires_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": "", "message": ""})

    configure_client(monkeypatch, handler)

    with pytest.raises(TTSGenerationError, match="output"):
        run_generate()


def test_generate_requires_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"output": {}})

    configure_client(monkeypatch, handler)

    with pytest.raises(TTSGenerationError, match="audio"):
        run_generate()


@pytest.mark.parametrize(
    "audio",
    [
        {},
        {"url": ""},
        {"url": "   "},
    ],
)
def test_generate_requires_non_empty_audio_url(
    monkeypatch: pytest.MonkeyPatch,
    audio: dict[str, str],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"output": {"audio": audio}})

    configure_client(monkeypatch, handler)

    with pytest.raises(TTSGenerationError, match="audio URL"):
        run_generate()
