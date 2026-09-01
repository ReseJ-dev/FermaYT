"""Tests for the ElevenLabs HTTP text-to-speech client."""

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from app.clients.tts_api import ElevenLabsTTSApiClient
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


def test_elevenlabs_generate_returns_audio_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == (
            "https://api.elevenlabs.io/v1/text-to-speech/voice-123"
            "?output_format=mp3_44100_128"
        )
        assert request.headers["xi-api-key"] == "explicit-secret"
        assert request.headers["Accept"] == "audio/mpeg"
        assert json.loads(request.content) == {
            "text": "Привет, мир!",
            "model_id": "eleven_multilingual_v2",
        }
        return httpx.Response(
            200,
            content=b"mp3-audio",
            headers={"content-type": "audio/mpeg"},
        )

    install_mock_transport(monkeypatch, handler)
    client = ElevenLabsTTSApiClient(
        api_key="explicit-secret",
        voice="voice-123",
    )

    result = asyncio.run(client.generate("  Привет, мир!  "))

    assert result == b"mp3-audio"


def test_elevenlabs_uses_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["xi-api-key"] == "environment-secret"
        return httpx.Response(
            200,
            content=b"audio",
            headers={"content-type": "application/octet-stream"},
        )

    monkeypatch.setenv("ELEVENLABS_API_KEY", "environment-secret")
    install_mock_transport(monkeypatch, handler)

    assert asyncio.run(ElevenLabsTTSApiClient().generate("Hello")) == b"audio"


@pytest.mark.parametrize("text", ["", "   "])
def test_elevenlabs_rejects_blank_text(text: str) -> None:
    with pytest.raises(TTSGenerationError, match="Invalid TTS text"):
        asyncio.run(ElevenLabsTTSApiClient(api_key="key").generate(text))


def test_elevenlabs_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    with pytest.raises(TTSGenerationError, match="ELEVENLABS_API_KEY"):
        asyncio.run(ElevenLabsTTSApiClient().generate("Hello"))


def test_elevenlabs_handles_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    install_mock_transport(monkeypatch, handler)

    with pytest.raises(TTSGenerationError, match="timed out"):
        asyncio.run(
            ElevenLabsTTSApiClient(api_key="key").generate("Hello")
        )


@pytest.mark.parametrize("status_code", [400, 401, 429, 500])
def test_elevenlabs_handles_http_errors(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    install_mock_transport(
        monkeypatch,
        lambda request: httpx.Response(
            status_code,
            request=request,
            json={"detail": "provider failure"},
        ),
    )

    with pytest.raises(TTSGenerationError, match=f"HTTP {status_code}"):
        asyncio.run(
            ElevenLabsTTSApiClient(api_key="key").generate("Hello")
        )


@pytest.mark.parametrize(
    ("content", "content_type", "message"),
    [
        (b"", "audio/mpeg", "empty audio"),
        (b'{"error":"bad"}', "application/json", "unexpected response"),
    ],
)
def test_elevenlabs_rejects_invalid_success_response(
    monkeypatch: pytest.MonkeyPatch,
    content: bytes,
    content_type: str,
    message: str,
) -> None:
    install_mock_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            request=request,
            content=content,
            headers={"content-type": content_type},
        ),
    )

    with pytest.raises(TTSGenerationError, match=message):
        asyncio.run(
            ElevenLabsTTSApiClient(api_key="key").generate("Hello")
        )
