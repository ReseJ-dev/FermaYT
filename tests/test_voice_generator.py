"""Tests for voice generation orchestration."""

import asyncio

import pytest

import app.generators.voice as voice_generator
from app.clients.tts_api import QwenTTSApiClient
from app.generators.voice import generate_voice, validate_tts_text


class FakeQwenTTSApiClient(QwenTTSApiClient):
    def __init__(self, audio_url: str) -> None:
        self.audio_url = audio_url
        self.received_text: str | None = None

    async def generate(self, text: str) -> str:
        self.received_text = text
        return self.audio_url


@pytest.mark.parametrize("text", ["", "   ", "\t\n"])
def test_validate_tts_text_rejects_blank_text(text: str) -> None:
    with pytest.raises(ValueError, match="TTS text must not be empty"):
        validate_tts_text(text)


def test_validate_tts_text_trims_surrounding_whitespace() -> None:
    result = validate_tts_text("  Welcome to the video  ")

    assert result == "Welcome to the video"
    assert isinstance(result, str)


def test_validate_tts_text_returns_valid_text() -> None:
    text = "Welcome to the video"

    assert validate_tts_text(text) == text


def test_generate_voice_runs_complete_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeQwenTTSApiClient("https://example.com/audio.wav")
    download_call: tuple[str, str] | None = None

    async def fake_download(url: str, output_path: str) -> str:
        nonlocal download_call
        download_call = (url, output_path)
        return output_path

    monkeypatch.setattr(voice_generator, "download_file", fake_download)

    result = asyncio.run(
        generate_voice(
            "  Welcome to the video  ",
            "output/voice.wav",
            client,
        )
    )

    assert client.received_text == "Welcome to the video"
    assert download_call == (
        "https://example.com/audio.wav",
        "output/voice.wav",
    )
    assert result == "output/voice.wav"


def test_generate_voice_creates_default_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeQwenTTSApiClient("https://example.com/default.wav")

    async def fake_download(url: str, output_path: str) -> str:
        return output_path

    monkeypatch.setattr(voice_generator, "QwenTTSApiClient", lambda: client)
    monkeypatch.setattr(voice_generator, "download_file", fake_download)

    result = asyncio.run(generate_voice("Welcome", "output/default.wav"))

    assert client.received_text == "Welcome"
    assert result == "output/default.wav"
