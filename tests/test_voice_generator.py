"""Tests for voice generation orchestration."""

import pytest

from app.generators.voice import validate_tts_text


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
