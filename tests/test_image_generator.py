"""Tests for image generation helpers."""

import pytest

from app.generators.image import validate_image_prompt


@pytest.mark.parametrize("prompt", ["", "   ", "\t\n"])
def test_validate_image_prompt_rejects_blank_prompt(prompt: str) -> None:
    with pytest.raises(ValueError, match="image prompt must not be empty"):
        validate_image_prompt(prompt)


def test_validate_image_prompt_trims_surrounding_whitespace() -> None:
    result = validate_image_prompt("  cinematic mountain landscape  ")

    assert result == "cinematic mountain landscape"
    assert isinstance(result, str)


def test_validate_image_prompt_returns_valid_prompt() -> None:
    prompt = "cinematic mountain landscape"

    assert validate_image_prompt(prompt) == prompt
