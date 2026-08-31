"""Tests for image generation helpers."""

import asyncio

import pytest

import app.generators.image as image_generator
from app.clients.image_api import ImageApiClient
from app.generators.image import (
    build_image_generation_prompt,
    generate_image,
    validate_image_prompt,
)


class FakeImageApiClient(ImageApiClient):
    def __init__(self, image_url: str) -> None:
        self.image_url = image_url
        self.received_prompt: str | None = None

    async def generate(self, prompt: str) -> str:
        self.received_prompt = prompt
        return self.image_url


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


def test_build_image_generation_prompt_combines_scene_and_style() -> None:
    scene_prompt = "  A boy opens a glowing book  "
    style_prompt = "  layered cardboard, warm colors  "

    result = build_image_generation_prompt(scene_prompt, style_prompt)

    assert result == (
        "A boy opens a glowing book\n\nlayered cardboard, warm colors"
    )
    assert scene_prompt == "  A boy opens a glowing book  "
    assert style_prompt == "  layered cardboard, warm colors  "


@pytest.mark.parametrize("style_prompt", [None, "", "   ", "\t\n"])
def test_build_image_generation_prompt_ignores_empty_style(
    style_prompt: str | None,
) -> None:
    result = build_image_generation_prompt(
        "  A boy opens a glowing book  ",
        style_prompt,
    )

    assert result == "A boy opens a glowing book"


def test_build_image_generation_prompt_validates_scene_prompt() -> None:
    with pytest.raises(ValueError, match="image prompt must not be empty"):
        build_image_generation_prompt("   ", "paper art")


def test_generate_image_runs_complete_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeImageApiClient("https://example.com/generated.png")
    download_call: tuple[str, str] | None = None

    async def fake_download(url: str, output_path: str) -> str:
        nonlocal download_call
        download_call = (url, output_path)
        return output_path

    monkeypatch.setattr(image_generator, "download_file", fake_download)

    result = asyncio.run(
        generate_image(
            "  cinematic mountain  ",
            "output/image.png",
            client,
        )
    )

    assert client.received_prompt == "cinematic mountain"
    assert download_call == (
        "https://example.com/generated.png",
        "output/image.png",
    )
    assert result == "output/image.png"


def test_generate_image_creates_default_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeImageApiClient("https://example.com/default.png")

    async def fake_download(url: str, output_path: str) -> str:
        return output_path

    monkeypatch.setattr(image_generator, "ImageApiClient", lambda: client)
    monkeypatch.setattr(image_generator, "download_file", fake_download)

    result = asyncio.run(generate_image("A forest", "output/default.png"))

    assert client.received_prompt == "A forest"
    assert result == "output/default.png"
