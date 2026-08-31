"""Tests for provider contracts and registry."""

import pytest

from app.clients.image_api import QwenImageProvider, SeedreamImageProvider
from app.clients.tts_api import QwenTTSApiClient
from app.providers import (
    ImageProvider,
    TTSProvider,
    get_image_provider,
    get_tts_provider,
)


def test_get_seedream_image_provider_with_config() -> None:
    provider = get_image_provider(
        "seedream",
        {
            "api_key": "key",
            "endpoint": "https://example.com/seedream",
            "model": "seedream-model",
        },
    )

    assert isinstance(provider, SeedreamImageProvider)
    assert isinstance(provider, ImageProvider)
    assert provider.api_key == "key"
    assert provider.endpoint == "https://example.com/seedream"
    assert provider.model == "seedream-model"


def test_get_qwen_image_provider_with_config() -> None:
    provider = get_image_provider(
        "qwen",
        {"api_key": "key", "endpoint": "https://example.com/qwen"},
    )

    assert isinstance(provider, QwenImageProvider)
    assert isinstance(provider, ImageProvider)


def test_get_qwen_tts_provider_with_config() -> None:
    provider = get_tts_provider(
        "qwen",
        {"api_key": "key", "voice": "Cherry", "language": "Russian"},
    )

    assert isinstance(provider, QwenTTSApiClient)
    assert isinstance(provider, TTSProvider)
    assert provider.api_key == "key"


@pytest.mark.parametrize("factory", [get_image_provider, get_tts_provider])
def test_unknown_provider_is_rejected(factory: object) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        factory("unknown")  # type: ignore[operator]
