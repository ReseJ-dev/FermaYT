"""Tests for provider contracts and registry."""

import pytest

from app.clients.image_api import QwenImageProvider, SeedreamImageProvider
from app.clients.tts_api import ElevenLabsTTSApiClient, QwenTTSApiClient
from app.providers import (
    ImageProvider,
    TTSProvider,
    get_image_provider,
    get_image_provider_capabilities,
    get_tts_provider,
    get_tts_provider_capabilities,
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


@pytest.mark.parametrize("provider_name", ["seedream", "qwen"])
def test_current_image_clients_advertise_verified_reference_capabilities(
    provider_name: str,
) -> None:
    provider = get_image_provider(provider_name)
    capabilities = get_image_provider_capabilities(provider)

    assert capabilities.text_to_image is True
    assert capabilities.reference_generation is True
    assert capabilities.multiple_references is True
    assert capabilities.max_reference_images == {
        "seedream": 14,
        "qwen": 3,
    }[provider_name]
    assert capabilities.image_editing is True
    assert capabilities.seed is False
    assert capabilities.aspect_ratio is False


def test_get_qwen_tts_provider_with_config() -> None:
    provider = get_tts_provider(
        "qwen",
        {"api_key": "key", "voice": "Cherry", "language": "Russian"},
    )

    assert isinstance(provider, QwenTTSApiClient)
    assert isinstance(provider, TTSProvider)
    assert provider.api_key == "key"


def test_get_elevenlabs_tts_provider_with_config() -> None:
    provider = get_tts_provider(
        "elevenlabs",
        {"api_key": "key", "voice": "voice-id"},
    )

    assert isinstance(provider, ElevenLabsTTSApiClient)
    assert isinstance(provider, TTSProvider)
    assert provider.api_key == "key"
    assert provider.voice == "voice-id"


@pytest.mark.parametrize("provider_name", ["qwen", "elevenlabs"])
def test_current_tts_integrations_do_not_claim_unimplemented_timestamps(
    provider_name: str,
) -> None:
    capabilities = get_tts_provider_capabilities(get_tts_provider(provider_name))
    assert capabilities.supports_word_timestamps is False
    assert capabilities.supports_character_timestamps is False
    assert capabilities.supports_sentence_timestamps is False
    assert capabilities.supports_speech_marks is False


@pytest.mark.parametrize("factory", [get_image_provider, get_tts_provider])
def test_unknown_provider_is_rejected(factory: object) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        factory("unknown")  # type: ignore[operator]
