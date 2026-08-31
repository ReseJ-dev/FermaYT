"""Minimal provider contracts and registry."""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from app.clients.image_api import QwenImageProvider, SeedreamImageProvider
from app.clients.tts_api import QwenTTSApiClient


@runtime_checkable
class ImageProvider(Protocol):
    async def generate(self, prompt: str) -> str: ...


@runtime_checkable
class TTSProvider(Protocol):
    async def generate(self, text: str) -> str: ...


def get_image_provider(
    name: str,
    config: Mapping[str, Any] | None = None,
) -> ImageProvider:
    """Create a configured image provider by its public ID."""
    options = dict(config or {})
    provider_name = name.strip().lower()
    if provider_name == "seedream":
        return SeedreamImageProvider(**options)
    if provider_name == "qwen":
        return QwenImageProvider(**options)
    raise ValueError(f"Unknown image provider: {name}")


def get_tts_provider(
    name: str,
    config: Mapping[str, Any] | None = None,
) -> TTSProvider:
    """Create a configured TTS provider by its public ID."""
    options = dict(config or {})
    if name.strip().lower() == "qwen":
        return QwenTTSApiClient(**options)
    raise ValueError(f"Unknown TTS provider: {name}")
