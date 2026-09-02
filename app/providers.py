"""Minimal provider contracts and registry."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from app.clients.image_api import QwenImageProvider, SeedreamImageProvider
from app.clients.tts_api import ElevenLabsTTSApiClient, QwenTTSApiClient
from app.provider_capabilities import ImageProviderCapabilities


@runtime_checkable
class ImageProvider(Protocol):
    capabilities: ImageProviderCapabilities

    async def generate(self, prompt: str) -> str: ...


class ImageReferenceRole(str, Enum):
    STYLE = "STYLE"
    CONTENT_CONTINUITY = "CONTENT_CONTINUITY"


@dataclass(frozen=True, slots=True)
class ImageReference:
    """An immutable image reference with one explicit responsibility."""

    reference_id: str
    file_path: str
    sha256: str
    role: ImageReferenceRole

    def __post_init__(self) -> None:
        if not self.reference_id.strip() or not self.file_path.strip():
            raise ValueError("image reference ID and path must not be empty")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256.lower()
        ):
            raise ValueError("image reference sha256 must be a 64-character hex digest")


@runtime_checkable
class ReferenceImageProvider(Protocol):
    async def generate_with_references(
        self,
        prompt: str,
        references: tuple[ImageReference, ...],
    ) -> str: ...


@runtime_checkable
class ImageEditingProvider(Protocol):
    async def edit(
        self,
        prompt: str,
        references: tuple[ImageReference, ...],
    ) -> str: ...


@runtime_checkable
class TTSProvider(Protocol):
    async def generate(self, text: str) -> str | bytes: ...


def get_image_provider_capabilities(
    provider: ImageProvider,
) -> ImageProviderCapabilities:
    """Discover capabilities from the concrete selected provider instance."""
    capabilities = getattr(provider, "capabilities", None)
    if not isinstance(capabilities, ImageProviderCapabilities):
        raise TypeError("Image provider does not declare validated capabilities")
    return capabilities


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
    provider_name = name.strip().lower()
    if provider_name == "qwen":
        return QwenTTSApiClient(**options)
    if provider_name == "elevenlabs":
        return ElevenLabsTTSApiClient(**options)
    raise ValueError(f"Unknown TTS provider: {name}")
