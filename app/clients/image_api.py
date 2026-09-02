"""Image generation API client."""

import base64
import mimetypes
import os
from pathlib import Path
from typing import ClassVar, Protocol

import httpx

from app.provider_capabilities import ImageProviderCapabilities


class ImageGenerationError(Exception):
    """Raised when an external image generation API fails."""


class _ReferenceInput(Protocol):
    file_path: str


def _encode_reference(reference: _ReferenceInput) -> str:
    path = reference.file_path.strip()
    if path.startswith(("https://", "http://", "data:image/")):
        return path
    file_path = Path(path)
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None or not mime_type.startswith("image/"):
        raise ImageGenerationError("Reference image format is not supported")
    try:
        encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    except OSError as exc:
        raise ImageGenerationError("Reference image cannot be read") from exc
    return f"data:{mime_type};base64,{encoded}"


def _prepare_references(
    references: tuple[_ReferenceInput, ...],
    *,
    maximum: int,
) -> list[str]:
    if not references or len(references) > maximum:
        raise ImageGenerationError(
            f"Image provider requires between 1 and {maximum} reference images"
        )
    return [_encode_reference(reference) for reference in references]


class BytePlusImageApiClient:
    """Minimal client for BytePlus ModelArk image generation."""

    API_URL: ClassVar[str] = (
        "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
    )
    MODEL_ID: ClassVar[str] = "seedream-5-0-260128"
    TIMEOUT_SECONDS: ClassVar[float] = 30.0
    capabilities: ClassVar[ImageProviderCapabilities] = ImageProviderCapabilities(
        reference_generation=True,
        multiple_references=True,
        max_reference_images=14,
        image_editing=True,
    )

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        model: str = MODEL_ID,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint if endpoint is not None else self.API_URL
        self.model = model

    async def generate(self, prompt: str) -> str:
        """Generate one image and return its temporary URL."""
        return await self._generate(prompt, ())

    async def generate_with_references(
        self,
        prompt: str,
        references: tuple[_ReferenceInput, ...],
    ) -> str:
        """Generate one continuity-aware image from up to 14 references."""
        images = _prepare_references(references, maximum=14)
        return await self._generate(prompt, tuple(images))

    async def edit(
        self,
        prompt: str,
        references: tuple[_ReferenceInput, ...],
    ) -> str:
        """Edit reference content through Seedream's image-input operation."""
        return await self.generate_with_references(prompt, references)

    async def _generate(
        self,
        prompt: str,
        images: tuple[str, ...],
    ) -> str:
        from app.generators.image import validate_image_prompt

        try:
            validated_prompt = validate_image_prompt(prompt)
        except ValueError as exc:
            raise ImageGenerationError("Invalid image prompt") from exc

        api_key = (
            self.api_key
            if self.api_key is not None
            else os.getenv("BYTEPLUS_ARK_API_KEY")
        )
        if not api_key or not api_key.strip():
            raise ImageGenerationError(
                "BYTEPLUS_ARK_API_KEY environment variable is not set"
            )

        payload: dict[str, object] = {
            "model": self.model,
            "prompt": validated_prompt,
            "size": "2K",
            "output_format": "png",
            "response_format": "url",
            "watermark": False,
            "sequential_image_generation": "disabled",
            "stream": False,
        }
        if images:
            payload["image"] = list(images)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.TIMEOUT_SECONDS
            ) as client:
                response = await client.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ImageGenerationError(
                "BytePlus Image API request timed out"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ImageGenerationError(
                "BytePlus Image API returned "
                f"HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise ImageGenerationError(
                "BytePlus Image API request failed"
            ) from exc

        try:
            response_data = response.json()
        except ValueError as exc:
            raise ImageGenerationError(
                "BytePlus Image API returned invalid JSON"
            ) from exc

        if not isinstance(response_data, dict):
            raise ImageGenerationError(
                "BytePlus Image API response does not contain image data"
            )

        images = response_data.get("data")
        if not isinstance(images, list) or not images:
            raise ImageGenerationError(
                "BytePlus Image API response does not contain image data"
            )

        first_image = images[0]
        if not isinstance(first_image, dict):
            raise ImageGenerationError(
                "BytePlus Image API response does not contain an image URL"
            )

        image_url = first_image.get("url")
        if not isinstance(image_url, str) or not image_url.strip():
            raise ImageGenerationError(
                "BytePlus Image API response does not contain an image URL"
            )

        return image_url.strip()


class QwenImageApiClient:
    """Minimal client for Qwen Image on Alibaba Cloud Model Studio."""

    MODEL_ID: ClassVar[str] = "qwen-image-3.0"
    TIMEOUT_SECONDS: ClassVar[float] = 30.0
    capabilities: ClassVar[ImageProviderCapabilities] = ImageProviderCapabilities(
        reference_generation=True,
        multiple_references=True,
        max_reference_images=3,
        image_editing=True,
    )

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        model: str = MODEL_ID,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model

    async def generate(self, prompt: str) -> str:
        """Generate one image and return its temporary URL."""
        return await self._generate(prompt, ())

    async def generate_with_references(
        self,
        prompt: str,
        references: tuple[_ReferenceInput, ...],
    ) -> str:
        """Generate or recompose an image using one to three references."""
        images = _prepare_references(references, maximum=3)
        return await self._generate(prompt, tuple(images))

    async def edit(
        self,
        prompt: str,
        references: tuple[_ReferenceInput, ...],
    ) -> str:
        """Edit one to three reference images using Qwen Image 3.0."""
        return await self.generate_with_references(prompt, references)

    async def _generate(
        self,
        prompt: str,
        images: tuple[str, ...],
    ) -> str:
        from app.generators.image import validate_image_prompt

        try:
            validated_prompt = validate_image_prompt(prompt)
        except ValueError as exc:
            raise ImageGenerationError("Invalid image prompt") from exc

        api_key = (
            self.api_key
            if self.api_key is not None
            else os.getenv("DASHSCOPE_API_KEY")
        )
        if not api_key or not api_key.strip():
            raise ImageGenerationError(
                "DASHSCOPE_API_KEY environment variable is not set"
            )

        endpoint = (
            self.endpoint
            if self.endpoint is not None
            else os.getenv("QWEN_IMAGE_ENDPOINT")
        )
        if not endpoint or not endpoint.strip():
            raise ImageGenerationError(
                "QWEN_IMAGE_ENDPOINT environment variable is not set"
            )

        content = [{"image": image} for image in images]
        content.append({"text": validated_prompt})
        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
            },
            "parameters": {
                "prompt_extend": True,
                "n": 1,
                "watermark": False,
            },
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.TIMEOUT_SECONDS
            ) as client:
                response = await client.post(
                    endpoint.strip(),
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ImageGenerationError(
                "Qwen Image API request timed out"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ImageGenerationError(
                f"Qwen Image API returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise ImageGenerationError(
                "Qwen Image API request failed"
            ) from exc

        try:
            response_data = response.json()
        except ValueError as exc:
            raise ImageGenerationError(
                "Qwen Image API returned invalid JSON"
            ) from exc

        if not isinstance(response_data, dict):
            raise ImageGenerationError(
                "Qwen Image API response does not contain output"
            )

        error_code = response_data.get("code")
        error_message = response_data.get("message")
        if error_code or error_message:
            raise ImageGenerationError(
                f"Qwen Image API error {error_code}: {error_message}"
            )

        output = response_data.get("output")
        if not isinstance(output, dict):
            raise ImageGenerationError(
                "Qwen Image API response does not contain output"
            )

        choices = output.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ImageGenerationError(
                "Qwen Image API response does not contain choices"
            )

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ImageGenerationError(
                "Qwen Image API response does not contain an image URL"
            )

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ImageGenerationError(
                "Qwen Image API response does not contain an image URL"
            )

        content = message.get("content")
        if not isinstance(content, list) or not content:
            raise ImageGenerationError(
                "Qwen Image API response does not contain an image URL"
            )

        first_content = content[0]
        if not isinstance(first_content, dict):
            raise ImageGenerationError(
                "Qwen Image API response does not contain an image URL"
            )

        image_url = first_content.get("image")
        if not isinstance(image_url, str) or not image_url.strip():
            raise ImageGenerationError(
                "Qwen Image API response does not contain an image URL"
            )

        return image_url.strip()


SeedreamImageProvider = BytePlusImageApiClient
QwenImageProvider = QwenImageApiClient
ImageApiClient = SeedreamImageProvider
