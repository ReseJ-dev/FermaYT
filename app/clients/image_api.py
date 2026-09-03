"""Image generation API client."""

import base64
import mimetypes
import os
from pathlib import Path
from typing import ClassVar, Protocol

import httpx

from app.provider_capabilities import ImageProviderCapabilities
from app.provider_diagnostics import (
    ImageProviderDiagnostic,
    safe_provider_response,
    sanitize_provider_message,
)


class ImageGenerationError(Exception):
    """Raised when an external image generation API fails."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic: ImageProviderDiagnostic | None = None,
    ) -> None:
        super().__init__(message)
        self.safe_diagnostic = diagnostic


def _diagnostic_error(
    message: str,
    *,
    provider: str,
    model: str | None,
    operation: str,
    error_type: str,
    request_stage: str,
    http_status: int | None = None,
    provider_error: object = None,
) -> ImageGenerationError:
    return ImageGenerationError(
        message,
        diagnostic=ImageProviderDiagnostic(
            provider=provider,
            model=model,
            operation=operation,
            error_type=error_type,
            request_stage=request_stage,
            http_status=http_status,
            provider_error=sanitize_provider_message(provider_error),
        ),
    )


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
        return await self._generate(prompt, (), operation="generate")

    async def generate_with_references(
        self,
        prompt: str,
        references: tuple[_ReferenceInput, ...],
    ) -> str:
        """Generate one continuity-aware image from up to 14 references."""
        try:
            images = _prepare_references(references, maximum=14)
        except ImageGenerationError as exc:
            raise _diagnostic_error(
                str(exc),
                provider="seedream",
                model=self.model,
                operation="reference",
                error_type="provider_validation",
                request_stage="reference_preparation",
                provider_error=str(exc),
            ) from exc
        return await self._generate(prompt, tuple(images), operation="reference")

    async def edit(
        self,
        prompt: str,
        references: tuple[_ReferenceInput, ...],
    ) -> str:
        """Edit reference content through Seedream's image-input operation."""
        try:
            images = _prepare_references(references, maximum=14)
        except ImageGenerationError as exc:
            raise _diagnostic_error(
                str(exc),
                provider="seedream",
                model=self.model,
                operation="edit",
                error_type="provider_validation",
                request_stage="reference_preparation",
                provider_error=str(exc),
            ) from exc
        return await self._generate(prompt, tuple(images), operation="edit")

    async def _generate(
        self,
        prompt: str,
        images: tuple[str, ...],
        *,
        operation: str,
    ) -> str:
        from app.generators.image import validate_image_prompt

        try:
            validated_prompt = validate_image_prompt(prompt)
        except ValueError as exc:
            raise _diagnostic_error(
                "Invalid image prompt",
                provider="seedream",
                model=self.model,
                operation=operation,
                error_type="provider_validation",
                request_stage="request_validation",
                provider_error=str(exc),
            ) from exc

        api_key = (
            self.api_key
            if self.api_key is not None
            else os.getenv("BYTEPLUS_ARK_API_KEY")
        )
        if not api_key or not api_key.strip():
            raise _diagnostic_error(
                "BYTEPLUS_ARK_API_KEY environment variable is not set",
                provider="seedream",
                model=self.model,
                operation=operation,
                error_type="provider_validation",
                request_stage="request_validation",
                provider_error="Provider credential is not configured",
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
            raise _diagnostic_error(
                "BytePlus Image API request timed out",
                provider="seedream",
                model=self.model,
                operation=operation,
                error_type="timeout",
                request_stage="provider_request",
                provider_error=str(exc),
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise _diagnostic_error(
                f"BytePlus Image API returned HTTP {exc.response.status_code}",
                provider="seedream",
                model=self.model,
                operation=operation,
                error_type="http",
                request_stage="provider_response",
                http_status=exc.response.status_code,
                provider_error=safe_provider_response(exc.response),
            ) from exc
        except httpx.RequestError as exc:
            raise _diagnostic_error(
                "BytePlus Image API request failed",
                provider="seedream",
                model=self.model,
                operation=operation,
                error_type="network",
                request_stage="provider_request",
                provider_error=str(exc),
            ) from exc

        try:
            response_data = response.json()
        except ValueError as exc:
            raise _diagnostic_error(
                "BytePlus Image API returned invalid JSON",
                provider="seedream",
                model=self.model,
                operation=operation,
                error_type="provider_validation",
                request_stage="provider_response_validation",
                provider_error="Invalid JSON response",
            ) from exc

        if not isinstance(response_data, dict):
            raise _diagnostic_error(
                "BytePlus Image API response does not contain image data",
                provider="seedream",
                model=self.model,
                operation=operation,
                error_type="provider_validation",
                request_stage="provider_response_validation",
                provider_error="Response does not contain image data",
            )

        images = response_data.get("data")
        if not isinstance(images, list) or not images:
            raise _diagnostic_error(
                "BytePlus Image API response does not contain image data",
                provider="seedream",
                model=self.model,
                operation=operation,
                error_type="provider_validation",
                request_stage="provider_response_validation",
                provider_error="Response image data is missing or empty",
            )

        first_image = images[0]
        if not isinstance(first_image, dict):
            raise _diagnostic_error(
                "BytePlus Image API response does not contain an image URL",
                provider="seedream",
                model=self.model,
                operation=operation,
                error_type="provider_validation",
                request_stage="provider_response_validation",
                provider_error="First image result is invalid",
            )

        image_url = first_image.get("url")
        if not isinstance(image_url, str) or not image_url.strip():
            raise _diagnostic_error(
                "BytePlus Image API response does not contain an image URL",
                provider="seedream",
                model=self.model,
                operation=operation,
                error_type="provider_validation",
                request_stage="provider_response_validation",
                provider_error="First image URL is missing or empty",
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
        return await self._generate(prompt, (), operation="generate")

    async def generate_with_references(
        self,
        prompt: str,
        references: tuple[_ReferenceInput, ...],
    ) -> str:
        """Generate or recompose an image using one to three references."""
        try:
            images = _prepare_references(references, maximum=3)
        except ImageGenerationError as exc:
            raise _diagnostic_error(
                str(exc),
                provider="qwen",
                model=self.model,
                operation="reference",
                error_type="provider_validation",
                request_stage="reference_preparation",
                provider_error=str(exc),
            ) from exc
        return await self._generate(prompt, tuple(images), operation="reference")

    async def edit(
        self,
        prompt: str,
        references: tuple[_ReferenceInput, ...],
    ) -> str:
        """Edit one to three reference images using Qwen Image 3.0."""
        try:
            images = _prepare_references(references, maximum=3)
        except ImageGenerationError as exc:
            raise _diagnostic_error(
                str(exc),
                provider="qwen",
                model=self.model,
                operation="edit",
                error_type="provider_validation",
                request_stage="reference_preparation",
                provider_error=str(exc),
            ) from exc
        return await self._generate(prompt, tuple(images), operation="edit")

    async def _generate(
        self,
        prompt: str,
        images: tuple[str, ...],
        *,
        operation: str,
    ) -> str:
        from app.generators.image import validate_image_prompt

        try:
            validated_prompt = validate_image_prompt(prompt)
        except ValueError as exc:
            raise _diagnostic_error(
                "Invalid image prompt",
                provider="qwen",
                model=self.model,
                operation=operation,
                error_type="provider_validation",
                request_stage="request_validation",
                provider_error=str(exc),
            ) from exc

        api_key = (
            self.api_key
            if self.api_key is not None
            else os.getenv("DASHSCOPE_API_KEY")
        )
        if not api_key or not api_key.strip():
            raise _diagnostic_error(
                "DASHSCOPE_API_KEY environment variable is not set",
                provider="qwen",
                model=self.model,
                operation=operation,
                error_type="provider_validation",
                request_stage="request_validation",
                provider_error="Provider credential is not configured",
            )

        endpoint = (
            self.endpoint
            if self.endpoint is not None
            else os.getenv("QWEN_IMAGE_ENDPOINT")
        )
        if not endpoint or not endpoint.strip():
            raise _diagnostic_error(
                "QWEN_IMAGE_ENDPOINT environment variable is not set",
                provider="qwen",
                model=self.model,
                operation=operation,
                error_type="provider_validation",
                request_stage="request_validation",
                provider_error="Provider endpoint is not configured",
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
            raise _diagnostic_error(
                "Qwen Image API request timed out",
                provider="qwen",
                model=self.model,
                operation=operation,
                error_type="timeout",
                request_stage="provider_request",
                provider_error=str(exc),
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise _diagnostic_error(
                f"Qwen Image API returned HTTP {exc.response.status_code}",
                provider="qwen",
                model=self.model,
                operation=operation,
                error_type="http",
                request_stage="provider_response",
                http_status=exc.response.status_code,
                provider_error=safe_provider_response(exc.response),
            ) from exc
        except httpx.RequestError as exc:
            raise _diagnostic_error(
                "Qwen Image API request failed",
                provider="qwen",
                model=self.model,
                operation=operation,
                error_type="network",
                request_stage="provider_request",
                provider_error=str(exc),
            ) from exc

        try:
            response_data = response.json()
        except ValueError as exc:
            raise _diagnostic_error(
                "Qwen Image API returned invalid JSON",
                provider="qwen",
                model=self.model,
                operation=operation,
                error_type="provider_validation",
                request_stage="provider_response_validation",
                provider_error="Invalid JSON response",
            ) from exc

        if not isinstance(response_data, dict):
            raise _diagnostic_error(
                "Qwen Image API response does not contain output",
                provider="qwen",
                model=self.model,
                operation=operation,
                error_type="provider_validation",
                request_stage="provider_response_validation",
                provider_error="Response is not a JSON object",
            )

        error_code = response_data.get("code")
        error_message = response_data.get("message")
        if error_code or error_message:
            safe_code = sanitize_provider_message(error_code) or "unknown"
            safe_error_message = (
                sanitize_provider_message(error_message) or "Provider error"
            )
            safe_message = sanitize_provider_message(
                {"code": error_code, "message": error_message}
            )
            raise _diagnostic_error(
                f"Qwen Image API error {safe_code}: {safe_error_message}",
                provider="qwen",
                model=self.model,
                operation=operation,
                error_type="provider_validation",
                request_stage="provider_response_validation",
                provider_error=safe_message,
            )

        output = response_data.get("output")
        if not isinstance(output, dict):
            raise _diagnostic_error(
                "Qwen Image API response does not contain output",
                provider="qwen",
                model=self.model,
                operation=operation,
                error_type="provider_validation",
                request_stage="provider_response_validation",
                provider_error="Response output is missing or invalid",
            )

        choices = output.get("choices")
        if not isinstance(choices, list) or not choices:
            raise _diagnostic_error(
                "Qwen Image API response does not contain choices",
                provider="qwen",
                model=self.model,
                operation=operation,
                error_type="provider_validation",
                request_stage="provider_response_validation",
                provider_error="Response choices are missing or empty",
            )

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise _diagnostic_error(
                "Qwen Image API response does not contain an image URL",
                provider="qwen",
                model=self.model,
                operation=operation,
                error_type="provider_validation",
                request_stage="provider_response_validation",
                provider_error="First choice is invalid",
            )

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise _diagnostic_error(
                "Qwen Image API response does not contain an image URL",
                provider="qwen",
                model=self.model,
                operation=operation,
                error_type="provider_validation",
                request_stage="provider_response_validation",
                provider_error="Choice message is missing or invalid",
            )

        content = message.get("content")
        if not isinstance(content, list) or not content:
            raise _diagnostic_error(
                "Qwen Image API response does not contain an image URL",
                provider="qwen",
                model=self.model,
                operation=operation,
                error_type="provider_validation",
                request_stage="provider_response_validation",
                provider_error="Message content is missing or empty",
            )

        first_content = content[0]
        if not isinstance(first_content, dict):
            raise _diagnostic_error(
                "Qwen Image API response does not contain an image URL",
                provider="qwen",
                model=self.model,
                operation=operation,
                error_type="provider_validation",
                request_stage="provider_response_validation",
                provider_error="First content item is invalid",
            )

        image_url = first_content.get("image")
        if not isinstance(image_url, str) or not image_url.strip():
            raise _diagnostic_error(
                "Qwen Image API response does not contain an image URL",
                provider="qwen",
                model=self.model,
                operation=operation,
                error_type="provider_validation",
                request_stage="provider_response_validation",
                provider_error="Image URL is missing or empty",
            )

        return image_url.strip()


SeedreamImageProvider = BytePlusImageApiClient
QwenImageProvider = QwenImageApiClient
ImageApiClient = SeedreamImageProvider
