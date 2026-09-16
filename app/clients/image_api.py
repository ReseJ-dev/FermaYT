"""Image generation API client."""

import asyncio
import base64
import json
import mimetypes
import os
import time
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
            validated_prompt, _ = normalize_image_prompt_for_provider(
                validate_image_prompt(prompt), "seedream"
            )
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
    TIMEOUT_SECONDS: ClassVar[float] = 180.0
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
        """Edit one to three reference images using the configured Qwen model."""
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
            validated_prompt, _ = normalize_image_prompt_for_provider(
                validate_image_prompt(prompt), "qwen"
            )
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


class KieZImageApiClient:
    """Asynchronous task client for Z-Image hosted by Kie.ai."""

    CREATE_URL: ClassVar[str] = "https://api.kie.ai/api/v1/jobs/createTask"
    STATUS_URL: ClassVar[str] = "https://api.kie.ai/api/v1/jobs/recordInfo"
    MODEL_ID: ClassVar[str] = "z-image"
    REQUEST_TIMEOUT_SECONDS: ClassVar[float] = 30.0
    GENERATION_TIMEOUT_SECONDS: ClassVar[float] = 600.0
    PROMPT_MAX_CHARACTERS: ClassVar[int] = 800
    POLL_INTERVAL_SECONDS: ClassVar[float] = 2.0
    capabilities: ClassVar[ImageProviderCapabilities] = ImageProviderCapabilities()

    def __init__(
        self,
        api_key: str | None = None,
        model: str = MODEL_ID,
        aspect_ratio: str = "9:16",
        poll_interval: float = POLL_INTERVAL_SECONDS,
        generation_timeout: float | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.aspect_ratio = aspect_ratio
        self.poll_interval = poll_interval
        configured_timeout = (
            generation_timeout
            if generation_timeout is not None
            else _environment_float(
                "KIE_ZIMAGE_GENERATION_TIMEOUT_SECONDS",
                self.GENERATION_TIMEOUT_SECONDS,
            )
        )
        self.generation_timeout = configured_timeout
        if aspect_ratio not in {"1:1", "4:3", "3:4", "16:9", "9:16"}:
            raise ValueError("Unsupported Kie.ai Z-Image aspect ratio")
        if poll_interval < 0:
            raise ValueError("Kie.ai polling interval must not be negative")
        if configured_timeout <= 0:
            raise ValueError("Kie.ai generation timeout must be positive")

    async def generate(self, prompt: str) -> str:
        """Create, poll, and return one generated Kie.ai image URL."""
        from app.generators.image import validate_image_prompt

        try:
            validated_prompt, _ = normalize_image_prompt_for_provider(
                validate_image_prompt(prompt), "zimage"
            )
        except ValueError as exc:
            raise _diagnostic_error(
                "Invalid image prompt",
                provider="zimage",
                model=self.model,
                operation="generate",
                error_type="provider_validation",
                request_stage="request_validation",
                provider_error=str(exc),
            ) from exc

        api_key = (
            self.api_key
            if self.api_key is not None
            else os.getenv("KIE_API_KEY", "").strip()
        )
        if not api_key:
            raise _diagnostic_error(
                "KIE_API_KEY environment variable is not set",
                provider="zimage",
                model=self.model,
                operation="generate",
                error_type="provider_validation",
                request_stage="request_validation",
                provider_error="Provider credential is not configured",
            )

        payload = {
            "model": self.model,
            "input": {
                "prompt": validated_prompt,
                "aspect_ratio": self.aspect_ratio,
                "nsfw_checker": True,
            },
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.REQUEST_TIMEOUT_SECONDS
            ) as client:
                response = await client.post(
                    self.CREATE_URL,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                task_id = _kie_task_id(response)
                return await self._poll_result(client, headers, task_id)
        except httpx.TimeoutException as exc:
            raise _diagnostic_error(
                "Kie.ai Z-Image request timed out",
                provider="zimage",
                model=self.model,
                operation="generate",
                error_type="timeout",
                request_stage="provider_request",
                provider_error=str(exc),
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise _diagnostic_error(
                f"Kie.ai Z-Image returned HTTP {exc.response.status_code}",
                provider="zimage",
                model=self.model,
                operation="generate",
                error_type="http",
                request_stage="provider_response",
                http_status=exc.response.status_code,
                provider_error=safe_provider_response(exc.response),
            ) from exc
        except httpx.RequestError as exc:
            raise _diagnostic_error(
                "Kie.ai Z-Image request failed",
                provider="zimage",
                model=self.model,
                operation="generate",
                error_type="network",
                request_stage="provider_request",
                provider_error=str(exc),
            ) from exc

        except ImageGenerationError:
            raise

    async def _poll_result(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        task_id: str,
    ) -> str:
        deadline = time.monotonic() + self.generation_timeout
        last_poll_error: str | None = None
        while time.monotonic() < deadline:
            try:
                response = await client.get(
                    self.STATUS_URL,
                    headers=headers,
                    params={"taskId": task_id},
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 429 and exc.response.status_code < 500:
                    raise
                last_poll_error = f"HTTP {exc.response.status_code}"
                await asyncio.sleep(self.poll_interval)
                continue
            except httpx.RequestError as exc:
                last_poll_error = type(exc).__name__
                await asyncio.sleep(self.poll_interval)
                continue
            state, image_url, failure = _kie_task_result(response)
            if state == "success" and image_url is not None:
                return image_url
            if state == "fail":
                raise _diagnostic_error(
                    "Kie.ai Z-Image generation failed",
                    provider="zimage",
                    model=self.model,
                    operation="generate",
                    error_type="provider_validation",
                    request_stage="provider_response",
                    provider_error=failure or "Generation task failed",
                )
            await asyncio.sleep(self.poll_interval)
        raise _diagnostic_error(
            "Kie.ai Z-Image generation timed out",
            provider="zimage",
            model=self.model,
            operation="generate",
            error_type="timeout",
            request_stage="provider_response",
            provider_error=(
                f"Generation task {task_id} did not finish within "
                f"{self.generation_timeout:g} seconds"
                + (f"; last polling error: {last_poll_error}" if last_poll_error else "")
            ),
        )


def _environment_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _kie_task_id(response: httpx.Response) -> str:
    try:
        body = response.json()
        if body.get("code") != 200:
            raise ValueError("task creation was rejected")
        task_id = body["data"]["taskId"]
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise _diagnostic_error(
            "Kie.ai Z-Image returned an invalid task response",
            provider="zimage",
            model=KieZImageApiClient.MODEL_ID,
            operation="generate",
            error_type="provider_validation",
            request_stage="provider_response_validation",
            provider_error="Task ID is missing or invalid",
        ) from exc
    if not isinstance(task_id, str) or not task_id.strip():
        raise _diagnostic_error(
            "Kie.ai Z-Image returned an invalid task response",
            provider="zimage",
            model=KieZImageApiClient.MODEL_ID,
            operation="generate",
            error_type="provider_validation",
            request_stage="provider_response_validation",
            provider_error="Task ID is missing or invalid",
        )
    return task_id.strip()


def _kie_task_result(response: httpx.Response) -> tuple[str, str | None, str | None]:
    try:
        body = response.json()
        if body.get("code") != 200:
            raise ValueError("task query was rejected")
        data = body["data"]
        state = data["state"]
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise _diagnostic_error(
            "Kie.ai Z-Image returned invalid task status",
            provider="zimage",
            model=KieZImageApiClient.MODEL_ID,
            operation="generate",
            error_type="provider_validation",
            request_stage="provider_response_validation",
            provider_error="Task status is missing or invalid",
        ) from exc
    if state != "success":
        return str(state), None, sanitize_provider_message(data.get("failMsg"))
    try:
        result = json.loads(data["resultJson"])
        image_url = result["resultUrls"][0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise _diagnostic_error(
            "Kie.ai Z-Image response does not contain an image URL",
            provider="zimage",
            model=KieZImageApiClient.MODEL_ID,
            operation="generate",
            error_type="provider_validation",
            request_stage="provider_response_validation",
            provider_error="Result URL is missing or invalid",
        ) from exc
    if not isinstance(image_url, str) or not image_url.strip():
        raise _diagnostic_error(
            "Kie.ai Z-Image response does not contain an image URL",
            provider="zimage",
            model=KieZImageApiClient.MODEL_ID,
            operation="generate",
            error_type="provider_validation",
            request_stage="provider_response_validation",
            provider_error="Result URL is missing or invalid",
        )
    return "success", image_url.strip(), None


def _fit_kie_zimage_prompt(prompt: str) -> str:
    """Fit semantic beat essentials and the style contract into Kie's limit."""
    maximum = KieZImageApiClient.PROMPT_MAX_CHARACTERS
    compact = " ".join(prompt.split())
    if len(compact) <= maximum:
        return compact
    style_marker = "Use this permanent drawing style throughout the entire illustration."
    legacy_marker = "STYLE CONTRACT ["
    if style_marker in compact:
        dynamic = compact.split(style_marker, 1)[0].strip()
    elif legacy_marker in compact:
        dynamic = compact.split(legacy_marker, 1)[0].strip()
    else:
        separator = "..."
        head = (maximum - len(separator)) * 2 // 3
        tail = maximum - len(separator) - head
        return f"{compact[:head].rstrip()}{separator}{compact[-tail:].lstrip()}"
    style = (
        "Draw a rough amateur hand-drawn 2D explainer with thick uneven black "
        "outlines, crude geometry, simple cartoon people with dot eyes, flat muted "
        "colors, minimal shading, sparse backgrounds, and imperfect perspective. "
        "Do not depict photorealism, realistic materials or anatomy, cinematic "
        "light, polished art, 3D, depth of field, or gradients. Prefer simplicity."
    )
    available = maximum - len(style) - 1
    semantic_sections = _extract_kie_semantic_sections(dynamic)
    desired_scene = _extract_kie_instruction(
        dynamic,
        "Create an illustration showing ",
        (
            " Create a new image for",
            " Edit the source image",
            " Use the attached images",
            " Also follow this project",
        ),
    )
    qa_correction = _extract_kie_instruction(
        dynamic,
        "Regenerate the illustration so that ",
        (" Preserve every correct visual element",),
    )
    if semantic_sections:
        labels_and_limits = (
            ("{}", "VISUAL FOCUS", 115),
            ("Depict {}.", "CURRENT PHYSICAL STATE", 85),
            ("Show how {}.", "WHAT CHANGED", 90),
            ("Frame the scene as {}.", "CURRENT CAMERA / COMPOSITION", 65),
            ("Keep the setting recognizable as {}.", "LOCATION CONTINUITY", 65),
            ("Exclude {}.", "DO NOT SHOW", 45),
        )
        parts = ["No visible text, labels, titles, watermarks, or UI."]
        if desired_scene:
            parts.append(
                f"Show {_truncate_at_word(desired_scene, 150).rstrip('.')} clearly."
            )
        if qa_correction:
            parts.append(
                f"Correct {_truncate_at_word(qa_correction, 115).rstrip('.')} while preserving correct elements."
            )
        parts.extend(
            template.format(_kie_semantic_excerpt(heading, section, limit))
            for template, heading, limit in labels_and_limits
            if (section := semantic_sections.get(heading))
        )
        dynamic = " ".join(parts)
    elif desired_scene or qa_correction:
        parts = ["No visible text, labels, titles, watermarks, or UI."]
        if desired_scene:
            parts.append(
                f"Show {_truncate_at_word(desired_scene, 300).rstrip('.')} clearly."
            )
        if qa_correction:
            parts.append(
                f"Correct {_truncate_at_word(qa_correction, 150).rstrip('.')} while preserving correct elements."
            )
        dynamic = " ".join(parts)
    if len(dynamic) > available:
        dynamic = _truncate_at_word(dynamic, available)
    fitted = f"{dynamic} {style}"
    return _truncate_at_word(fitted, maximum)


def normalize_image_prompt_for_provider(
    prompt: str,
    provider: str,
) -> tuple[str, list[dict[str, object]]]:
    """Return the exact provider prompt and an inspectable transformation trace."""
    normalized = prompt.strip()
    if not normalized:
        raise ValueError("image prompt must not be empty")
    transformations: list[dict[str, object]] = []
    if normalized != prompt:
        transformations.append(
            {"type": "TRIM", "before_length": len(prompt), "after_length": len(normalized)}
        )
    provider_id = provider.strip().lower()
    if provider_id == "zimage":
        fitted = _fit_kie_zimage_prompt(normalized)
        if fitted != normalized:
            transformations.append(
                {
                    "type": "ZIMAGE_LIMIT_NORMALIZATION",
                    "before_length": len(normalized),
                    "after_length": len(fitted),
                    "maximum_characters": KieZImageApiClient.PROMPT_MAX_CHARACTERS,
                }
            )
        normalized = fitted
    return normalized, transformations


_KIE_SEMANTIC_PREFIXES = (
    ("LOCATION CONTINUITY", "Draw the recurring setting with this stable layout."),
    ("PROJECT STYLE DIRECTION", "Follow this project drawing direction."),
    (
        "CHARACTER CONTINUITY",
        "Show these people with their established roles and appearance.",
    ),
    (
        "OBJECT CONTINUITY",
        "Include these story objects in their established positions.",
    ),
    ("CURRENT CAMERA / COMPOSITION", "Frame the scene this way."),
    ("CURRENT PHYSICAL STATE", "Depict this physical situation."),
    ("WHAT CHANGED", "Make this new physical change clearly visible."),
    ("VISUAL FOCUS", "Guide attention to the story-critical action."),
    ("DO NOT SHOW", "Exclude these story mistakes."),
    (
        "SIMPLIFICATION RULE",
        "Keep the image visually simple and immediately readable.",
    ),
    ("LOCATION CONTINUITY", "LOCATION CONTINUITY:"),
    ("PROJECT STYLE DIRECTION", "PROJECT STYLE DIRECTION:"),
    ("CHARACTER CONTINUITY", "CHARACTER CONTINUITY:"),
    ("OBJECT CONTINUITY", "OBJECT CONTINUITY:"),
    ("CURRENT CAMERA / COMPOSITION", "CURRENT CAMERA / COMPOSITION:"),
    ("CURRENT PHYSICAL STATE", "CURRENT PHYSICAL STATE:"),
    ("WHAT CHANGED", "WHAT CHANGED:"),
    ("VISUAL FOCUS", "VISUAL FOCUS:"),
    ("DO NOT SHOW", "DO NOT SHOW:"),
    ("SIMPLIFICATION RULE", "SIMPLIFICATION RULE:"),
)


def _extract_kie_semantic_sections(dynamic_prompt: str) -> dict[str, str]:
    positions: list[tuple[int, str, int]] = []
    for heading, marker in _KIE_SEMANTIC_PREFIXES:
        position = dynamic_prompt.find(marker)
        if position >= 0:
            positions.append((position, heading, position + len(marker)))
    positions.sort()
    sections: dict[str, str] = {}
    for index, (_, heading, content_start) in enumerate(positions):
        content_end = (
            positions[index + 1][0]
            if index + 1 < len(positions)
            else len(dynamic_prompt)
        )
        content = dynamic_prompt[content_start:content_end].strip()
        if content:
            sections[heading] = content
    return sections


def _kie_semantic_excerpt(heading: str, value: str, maximum: int) -> str:
    if heading == "VISUAL FOCUS" and "First notice:" in value:
        before, first_notice = value.split("First notice:", 1)
        value = f"{first_notice.strip()} {before.strip()}"
    value = value.replace(":", ",")
    return _truncate_at_word(value, maximum)


def _extract_kie_instruction(
    prompt: str,
    marker: str,
    end_markers: tuple[str, ...],
) -> str | None:
    if marker not in prompt:
        return None
    value = prompt.split(marker, 1)[1]
    endings = [value.find(item) for item in end_markers if item in value]
    if endings:
        value = value[: min(endings)]
    return value.strip(" .") or None


def _truncate_at_word(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    shortened = value[: max(1, maximum - 3)].rsplit(" ", 1)[0].rstrip(" ,.;:")
    return f"{shortened or value[: maximum - 3]}..."


SeedreamImageProvider = BytePlusImageApiClient
QwenImageProvider = QwenImageApiClient
ZImageProvider = KieZImageApiClient
ImageApiClient = SeedreamImageProvider
