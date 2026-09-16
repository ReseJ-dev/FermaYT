"""Direct DashScope clients for structured planning and visual QA."""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, ClassVar

from app.clients.structured_completion import (
    StructuredCompletionMetadata,
    complete_json_chat,
    missing_key_error,
)
from app.errors import StructuredAIProviderError


class _DashScopeStructuredClient:
    ENDPOINT: ClassVar[str] = (
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
    )
    TIMEOUT_SECONDS: ClassVar[float] = 300.0
    MAX_ATTEMPTS: ClassVar[int] = 3
    RETRY_BASE_DELAY_SECONDS: ClassVar[float] = 1.0
    MAX_OUTPUT_TOKENS: ClassVar[int] = 32_768
    provider: ClassVar[str] = "dashscope"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str,
        endpoint: str | None = None,
        timeout: float = TIMEOUT_SECONDS,
        max_attempts: int = MAX_ATTEMPTS,
        retry_base_delay: float = RETRY_BASE_DELAY_SECONDS,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        allow_transport_retries: bool = True,
    ) -> None:
        self.api_key = api_key
        self.model = model.strip()
        self.endpoint = endpoint or self.ENDPOINT
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.retry_base_delay = retry_base_delay
        self.max_output_tokens = max_output_tokens
        self.configured_max_output_tokens = max_output_tokens
        self.allow_transport_retries = allow_transport_retries
        self.last_completion_metadata: StructuredCompletionMetadata | None = None
        if not self.model:
            raise ValueError("DashScope model must not be empty")
        if timeout <= 0:
            raise ValueError("DashScope timeout must be positive")
        if max_attempts < 1:
            raise ValueError("DashScope max_attempts must be positive")
        if retry_base_delay < 0:
            raise ValueError("DashScope retry_base_delay must not be negative")
        if max_output_tokens < 1:
            raise ValueError("DashScope max_output_tokens must be positive")

    async def _complete(
        self,
        messages: list[dict[str, Any]],
        *,
        operation: str,
    ) -> str:
        api_key = self.api_key or os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise missing_key_error(
                provider=self.provider,
                model=self.model,
                operation=operation,
                environment_name="DASHSCOPE_API_KEY",
            )
        self.last_completion_metadata = None
        return await complete_json_chat(
            provider=self.provider,
            model=self.model,
            operation=operation,
            endpoint=self.endpoint,
            api_key=api_key,
            messages=messages,
            timeout=self.timeout,
            max_attempts=self.max_attempts if self.allow_transport_retries else 1,
            retry_base_delay=(self.retry_base_delay if self.allow_transport_retries else 0),
            max_output_tokens=self.max_output_tokens,
            temperature=0.1,
            on_metadata=self._capture_metadata,
        )

    def _capture_metadata(self, metadata: StructuredCompletionMetadata) -> None:
        self.last_completion_metadata = metadata


class DashScopeVisualPlanningClient(_DashScopeStructuredClient):
    """Production complete-story client for the existing VisualDirector."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "qwen-plus",
        endpoint: str | None = None,
        timeout: float = 600.0,
        max_attempts: int = 1,
        retry_base_delay: float = 0.0,
        max_output_tokens: int = _DashScopeStructuredClient.MAX_OUTPUT_TOKENS,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            endpoint=endpoint,
            timeout=timeout,
            max_attempts=max_attempts,
            retry_base_delay=retry_base_delay,
            max_output_tokens=max_output_tokens,
            allow_transport_retries=False,
        )

    async def generate(self, prompt: str) -> str:
        return await self._complete(
            [{"role": "user", "content": prompt}],
            operation="visual_planning",
        )


class DashScopeVisualQAClient(_DashScopeStructuredClient):
    """Production vision client for the existing VisualQAService."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "qwen-vl-max",
        endpoint: str | None = None,
        timeout: float = _DashScopeStructuredClient.TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(
            api_key=api_key, model=model, endpoint=endpoint, timeout=timeout
        )

    async def evaluate(self, prompt: str, image_paths: tuple[str, ...]) -> str:
        if not image_paths:
            raise StructuredAIProviderError("Visual QA requires at least one image")
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
            for path in image_paths
        ]
        content.append({"type": "text", "text": prompt})
        return await self._complete(
            [{"role": "user", "content": content}],
            operation="visual_qa",
        )


def _image_data_url(path: str) -> str:
    image = Path(path)
    if not image.is_file():
        raise StructuredAIProviderError("Visual QA image is missing")
    mime_type = mimetypes.guess_type(image)[0] or "image/png"
    if not mime_type.startswith("image/"):
        raise StructuredAIProviderError("Visual QA input is not an image")
    try:
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    except OSError as exc:
        raise StructuredAIProviderError("Visual QA image cannot be read") from exc
    return f"data:{mime_type};base64,{encoded}"
