"""Direct DashScope clients for structured planning and visual QA."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, ClassVar

import httpx

from app.errors import StructuredAIProviderError


class _DashScopeStructuredClient:
    ENDPOINT: ClassVar[str] = (
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
    )
    TIMEOUT_SECONDS: ClassVar[float] = 120.0
    provider: ClassVar[str] = "dashscope"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str,
        endpoint: str | None = None,
        timeout: float = TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = api_key
        self.model = model.strip()
        self.endpoint = endpoint or self.ENDPOINT
        self.timeout = timeout
        if not self.model:
            raise ValueError("DashScope model must not be empty")
        if timeout <= 0:
            raise ValueError("DashScope timeout must be positive")

    async def _complete(self, messages: list[dict[str, Any]]) -> str:
        api_key = self.api_key or os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise StructuredAIProviderError("DASHSCOPE_API_KEY is not configured")
        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise StructuredAIProviderError("DashScope request timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise StructuredAIProviderError(
                f"DashScope returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise StructuredAIProviderError("DashScope request failed") from exc
        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise StructuredAIProviderError(
                "DashScope returned an invalid structured response"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise StructuredAIProviderError("DashScope returned empty content")
        try:
            json.loads(content)
        except json.JSONDecodeError as exc:
            raise StructuredAIProviderError(
                "DashScope returned invalid JSON content"
            ) from exc
        return content.strip()


class DashScopeVisualPlanningClient(_DashScopeStructuredClient):
    """Production complete-story client for the existing VisualDirector."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "qwen-plus",
        endpoint: str | None = None,
        timeout: float = _DashScopeStructuredClient.TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(
            api_key=api_key, model=model, endpoint=endpoint, timeout=timeout
        )

    async def generate(self, prompt: str) -> str:
        return await self._complete([{"role": "user", "content": prompt}])


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
        return await self._complete([{"role": "user", "content": content}])


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
