"""Kimi client for structured whole-story visual planning."""

from __future__ import annotations

import json
import os
from typing import ClassVar

import httpx

from app.errors import StructuredAIProviderError


class KimiVisualPlanningClient:
    """Call Kimi's OpenAI-compatible Chat Completions JSON mode."""

    ENDPOINT: ClassVar[str] = "https://api.moonshot.ai/v1/chat/completions"
    TIMEOUT_SECONDS: ClassVar[float] = 300.0
    provider: ClassVar[str] = "kimi"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "kimi-k3",
        endpoint: str | None = None,
        timeout: float = TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = api_key
        self.model = model.strip()
        self.endpoint = endpoint or self.ENDPOINT
        self.timeout = timeout
        if not self.model:
            raise ValueError("Kimi model must not be empty")
        if timeout <= 0:
            raise ValueError("Kimi timeout must be positive")

    async def generate(self, prompt: str) -> str:
        """Generate one JSON visual plan for the complete narration."""
        api_key = (
            self.api_key
            or os.getenv("MOONSHOT_API_KEY", "").strip()
            or os.getenv("KIMI_API_KEY", "").strip()
        )
        if not api_key:
            raise StructuredAIProviderError("MOONSHOT_API_KEY is not configured")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
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
            raise StructuredAIProviderError("Kimi request timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise StructuredAIProviderError(
                f"Kimi returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise StructuredAIProviderError("Kimi request failed") from exc
        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise StructuredAIProviderError(
                "Kimi returned an invalid structured response"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise StructuredAIProviderError("Kimi returned empty content")
        try:
            json.loads(content)
        except json.JSONDecodeError as exc:
            raise StructuredAIProviderError("Kimi returned invalid JSON content") from exc
        return content.strip()
