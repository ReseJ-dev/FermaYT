"""Kimi client for structured whole-story visual planning."""

from __future__ import annotations

import os
from typing import ClassVar

from app.clients.structured_completion import complete_json_chat, missing_key_error


class KimiVisualPlanningClient:
    """Call Kimi's OpenAI-compatible Chat Completions JSON mode."""

    ENDPOINT: ClassVar[str] = "https://api.moonshot.ai/v1/chat/completions"
    TIMEOUT_SECONDS: ClassVar[float] = 600.0
    MAX_ATTEMPTS: ClassVar[int] = 3
    RETRY_BASE_DELAY_SECONDS: ClassVar[float] = 1.0
    MAX_OUTPUT_TOKENS: ClassVar[int] = 32_768
    REASONING_EFFORT: ClassVar[str] = "low"
    provider: ClassVar[str] = "kimi"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "kimi-k3",
        endpoint: str | None = None,
        timeout: float = TIMEOUT_SECONDS,
        max_attempts: int = MAX_ATTEMPTS,
        retry_base_delay: float = RETRY_BASE_DELAY_SECONDS,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        reasoning_effort: str = REASONING_EFFORT,
    ) -> None:
        self.api_key = api_key
        self.model = model.strip()
        self.endpoint = endpoint or self.ENDPOINT
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.retry_base_delay = retry_base_delay
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort.strip().lower()
        if not self.model:
            raise ValueError("Kimi model must not be empty")
        if timeout <= 0:
            raise ValueError("Kimi timeout must be positive")
        if max_attempts < 1:
            raise ValueError("Kimi max_attempts must be positive")
        if retry_base_delay < 0:
            raise ValueError("Kimi retry_base_delay must not be negative")
        if max_output_tokens < 1:
            raise ValueError("Kimi max_output_tokens must be positive")
        if self.reasoning_effort not in {"low", "high", "max"}:
            raise ValueError("Kimi reasoning_effort must be low, high, or max")

    async def generate(self, prompt: str) -> str:
        """Generate one JSON visual plan for the complete narration."""
        api_key = (
            self.api_key
            or os.getenv("MOONSHOT_API_KEY", "").strip()
            or os.getenv("KIMI_API_KEY", "").strip()
        )
        if not api_key:
            raise missing_key_error(
                provider=self.provider,
                model=self.model,
                operation="visual_planning",
                environment_name="MOONSHOT_API_KEY",
            )
        return await complete_json_chat(
            provider=self.provider,
            model=self.model,
            operation="visual_planning",
            endpoint=self.endpoint,
            api_key=api_key,
            messages=[{"role": "user", "content": prompt}],
            timeout=self.timeout,
            max_attempts=self.max_attempts,
            retry_base_delay=self.retry_base_delay,
            max_output_tokens=self.max_output_tokens,
            output_token_parameter="max_completion_tokens",
            reasoning_effort=self.reasoning_effort,
        )
