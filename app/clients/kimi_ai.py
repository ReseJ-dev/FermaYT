"""Kimi client for structured whole-story visual planning."""

from __future__ import annotations

import os
from typing import ClassVar

import httpx

from app.clients.structured_completion import (
    StructuredCompletionMetadata,
    complete_json_chat,
    missing_key_error,
)
from app.errors import StructuredAIProviderError
from app.provider_diagnostics import (
    StructuredAIProviderDiagnostic,
    safe_provider_response,
)


class KimiVisualPlanningClient:
    """Call Kimi's OpenAI-compatible Chat Completions JSON mode."""

    ENDPOINT: ClassVar[str] = "https://api.moonshot.ai/v1/chat/completions"
    TIMEOUT_SECONDS: ClassVar[float] = 600.0
    MAX_ATTEMPTS: ClassVar[int] = 1
    RETRY_BASE_DELAY_SECONDS: ClassVar[float] = 0.0
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
        self.configured_max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort.strip().lower()
        self.last_completion_metadata: StructuredCompletionMetadata | None = None
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
        self.last_completion_metadata = None
        return await complete_json_chat(
            provider=self.provider,
            model=self.model,
            operation="visual_planning",
            endpoint=self.endpoint,
            api_key=api_key,
            messages=[{"role": "user", "content": prompt}],
            timeout=self.timeout,
            # A planning POST may already be executing when the connection fails.
            # It must never be silently resubmitted by the transport.
            max_attempts=1,
            retry_base_delay=0,
            max_output_tokens=self.max_output_tokens,
            output_token_parameter="max_completion_tokens",
            # K3 uses reasoning_effort. K2.6 keeps thinking enabled by default
            # and rejects the K3-only reasoning_effort request field.
            reasoning_effort=(
                self.reasoning_effort if self.model == "kimi-k3" else None
            ),
            # K2.6 enables thinking by default. Structured planning needs the
            # bounded completion budget for the final JSON rather than hidden
            # reasoning that can exhaust it before content is emitted.
            thinking=({"type": "disabled"} if self.model == "kimi-k2.6" else None),
            on_metadata=self._capture_metadata,
        )

    def _capture_metadata(self, metadata: StructuredCompletionMetadata) -> None:
        self.last_completion_metadata = metadata

    async def preflight(self) -> None:
        """Verify key/model availability without creating a paid completion."""
        api_key = (
            self.api_key
            or os.getenv("MOONSHOT_API_KEY", "").strip()
            or os.getenv("KIMI_API_KEY", "").strip()
        )
        if not api_key:
            raise missing_key_error(
                provider=self.provider,
                model=self.model,
                operation="visual_planning_preflight",
                environment_name="MOONSHOT_API_KEY",
            )
        try:
            async with httpx.AsyncClient(timeout=min(self.timeout, 30.0)) as client:
                response = await client.get(
                    self._models_endpoint(),
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise self._preflight_error("PLANNING_TIMEOUT") from exc
        except httpx.HTTPStatusError as exc:
            categories = {
                400: "PLANNING_BAD_REQUEST",
                401: "PLANNING_AUTH_ERROR",
                402: "PLANNING_PAYMENT_REQUIRED",
                403: "PLANNING_AUTH_ERROR",
                404: "PLANNING_NOT_FOUND",
                429: "PLANNING_RATE_LIMIT",
            }
            raise self._preflight_error(
                categories.get(exc.response.status_code, "PLANNING_PROVIDER_5XX"),
                http_status=exc.response.status_code,
                provider_error=safe_provider_response(exc.response),
            ) from exc
        except httpx.RequestError as exc:
            raise self._preflight_error("PLANNING_NETWORK_ERROR") from exc
        try:
            body = response.json()
            available = {
                item.get("id") for item in body["data"] if isinstance(item, dict)
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise self._preflight_error("PLANNING_INVALID_JSON") from exc
        if self.model not in available:
            raise self._preflight_error("PLANNING_NOT_FOUND")

    def _models_endpoint(self) -> str:
        suffix = "/chat/completions"
        return (
            f"{self.endpoint[: -len(suffix)]}/models"
            if self.endpoint.endswith(suffix)
            else "https://api.moonshot.ai/v1/models"
        )

    def _preflight_error(
        self,
        category: str,
        *,
        http_status: int | None = None,
        provider_error: str | None = None,
    ) -> StructuredAIProviderError:
        diagnostic = StructuredAIProviderDiagnostic(
            provider=self.provider,
            model=self.model,
            operation="visual_planning_preflight",
            category=category,
            attempt=1,
            max_attempts=1,
            http_status=http_status,
            provider_error=provider_error,
        )
        return StructuredAIProviderError(
            "Kimi API preflight failed",
            diagnostic=diagnostic,
            user_summary="Kimi API preflight failed",
        )
