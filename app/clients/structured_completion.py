"""Safe JSON chat-completion transport with bounded transient retries."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from app.errors import StructuredAIProviderError
from app.provider_diagnostics import (
    StructuredAIProviderDiagnostic,
    safe_provider_response,
    sanitize_provider_message,
)

logger = logging.getLogger(__name__)

_TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504}
_REQUEST_ID_HEADERS = (
    "x-request-id",
    "request-id",
    "x-dashscope-request-id",
)
_PREVIEW_LIMIT = 500


@dataclass(frozen=True, slots=True)
class StructuredCompletionMetadata:
    """Non-secret provider accounting data returned with one completion."""

    provider_request_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


async def complete_json_chat(
    *,
    provider: str,
    model: str,
    operation: str,
    endpoint: str,
    api_key: str,
    messages: list[dict[str, Any]],
    timeout: float,
    max_attempts: int,
    retry_base_delay: float,
    max_output_tokens: int,
    output_token_parameter: str = "max_tokens",
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    thinking: dict[str, str] | None = None,
    on_metadata: Callable[[StructuredCompletionMetadata], None] | None = None,
) -> str:
    """Return strictly parsed JSON content or a diagnostic-rich safe error."""
    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        output_token_parameter: max_output_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if thinking is not None:
        payload["thinking"] = thinking

    for attempt in range(1, max_attempts + 1):
        diagnostic: StructuredAIProviderDiagnostic | None = None
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            diagnostic = _diagnostic(
                provider, model, operation, "PLANNING_TIMEOUT", attempt,
                max_attempts, timeout_seconds=timeout,
                retry_exhausted=attempt >= max_attempts,
            )
            if await _retry_if_transient(diagnostic, attempt, max_attempts, retry_base_delay):
                continue
            raise _provider_error(diagnostic, "Structured provider request timed out") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            diagnostic = _diagnostic(
                provider,
                model,
                operation,
                _http_category(status),
                attempt,
                max_attempts,
                http_status=status,
                provider_error=safe_provider_response(exc.response),
                request_id=_request_id(exc.response),
                retry_exhausted=(
                    attempt >= max_attempts
                    if status in _TRANSIENT_HTTP_STATUSES
                    else None
                ),
            )
            if status in _TRANSIENT_HTTP_STATUSES and await _retry_if_transient(
                diagnostic, attempt, max_attempts, retry_base_delay
            ):
                continue
            raise _provider_error(
                diagnostic, f"Structured provider returned HTTP {status}"
            ) from exc
        except httpx.RequestError as exc:
            diagnostic = _diagnostic(
                provider,
                model,
                operation,
                "PLANNING_NETWORK_ERROR",
                attempt,
                max_attempts,
                provider_error=type(exc).__name__,
                retry_exhausted=attempt >= max_attempts,
            )
            if await _retry_if_transient(diagnostic, attempt, max_attempts, retry_base_delay):
                continue
            raise _provider_error(diagnostic, "Structured provider request failed") from exc

        content, metadata = _parse_response(
            response,
            provider=provider,
            model=model,
            operation=operation,
            attempt=attempt,
            max_attempts=max_attempts,
        )
        if on_metadata is not None:
            on_metadata(metadata)
        return content

    raise AssertionError("structured completion retry loop did not terminate")


def missing_key_error(
    *,
    provider: str,
    model: str,
    operation: str,
    environment_name: str,
) -> StructuredAIProviderError:
    diagnostic = _diagnostic(
        provider, model, operation, "PLANNING_AUTH_ERROR", 0, 1,
        provider_error=f"{environment_name} is not configured",
    )
    return _provider_error(diagnostic, f"{environment_name} is not configured")


def _parse_response(
    response: httpx.Response,
    *,
    provider: str,
    model: str,
    operation: str,
    attempt: int,
    max_attempts: int,
) -> tuple[str, StructuredCompletionMetadata]:
    request_id = _request_id(response)
    try:
        body = response.json()
    except ValueError as exc:
        preview = _safe_preview(response.text)
        diagnostic = _diagnostic(
            provider, model, operation, "PLANNING_INVALID_JSON", attempt,
            max_attempts, request_id=request_id,
            response_length=len(response.content), response_preview=preview,
        )
        raise _provider_error(diagnostic, "Structured provider returned malformed response JSON") from exc

    try:
        choice = body["choices"][0]
        message = choice["message"]
        content = message["content"]
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError) as exc:
        diagnostic = _diagnostic(
            provider, model, operation,
            "PLANNING_STRUCTURED_JSON_INCOMPATIBILITY", attempt, max_attempts,
            provider_error=safe_provider_response(response), request_id=request_id,
            response_length=len(response.content),
        )
        raise _provider_error(diagnostic, "Structured provider response has an incompatible shape") from exc

    if not isinstance(content, str) or not content.strip():
        diagnostic = _diagnostic(
            provider, model, operation, "PLANNING_EMPTY_RESPONSE", attempt,
            max_attempts, request_id=request_id,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        )
        raise _provider_error(diagnostic, "Structured provider returned empty content")

    normalized = content.strip()
    try:
        json.loads(normalized)
    except json.JSONDecodeError as exc:
        truncated = _looks_truncated(exc, normalized, finish_reason)
        diagnostic = _diagnostic(
            provider,
            model,
            operation,
            "PLANNING_TRUNCATED_OUTPUT" if truncated else "PLANNING_INVALID_JSON",
            attempt,
            max_attempts,
            request_id=request_id,
            response_length=len(normalized),
            response_preview=_safe_preview(normalized),
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        )
        summary = (
            "Structured provider output was truncated"
            if truncated
            else "Structured provider returned invalid JSON content"
        )
        raise _provider_error(diagnostic, summary) from exc
    usage = body.get("usage") if isinstance(body, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    return normalized, StructuredCompletionMetadata(
        provider_request_id=request_id or _string_value(body.get("id")),
        input_tokens=_integer_value(
            usage.get("prompt_tokens", usage.get("input_tokens"))
        ),
        output_tokens=_integer_value(
            usage.get("completion_tokens", usage.get("output_tokens"))
        ),
        total_tokens=_integer_value(usage.get("total_tokens")),
    )


def _integer_value(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


async def _retry_if_transient(
    diagnostic: StructuredAIProviderDiagnostic,
    attempt: int,
    max_attempts: int,
    base_delay: float,
) -> bool:
    if attempt >= max_attempts:
        return False
    delay = base_delay * (2 ** (attempt - 1))
    logger.warning(
        "%s\nretry_delay_seconds=%s",
        diagnostic.format("Transient structured-provider failure; retrying"),
        delay,
    )
    await asyncio.sleep(delay)
    return True


def _http_category(status: int) -> str:
    if status in {401, 403}:
        return "PLANNING_AUTH_ERROR"
    if status == 400:
        return "PLANNING_BAD_REQUEST"
    if status == 404:
        return "PLANNING_NOT_FOUND"
    if status == 402:
        return "PLANNING_PAYMENT_REQUIRED"
    if status == 429:
        return "PLANNING_RATE_LIMIT"
    if 500 <= status <= 599:
        return "PLANNING_PROVIDER_5XX"
    return "PLANNING_UNKNOWN_ERROR"


def _diagnostic(
    provider: str,
    model: str,
    operation: str,
    category: str,
    attempt: int,
    max_attempts: int,
    **kwargs: object,
) -> StructuredAIProviderDiagnostic:
    return StructuredAIProviderDiagnostic(
        provider=provider,
        model=model,
        operation=operation,
        category=category,
        attempt=attempt,
        max_attempts=max_attempts,
        **kwargs,  # type: ignore[arg-type]
    )


def _provider_error(
    diagnostic: StructuredAIProviderDiagnostic,
    summary: str,
) -> StructuredAIProviderError:
    return StructuredAIProviderError(
        diagnostic.format(summary),
        diagnostic=diagnostic,
        user_summary="Visual planning provider failed",
    )


def _request_id(response: httpx.Response) -> str | None:
    for name in _REQUEST_ID_HEADERS:
        value = response.headers.get(name)
        if value:
            return sanitize_provider_message(value)
    try:
        body = response.json()
    except ValueError:
        return None
    if isinstance(body, dict):
        value = body.get("request_id") or body.get("requestId")
        return sanitize_provider_message(value)
    return None


def _safe_preview(value: str) -> str | None:
    return sanitize_provider_message(value[:_PREVIEW_LIMIT])


def _looks_truncated(
    error: json.JSONDecodeError,
    content: str,
    finish_reason: object,
) -> bool:
    if finish_reason in {"length", "max_tokens", "max_output_tokens"}:
        return True
    near_end = error.pos >= max(len(content) - 4, 0)
    looks_like_json = content.lstrip().startswith(("{", "["))
    eof_message = any(
        marker in error.msg.lower()
        for marker in ("unterminated", "expecting value", "expecting ',' delimiter")
    )
    return looks_like_json and near_end and eof_message
