"""Safe structured diagnostics for external image-provider failures."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from typing import Any

import httpx

_MAX_PROVIDER_MESSAGE = 1000
_DATA_URL = re.compile(
    r"data:image/[a-zA-Z0-9.+-]+;base64,[a-zA-Z0-9+/=_-]+",
    re.IGNORECASE,
)
_LONG_BASE64 = re.compile(r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/=_-]{256,}")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(authorization|api[_ -]?key|token|secret)"
    r"(\s*[\"']?\s*[:=]\s*[\"']?)[^\s,;\"']+",
)
_BEARER = re.compile(r"(?i)bearer\s+[^\s,;\"']+")


@dataclass(frozen=True, slots=True)
class ImageProviderDiagnostic:
    provider: str
    model: str | None
    operation: str
    error_type: str
    request_stage: str
    http_status: int | None = None
    provider_error: str | None = None
    master_scene_id: str | None = None
    beat_id: str | None = None

    def with_context(
        self,
        *,
        request_stage: str | None = None,
        master_scene_id: str | None = None,
        beat_id: str | None = None,
    ) -> ImageProviderDiagnostic:
        return replace(
            self,
            request_stage=request_stage or self.request_stage,
            master_scene_id=master_scene_id or self.master_scene_id,
            beat_id=beat_id or self.beat_id,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }

    def format(self, summary: str) -> str:
        values = self.as_dict()
        return "\n".join(
            (
                summary,
                *(
                    f"{key}={json.dumps(value, ensure_ascii=False)}"
                    for key, value in values.items()
                ),
            )
        )


def sanitize_provider_message(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            value = type(value).__name__
    message = _DATA_URL.sub("[redacted-image]", value)
    message = _LONG_BASE64.sub("[redacted-base64]", message)
    message = _BEARER.sub("Bearer [redacted]", message)
    message = _SECRET_ASSIGNMENT.sub(r"\1\2[redacted]", message)
    message = " ".join(message.split())
    if not message:
        return None
    return message[:_MAX_PROVIDER_MESSAGE]


def safe_provider_response(response: httpx.Response) -> str | None:
    """Extract only known diagnostic fields instead of exposing a whole payload."""
    try:
        payload = response.json()
    except ValueError:
        return sanitize_provider_message(response.text)
    extracted = _extract_error_fields(payload)
    return sanitize_provider_message(extracted)


def diagnostic_from_exception(
    error: BaseException,
    *,
    provider: str,
    model: str | None,
    operation: str,
    request_stage: str,
) -> ImageProviderDiagnostic:
    existing = find_image_provider_diagnostic(error)
    if existing is not None:
        return existing
    http_status: int | None = None
    if isinstance(error, httpx.TimeoutException):
        error_type = "timeout"
    elif isinstance(error, httpx.HTTPStatusError):
        error_type = "http"
        http_status = error.response.status_code
    elif isinstance(error, httpx.RequestError):
        error_type = "network"
    else:
        error_type = "provider_validation"
    provider_error = (
        safe_provider_response(error.response)
        if isinstance(error, httpx.HTTPStatusError)
        else sanitize_provider_message(str(error))
    )
    return ImageProviderDiagnostic(
        provider=provider,
        model=model,
        operation=operation,
        error_type=error_type,
        request_stage=request_stage,
        http_status=http_status,
        provider_error=provider_error,
    )


def find_image_provider_diagnostic(
    error: BaseException,
) -> ImageProviderDiagnostic | None:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        diagnostic = getattr(current, "safe_diagnostic", None)
        if isinstance(diagnostic, ImageProviderDiagnostic):
            return diagnostic
        current = current.__cause__ or current.__context__
    return None


def _extract_error_fields(payload: Any) -> object:
    if not isinstance(payload, dict):
        return "Provider returned a non-object error response"
    selected: dict[str, object] = {}
    for key in ("code", "message", "error", "type", "request_id"):
        if key in payload:
            selected[key] = payload[key]
    error = payload.get("error")
    if isinstance(error, dict):
        selected["error"] = {
            key: error[key]
            for key in ("code", "message", "type")
            if key in error
        }
    return selected or "Provider returned an HTTP error without diagnostic fields"
