"""Official asynchronous API adapters for Vidu, Wan, and Seedance."""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx

from app.errors import (
    VIDEO_AUTH_ERROR,
    VIDEO_BAD_REQUEST,
    VIDEO_PROVIDER_UNAVAILABLE,
    VIDEO_RATE_LIMIT,
    VIDEO_SUBMISSION_TIMEOUT_UNKNOWN,
    VideoGenerationError,
)
from app.video_model_profiles import (
    VideoPayloadStrategy,
    get_verified_video_model_profile,
    unsupported_video_model_profile,
)
from app.video_providers import (
    RemoteVideoTaskState,
    VideoGenerationRequest,
    VideoOperation,
    VideoProviderCapabilities,
    VideoReference,
    VideoReferenceRole,
    VideoSubmission,
    VideoTaskResult,
)

_REQUEST_TIMEOUT = 30.0


class _AsyncVideoHTTPProvider:
    provider_id: str
    model: str
    capabilities: VideoProviderCapabilities

    def __init__(self, *, api_key: str | None, model: str, endpoint: str) -> None:
        self.api_key = api_key
        self.model = model.strip()
        self.endpoint = endpoint.rstrip("/")
        if not self.model:
            raise ValueError("video model must not be empty")

    def _key(self, environment_name: str) -> str:
        value = self.api_key or os.getenv(environment_name, "").strip()
        if not value:
            raise VideoGenerationError(
                VIDEO_AUTH_ERROR,
                f"{environment_name} is not configured",
                provider=self.provider_id,
            )
        return value

    def validate_configuration(self) -> None:
        """Fail before a paid submission boundary when credentials are absent."""
        self._headers()

    async def _post_json(
        self, url: str, *, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                response = await client.post(url, headers=headers, json=payload)
            _raise_video_http(response, self.provider_id, submission=True)
            return _json_object(response, self.provider_id)
        except httpx.TimeoutException as exc:
            raise VideoGenerationError(
                VIDEO_SUBMISSION_TIMEOUT_UNKNOWN,
                "Submission timed out after the paid request may have reached the provider",
                provider=self.provider_id,
            ) from exc
        except httpx.RequestError as exc:
            raise VideoGenerationError(
                VIDEO_PROVIDER_UNAVAILABLE,
                "Video provider request failed before a task response was received",
                provider=self.provider_id,
            ) from exc

    async def _get_json(self, url: str, *, headers: dict[str, str]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                response = await client.get(url, headers=headers)
            _raise_video_http(response, self.provider_id, submission=False)
            return _json_object(response, self.provider_id)
        except httpx.RequestError as exc:
            raise VideoGenerationError(
                VIDEO_PROVIDER_UNAVAILABLE,
                "Video task status is temporarily unavailable",
                provider=self.provider_id,
            ) from exc


class ViduVideoProvider(_AsyncVideoHTTPProvider):
    """Vidu Enterprise v2 task API adapter (Q3/Q2 model-configurable)."""

    provider_id = "vidu"
    DEFAULT_ENDPOINT = "https://api.vidu.com/ent/v2"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "viduq3-turbo",
        endpoint: str = DEFAULT_ENDPOINT,
    ) -> None:
        super().__init__(api_key=api_key, model=model, endpoint=endpoint)
        self.model_profile = get_verified_video_model_profile(
            self.provider_id, self.model
        ) or unsupported_video_model_profile(self.provider_id, self.model)
        self.capabilities = self.model_profile.capabilities

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Token {self._key('VIDU_API_KEY')}",
            "Content-Type": "application/json",
        }

    async def submit(self, request: VideoGenerationRequest) -> VideoSubmission:
        operation_profile = self.model_profile.operation_profile(request.operation)
        if operation_profile is None:
            raise _unsupported(self.provider_id, request.operation)
        _validate_profile_references(self.provider_id, request, operation_profile)
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": request.prompt,
            "duration": request.duration_seconds,
            "resolution": request.resolution.lower(),
            "aspect_ratio": request.aspect_ratio,
            "audio": False,
            "bgm": False,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.callback_url:
            payload["callback_url"] = request.callback_url
        if request.references:
            references = request.references
            if operation_profile.payload_strategy is VideoPayloadStrategy.VIDU_IMAGE:
                references = tuple(
                    item
                    for item in references
                    if item.role is VideoReferenceRole.FIRST_FRAME
                )[:1]
            elif (
                operation_profile.payload_strategy
                is VideoPayloadStrategy.VIDU_FIRST_LAST
            ):
                references = tuple(
                    item
                    for role in (
                        VideoReferenceRole.FIRST_FRAME,
                        VideoReferenceRole.LAST_FRAME,
                    )
                    for item in references
                    if item.role is role
                )[:2]
            payload["images"] = [_media_value(item) for item in references]
        data = await self._post_json(
            f"{self.endpoint}/{operation_profile.create_path}",
            headers=self._headers(),
            payload=payload,
        )
        task_id = _required_task_id(data, self.provider_id)
        return VideoSubmission(task_id, _vidu_state(data.get("state")), data)

    async def get_task_status(self, remote_task_id: str) -> VideoTaskResult:
        status_path = _profile_status_path(self.model_profile, remote_task_id)
        data = await self._get_json(
            f"{self.endpoint}/{status_path}",
            headers=self._headers(),
        )
        state = _vidu_state(data.get("state"))
        creations = data.get("creations")
        url = None
        if isinstance(creations, list) and creations and isinstance(creations[0], dict):
            candidate = creations[0].get("url")
            url = candidate if isinstance(candidate, str) else None
        credits = _optional_float(data.get("credits"))
        return VideoTaskResult(
            state,
            result_url=url,
            error_code=_optional_string(data.get("err_code")),
            error_message=_optional_string(data.get("message")),
            consumed_credits=credits,
            usage={"credits": credits} if credits is not None else None,
            provider_metadata=data,
        )

    async def cancel_task(self, remote_task_id: str) -> None:
        if not self.model_profile.cancel_path:
            raise _unsupported(self.provider_id, VideoOperation.IMAGE_TO_VIDEO)
        await self._post_json(
            f"{self.endpoint}/{self.model_profile.cancel_path.format(id=remote_task_id)}",
            headers=self._headers(),
            payload={"id": remote_task_id},
        )


class WanVideoProvider(_AsyncVideoHTTPProvider):
    """Alibaba Model Studio workspace video-synthesis task adapter."""

    provider_id = "wan"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "wan2.7-i2v-2026-04-25",
        endpoint: str | None = None,
        workspace_id: str | None = None,
        region: str = "ap-southeast-1",
    ) -> None:
        configured = endpoint or os.getenv("DASHSCOPE_VIDEO_ENDPOINT", "").strip()
        workspace = workspace_id or os.getenv("DASHSCOPE_WORKSPACE_ID", "").strip()
        self.workspace_endpoint_configured = bool(configured or workspace)
        if not configured:
            if not workspace:
                configured = "https://dashscope-intl.aliyuncs.com/api/v1"
            else:
                configured = f"https://{workspace}.{region}.maas.aliyuncs.com/api/v1"
        super().__init__(api_key=api_key, model=model, endpoint=configured)
        self.model_profile = get_verified_video_model_profile(
            self.provider_id, self.model
        ) or unsupported_video_model_profile(self.provider_id, self.model)
        self.capabilities = self.model_profile.capabilities

    def validate_configuration(self) -> None:
        super().validate_configuration()
        if not self.workspace_endpoint_configured:
            raise VideoGenerationError(
                VIDEO_BAD_REQUEST,
                "DASHSCOPE_WORKSPACE_ID or DASHSCOPE_VIDEO_ENDPOINT is required "
                "for current Wan video models",
                provider=self.provider_id,
            )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._key('DASHSCOPE_API_KEY')}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }

    async def submit(self, request: VideoGenerationRequest) -> VideoSubmission:
        operation_profile = self.model_profile.operation_profile(request.operation)
        if operation_profile is None:
            raise _unsupported(self.provider_id, request.operation)
        _validate_profile_references(self.provider_id, request, operation_profile)
        media = [
            _wan_media(item, operation_profile.payload_strategy)
            for item in request.references
        ]
        input_value: dict[str, Any] = {"prompt": request.prompt}
        if media:
            input_value["media"] = media
        parameters: dict[str, Any] = {
            "resolution": request.resolution.upper(),
            "duration": request.duration_seconds,
            "prompt_extend": True,
            "watermark": False,
        }
        if (
            request.aspect_ratio
            and operation_profile.supported_aspect_ratios
            and request.aspect_ratio in operation_profile.supported_aspect_ratios
        ):
            parameters["ratio"] = request.aspect_ratio
        if request.seed is not None and self.model_profile.supports_seed:
            parameters["seed"] = request.seed
        data = await self._post_json(
            f"{self.endpoint}/{operation_profile.create_path}",
            headers=self._headers(),
            payload={
                "model": self.model,
                "input": input_value,
                "parameters": parameters,
            },
        )
        output = data.get("output") if isinstance(data.get("output"), dict) else data
        task_id = _required_task_id(output, self.provider_id)
        return VideoSubmission(task_id, RemoteVideoTaskState.QUEUED, data)

    async def get_task_status(self, remote_task_id: str) -> VideoTaskResult:
        status_path = _profile_status_path(self.model_profile, remote_task_id)
        data = await self._get_json(
            f"{self.endpoint}/{status_path}", headers=self._headers()
        )
        output = data.get("output") if isinstance(data.get("output"), dict) else {}
        state = _dashscope_state(output.get("task_status"))
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        return VideoTaskResult(
            state,
            result_url=_optional_string(output.get("video_url")),
            error_code=_optional_string(output.get("code")),
            error_message=_optional_string(output.get("message")),
            usage=usage,
            provider_metadata=data,
        )


class SeedanceVideoProvider(_AsyncVideoHTTPProvider):
    """BytePlus ModelArk Contents Generations v3 adapter."""

    provider_id = "seedance"
    DEFAULT_ENDPOINT = "https://ark.ap-southeast.bytepluses.com/api/v3"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "seedance-1-0-lite-i2v-250428",
        endpoint: str | None = None,
    ) -> None:
        configured = endpoint or os.getenv("BYTEPLUS_ENDPOINT", "").strip()
        super().__init__(
            api_key=api_key,
            model=model,
            endpoint=configured or self.DEFAULT_ENDPOINT,
        )
        self.model_profile = get_verified_video_model_profile(
            self.provider_id, self.model
        ) or unsupported_video_model_profile(self.provider_id, self.model)
        self.capabilities = self.model_profile.capabilities

    def _headers(self) -> dict[str, str]:
        key = (
            self.api_key
            or os.getenv("BYTEPLUS_API_KEY", "").strip()
            or os.getenv("BYTEPLUS_ARK_API_KEY", "").strip()
        )
        if not key:
            raise VideoGenerationError(
                VIDEO_AUTH_ERROR,
                "BYTEPLUS_API_KEY is not configured",
                provider=self.provider_id,
            )
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    async def submit(self, request: VideoGenerationRequest) -> VideoSubmission:
        operation_profile = self.model_profile.operation_profile(request.operation)
        if operation_profile is None:
            raise _unsupported(self.provider_id, request.operation)
        _validate_profile_references(self.provider_id, request, operation_profile)
        content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
        for reference in request.references:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _media_value(reference)},
                    "role": _seedance_role(reference.role),
                }
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "content": content,
            "duration": request.duration_seconds,
            "resolution": request.resolution.lower(),
            "ratio": request.aspect_ratio,
            "watermark": False,
        }
        if request.disable_audio and self.capabilities.supports_disable_audio:
            payload["generate_audio"] = False
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.callback_url:
            payload["callback_url"] = request.callback_url
        data = await self._post_json(
            f"{self.endpoint}/{operation_profile.create_path}",
            headers=self._headers(),
            payload=payload,
        )
        task_id = _required_task_id(data, self.provider_id, key="id")
        return VideoSubmission(task_id, _seedance_state(data.get("status")), data)

    async def get_task_status(self, remote_task_id: str) -> VideoTaskResult:
        status_path = _profile_status_path(self.model_profile, remote_task_id)
        data = await self._get_json(
            f"{self.endpoint}/{status_path}",
            headers=self._headers(),
        )
        content = data.get("content")
        url = None
        if isinstance(content, dict):
            url = _optional_string(content.get("video_url"))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    url = _optional_string(item.get("video_url") or item.get("url"))
                    if url:
                        break
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        return VideoTaskResult(
            _seedance_state(data.get("status")),
            result_url=url,
            error_code=_optional_string(data.get("error_code")),
            error_message=_optional_string(data.get("error")),
            usage=usage,
            provider_metadata=data,
        )

    async def cancel_task(self, remote_task_id: str) -> None:
        if not self.model_profile.cancel_path:
            raise _unsupported(self.provider_id, VideoOperation.IMAGE_TO_VIDEO)
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                response = await client.delete(
                    f"{self.endpoint}/{self.model_profile.cancel_path.format(id=remote_task_id)}",
                    headers=self._headers(),
                )
            _raise_video_http(response, self.provider_id, submission=False)
        except httpx.RequestError as exc:
            raise VideoGenerationError(
                VIDEO_PROVIDER_UNAVAILABLE,
                "Could not cancel the remote video task",
                provider=self.provider_id,
                remote_task_id=remote_task_id,
            ) from exc


def _media_value(reference: VideoReference) -> str:
    path = Path(reference.file_path)
    if reference.file_path.startswith(("http://", "https://", "data:")):
        return reference.file_path
    if not path.is_file():
        raise VideoGenerationError(
            VIDEO_BAD_REQUEST, f"Reference file does not exist: {path}"
        )
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _wan_media(
    reference: VideoReference, strategy: VideoPayloadStrategy
) -> dict[str, str]:
    if reference.role is VideoReferenceRole.SOURCE_VIDEO:
        media_type = (
            "first_clip"
            if strategy is VideoPayloadStrategy.WAN_CONTINUATION
            else "video"
            if strategy is VideoPayloadStrategy.WAN_EDIT
            else ""
        )
        if not media_type:
            raise ValueError(f"{strategy.value} does not accept SOURCE_VIDEO")
        media_value = _wan_video_value(reference)
    elif reference.role is VideoReferenceRole.FIRST_FRAME:
        media_type = "first_frame"
        media_value = _media_value(reference)
    elif reference.role is VideoReferenceRole.LAST_FRAME:
        media_type = "last_frame"
        media_value = _media_value(reference)
    else:
        media_type = "reference_image"
        media_value = _media_value(reference)
    return {"type": media_type, "url": media_value}


def _wan_video_value(reference: VideoReference) -> str:
    value = reference.file_path.strip()
    if value.startswith(("http://", "https://", "ark_asset_vid_")):
        return value
    raise VideoGenerationError(
        VIDEO_BAD_REQUEST,
        "Wan SOURCE_VIDEO requires a public URL or ark_asset_vid_ Asset Center ID; "
        "local video Base64 is not supported by the verified API",
        provider="wan",
    )


def _seedance_role(role: VideoReferenceRole) -> str:
    return {
        VideoReferenceRole.FIRST_FRAME: "first_frame",
        VideoReferenceRole.LAST_FRAME: "last_frame",
    }.get(role, "reference_image")


def _required_task_id(
    data: dict[str, Any], provider: str, *, key: str = "task_id"
) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise VideoGenerationError(
            VIDEO_PROVIDER_UNAVAILABLE,
            "Provider response did not contain a remote task ID",
            provider=provider,
        )
    return value.strip()


def _validate_profile_references(
    provider: str, request: VideoGenerationRequest, operation_profile: Any
) -> None:
    accepted = operation_profile.accepted_reference_roles
    ignored = operation_profile.ignored_reference_roles
    unsupported = [
        item.role.value
        for item in request.references
        if item.role not in accepted and item.role not in ignored
    ]
    if unsupported:
        raise VideoGenerationError(
            VIDEO_BAD_REQUEST,
            f"{request.operation.value} does not support reference role(s): "
            + ", ".join(unsupported),
            provider=provider,
        )
    image_count = sum(
        item.role is not VideoReferenceRole.SOURCE_VIDEO
        and item.role in accepted
        for item in request.references
    )
    if (
        operation_profile.max_reference_images
        and image_count > operation_profile.max_reference_images
    ):
        raise VideoGenerationError(
            VIDEO_BAD_REQUEST,
            f"{request.operation.value} accepts at most "
            f"{operation_profile.max_reference_images} image references",
            provider=provider,
        )
    required_roles = {
        VideoOperation.IMAGE_TO_VIDEO: (VideoReferenceRole.FIRST_FRAME,),
        VideoOperation.FIRST_LAST_TO_VIDEO: (
            VideoReferenceRole.FIRST_FRAME,
            VideoReferenceRole.LAST_FRAME,
        ),
        VideoOperation.VIDEO_CONTINUATION: (VideoReferenceRole.SOURCE_VIDEO,),
        VideoOperation.VIDEO_EDIT: (VideoReferenceRole.SOURCE_VIDEO,),
    }.get(request.operation, ())
    for role in required_roles:
        count = sum(item.role is role for item in request.references)
        if count != 1:
            raise VideoGenerationError(
                VIDEO_BAD_REQUEST,
                f"{request.operation.value} requires exactly one {role.value}",
                provider=provider,
            )
    if request.operation is VideoOperation.REFERENCE_TO_VIDEO and not request.references:
        raise VideoGenerationError(
            VIDEO_BAD_REQUEST,
            "REFERENCE_TO_VIDEO requires at least one verified image reference",
            provider=provider,
        )


def _profile_status_path(profile: Any, remote_task_id: str) -> str:
    if not profile.verified or not profile.status_endpoint:
        raise VideoGenerationError(
            VIDEO_BAD_REQUEST,
            "Video task status is unavailable for an unverified model",
            provider=profile.provider_id,
            remote_task_id=remote_task_id,
        )
    return profile.status_endpoint.format(id=remote_task_id)


def _vidu_state(value: object) -> RemoteVideoTaskState:
    return {
        "created": RemoteVideoTaskState.QUEUED,
        "queueing": RemoteVideoTaskState.QUEUED,
        "processing": RemoteVideoTaskState.PROCESSING,
        "success": RemoteVideoTaskState.SUCCEEDED,
        "failed": RemoteVideoTaskState.FAILED,
        "cancelled": RemoteVideoTaskState.CANCELLED,
    }.get(str(value).casefold(), RemoteVideoTaskState.UNKNOWN)


def _dashscope_state(value: object) -> RemoteVideoTaskState:
    return {
        "PENDING": RemoteVideoTaskState.QUEUED,
        "RUNNING": RemoteVideoTaskState.PROCESSING,
        "SUCCEEDED": RemoteVideoTaskState.SUCCEEDED,
        "FAILED": RemoteVideoTaskState.FAILED,
        "CANCELED": RemoteVideoTaskState.CANCELLED,
    }.get(str(value).upper(), RemoteVideoTaskState.UNKNOWN)


def _seedance_state(value: object) -> RemoteVideoTaskState:
    return {
        "queued": RemoteVideoTaskState.QUEUED,
        "running": RemoteVideoTaskState.PROCESSING,
        "succeeded": RemoteVideoTaskState.SUCCEEDED,
        "failed": RemoteVideoTaskState.FAILED,
        "cancelled": RemoteVideoTaskState.CANCELLED,
        "canceled": RemoteVideoTaskState.CANCELLED,
    }.get(str(value).casefold(), RemoteVideoTaskState.UNKNOWN)


def _raise_video_http(
    response: httpx.Response, provider: str, *, submission: bool
) -> None:
    if response.status_code < 400:
        return
    code = (
        VIDEO_AUTH_ERROR
        if response.status_code in {401, 403}
        else VIDEO_RATE_LIMIT
        if response.status_code == 429
        else VIDEO_BAD_REQUEST
        if response.status_code < 500
        else VIDEO_PROVIDER_UNAVAILABLE
    )
    stage = "submission" if submission else "polling"
    raise VideoGenerationError(
        code,
        f"{provider} {stage} returned HTTP {response.status_code}",
        provider=provider,
        diagnostic={"http_status": response.status_code},
    )


def _json_object(response: httpx.Response, provider: str) -> dict[str, Any]:
    try:
        value = response.json()
    except ValueError as exc:
        raise VideoGenerationError(
            VIDEO_PROVIDER_UNAVAILABLE,
            "Video provider returned invalid JSON",
            provider=provider,
        ) from exc
    if not isinstance(value, dict):
        raise VideoGenerationError(
            VIDEO_PROVIDER_UNAVAILABLE,
            "Video provider returned an invalid response object",
            provider=provider,
        )
    return value


def _unsupported(provider: str, operation: VideoOperation) -> VideoGenerationError:
    from app.errors import VIDEO_CAPABILITY_UNSUPPORTED

    return VideoGenerationError(
        VIDEO_CAPABILITY_UNSUPPORTED,
        f"{provider} does not support {operation.value} for the selected model",
        provider=provider,
    )


def _optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
