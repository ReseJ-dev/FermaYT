"""Provider-neutral contracts for asynchronous paid video generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class VideoOperation(str, Enum):
    TEXT_TO_VIDEO = "TEXT_TO_VIDEO"
    IMAGE_TO_VIDEO = "IMAGE_TO_VIDEO"
    FIRST_LAST_TO_VIDEO = "FIRST_LAST_TO_VIDEO"
    REFERENCE_TO_VIDEO = "REFERENCE_TO_VIDEO"
    VIDEO_CONTINUATION = "VIDEO_CONTINUATION"
    VIDEO_EDIT = "VIDEO_EDIT"


class VideoReferenceRole(str, Enum):
    FIRST_FRAME = "FIRST_FRAME"
    LAST_FRAME = "LAST_FRAME"
    STYLE_REFERENCE = "STYLE_REFERENCE"
    MASTER_LOCATION = "MASTER_LOCATION"
    CHARACTER_REFERENCE = "CHARACTER_REFERENCE"
    PREVIOUS_STATE = "PREVIOUS_STATE"
    SOURCE_VIDEO = "SOURCE_VIDEO"


class RemoteVideoTaskState(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class VideoProviderCapabilities:
    supports_text_to_video: bool = False
    supports_image_to_video: bool = False
    supports_first_frame: bool = False
    supports_last_frame: bool = False
    supports_first_last_frame: bool = False
    supports_reference_images: bool = False
    max_reference_images: int = 0
    supports_reference_video: bool = False
    supports_video_continuation: bool = False
    supports_video_editing: bool = False
    supports_audio_generation: bool = False
    supports_disable_audio: bool = False
    supported_durations: tuple[int, ...] = ()
    duration_range: tuple[int, int] | None = None
    supported_resolutions: tuple[str, ...] = ()
    supported_aspect_ratios: tuple[str, ...] = ()
    supports_seed: bool = False
    supports_callback: bool = False
    supports_server_cancel: bool = False
    task_expiration_seconds: int | None = None

    def __post_init__(self) -> None:
        if self.max_reference_images < 0:
            raise ValueError("max_reference_images must not be negative")
        if not self.supports_reference_images and self.max_reference_images:
            raise ValueError("reference limit requires reference-image support")
        if self.duration_range is not None:
            low, high = self.duration_range
            if low < 1 or high < low:
                raise ValueError("duration_range is invalid")

    def supports(self, operation: VideoOperation) -> bool:
        return {
            VideoOperation.TEXT_TO_VIDEO: self.supports_text_to_video,
            VideoOperation.IMAGE_TO_VIDEO: self.supports_image_to_video,
            VideoOperation.FIRST_LAST_TO_VIDEO: self.supports_first_last_frame,
            VideoOperation.REFERENCE_TO_VIDEO: self.supports_reference_images,
            VideoOperation.VIDEO_CONTINUATION: self.supports_video_continuation,
            VideoOperation.VIDEO_EDIT: self.supports_video_editing,
        }[operation]

    def snapshot(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "supported_durations",
            "supported_resolutions",
            "supported_aspect_ratios",
        ):
            value[key] = list(value[key])
        if self.duration_range is not None:
            value["duration_range"] = list(self.duration_range)
        return value


@dataclass(frozen=True, slots=True)
class VideoReference:
    reference_id: str
    role: VideoReferenceRole
    file_path: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.reference_id.strip() or not self.file_path.strip():
            raise ValueError("video reference ID and path must not be empty")
        if len(self.sha256) != 64 or any(
            item not in "0123456789abcdef" for item in self.sha256.casefold()
        ):
            raise ValueError("video reference sha256 must be a hex digest")


@dataclass(frozen=True, slots=True)
class VideoGenerationRequest:
    operation: VideoOperation
    prompt: str
    duration_seconds: int
    resolution: str
    aspect_ratio: str
    references: tuple[VideoReference, ...] = ()
    seed: int | None = None
    disable_audio: bool = True
    callback_url: str | None = None


@dataclass(frozen=True, slots=True)
class VideoSubmission:
    remote_task_id: str
    state: RemoteVideoTaskState = RemoteVideoTaskState.QUEUED
    provider_metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class VideoTaskResult:
    state: RemoteVideoTaskState
    result_url: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    actual_cost: float | None = None
    consumed_credits: float | None = None
    usage: Mapping[str, Any] | None = None
    provider_metadata: Mapping[str, Any] | None = None


@runtime_checkable
class VideoGenerationProvider(Protocol):
    provider_id: str
    model: str
    capabilities: VideoProviderCapabilities

    async def submit(self, request: VideoGenerationRequest) -> VideoSubmission: ...

    async def get_task_status(self, remote_task_id: str) -> VideoTaskResult: ...


@runtime_checkable
class CancellableVideoGenerationProvider(Protocol):
    async def cancel_task(self, remote_task_id: str) -> None: ...


def get_video_provider_capabilities(
    provider: VideoGenerationProvider,
) -> VideoProviderCapabilities:
    capabilities = getattr(provider, "capabilities", None)
    if not isinstance(capabilities, VideoProviderCapabilities):
        raise TypeError("Video provider does not declare validated capabilities")
    return capabilities


def select_video_references(
    operation: VideoOperation,
    references: tuple[VideoReference, ...],
    capabilities: VideoProviderCapabilities,
    *,
    accepted_roles: frozenset[VideoReferenceRole] | None = None,
    ignored_roles: frozenset[VideoReferenceRole] | None = None,
    max_reference_images: int | None = None,
) -> tuple[VideoReference, ...]:
    """Select references by semantic authority, never by caller list position."""
    by_role: dict[VideoReferenceRole, list[VideoReference]] = {}
    for reference in references:
        by_role.setdefault(reference.role, []).append(reference)

    if accepted_roles is not None:
        ignored = ignored_roles or frozenset()
        unsupported = [
            reference.role.value
            for reference in references
            if reference.role not in accepted_roles and reference.role not in ignored
        ]
        if unsupported:
            raise ValueError(
                f"{operation.value} does not support reference role(s): "
                + ", ".join(unsupported)
            )
        accepted_images = [
            reference
            for reference in references
            if reference.role in accepted_roles
            and reference.role is not VideoReferenceRole.SOURCE_VIDEO
        ]
        if max_reference_images is not None and len(accepted_images) > max_reference_images:
            raise ValueError(
                f"{operation.value} accepts at most {max_reference_images} image references"
            )

    mandatory: list[VideoReferenceRole] = []
    if operation is VideoOperation.IMAGE_TO_VIDEO:
        mandatory = [VideoReferenceRole.FIRST_FRAME]
    elif operation is VideoOperation.FIRST_LAST_TO_VIDEO:
        mandatory = [VideoReferenceRole.FIRST_FRAME, VideoReferenceRole.LAST_FRAME]
    elif operation in {VideoOperation.VIDEO_CONTINUATION, VideoOperation.VIDEO_EDIT}:
        mandatory = [VideoReferenceRole.SOURCE_VIDEO]
    selected: list[VideoReference] = []
    for role in mandatory:
        if accepted_roles is not None and role not in accepted_roles:
            raise ValueError(f"{operation.value} profile does not accept {role.value}")
        values = by_role.get(role, [])
        if not values:
            raise ValueError(f"{operation.value} requires {role.value}")
        if len(values) > 1:
            raise ValueError(f"{operation.value} accepts only one {role.value}")
        selected.append(values[0])

    # First/last/source inputs belong to the operation contract. Any extra
    # identity/style images require explicit reference-image capability.
    if accepted_roles is None and not capabilities.supports_reference_images:
        return tuple(selected)

    optional_order = (
        (
            VideoReferenceRole.CHARACTER_REFERENCE,
            VideoReferenceRole.PREVIOUS_STATE,
            VideoReferenceRole.MASTER_LOCATION,
            VideoReferenceRole.STYLE_REFERENCE,
        )
        if operation is VideoOperation.REFERENCE_TO_VIDEO
        else (
            VideoReferenceRole.PREVIOUS_STATE,
            VideoReferenceRole.CHARACTER_REFERENCE,
            VideoReferenceRole.MASTER_LOCATION,
            VideoReferenceRole.STYLE_REFERENCE,
        )
    )
    limit = (
        max_reference_images
        if max_reference_images is not None
        else capabilities.max_reference_images
    )
    for role in optional_order:
        if accepted_roles is not None and role not in accepted_roles:
            continue
        for reference in by_role.get(role, []):
            if reference not in selected:
                selected.append(reference)
            if (
                limit
                and len(
                    [
                        item
                        for item in selected
                        if item.role is not VideoReferenceRole.SOURCE_VIDEO
                    ]
                )
                >= limit
            ):
                break
        if (
            limit
            and len(
                [
                    item
                    for item in selected
                    if item.role is not VideoReferenceRole.SOURCE_VIDEO
                ]
            )
            >= limit
        ):
            break
    if operation is VideoOperation.FIRST_LAST_TO_VIDEO:
        selected = selected[:2]
    return tuple(selected)
