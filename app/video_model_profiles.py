"""Verified, fail-closed video provider/model capability profiles.

Only exact provider/model pairs listed here may authorize a paid operation.  The
registry deliberately contains no model-name heuristics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from app.video_providers import (
    VideoOperation,
    VideoProviderCapabilities,
    VideoReferenceRole,
)


class VideoPayloadStrategy(str, Enum):
    VIDU_TEXT = "VIDU_TEXT"
    VIDU_IMAGE = "VIDU_IMAGE"
    VIDU_FIRST_LAST = "VIDU_FIRST_LAST"
    VIDU_REFERENCE_IMAGES = "VIDU_REFERENCE_IMAGES"
    VIDU_REFERENCE_SUBJECTS = "VIDU_REFERENCE_SUBJECTS"
    WAN_TEXT = "WAN_TEXT"
    WAN_IMAGE = "WAN_IMAGE"
    WAN_FIRST_LAST = "WAN_FIRST_LAST"
    WAN_CONTINUATION = "WAN_CONTINUATION"
    WAN_EDIT = "WAN_EDIT"
    SEEDANCE_TEXT = "SEEDANCE_TEXT"
    SEEDANCE_IMAGE = "SEEDANCE_IMAGE"
    SEEDANCE_FIRST_LAST = "SEEDANCE_FIRST_LAST"
    SEEDANCE_REFERENCE_IMAGES = "SEEDANCE_REFERENCE_IMAGES"


@dataclass(frozen=True, slots=True)
class VideoOperationProfile:
    operation: VideoOperation
    create_path: str
    payload_strategy: VideoPayloadStrategy
    supported_durations: tuple[int, ...] = ()
    duration_range: tuple[int, int] | None = None
    supported_resolutions: tuple[str, ...] = ()
    supported_aspect_ratios: tuple[str, ...] = ()
    accepted_reference_roles: frozenset[VideoReferenceRole] = frozenset()
    ignored_reference_roles: frozenset[VideoReferenceRole] = frozenset()
    max_reference_images: int = 0

    def reference_role_behavior(self, role: VideoReferenceRole) -> str:
        if role in self.accepted_reference_roles:
            return "SUPPORTED_AND_MAPPED"
        if role in self.ignored_reference_roles:
            return "IGNORED_BY_DESIGN"
        return "UNSUPPORTED_ERROR"


@dataclass(frozen=True, slots=True)
class ProviderModelProfile:
    provider_id: str
    model_id: str
    operations: Mapping[VideoOperation, VideoOperationProfile]
    status_endpoint: str
    official_documentation: tuple[str, ...]
    verified_on: str
    supports_audio_generation: bool = False
    supports_disable_audio: bool = False
    supports_seed: bool = False
    supports_callback: bool = False
    supports_server_cancel: bool = False
    cancel_path: str | None = None
    task_expiration_seconds: int | None = None
    verified: bool = True

    def operation_profile(
        self, operation: VideoOperation
    ) -> VideoOperationProfile | None:
        return self.operations.get(operation)

    def snapshot(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "verified": self.verified,
            "verified_on": self.verified_on,
            "official_documentation": list(self.official_documentation),
            "status_endpoint": self.status_endpoint,
            "cancel_path": self.cancel_path,
            "operations": {
                operation.value: {
                    "create_path": item.create_path,
                    "payload_strategy": item.payload_strategy.value,
                    "supported_durations": list(item.supported_durations),
                    "duration_range": (
                        list(item.duration_range)
                        if item.duration_range is not None
                        else None
                    ),
                    "supported_resolutions": list(item.supported_resolutions),
                    "supported_aspect_ratios": list(item.supported_aspect_ratios),
                    "accepted_reference_roles": sorted(
                        role.value for role in item.accepted_reference_roles
                    ),
                    "reference_role_behavior": {
                        role.value: item.reference_role_behavior(role)
                        for role in VideoReferenceRole
                    },
                    "max_reference_images": item.max_reference_images,
                }
                for operation, item in self.operations.items()
            },
        }

    @property
    def capabilities(self) -> VideoProviderCapabilities:
        operations = self.operations
        reference_profile = operations.get(VideoOperation.REFERENCE_TO_VIDEO)
        durations = {
            duration
            for profile in operations.values()
            for duration in profile.supported_durations
        }
        ranges = {
            profile.duration_range
            for profile in operations.values()
            if profile.duration_range is not None
        }
        resolutions = {
            value
            for profile in operations.values()
            for value in profile.supported_resolutions
        }
        ratios = {
            value
            for profile in operations.values()
            for value in profile.supported_aspect_ratios
        }
        return VideoProviderCapabilities(
            supports_text_to_video=VideoOperation.TEXT_TO_VIDEO in operations,
            supports_image_to_video=VideoOperation.IMAGE_TO_VIDEO in operations,
            supports_first_frame=any(
                VideoReferenceRole.FIRST_FRAME in profile.accepted_reference_roles
                for profile in operations.values()
            ),
            supports_last_frame=any(
                VideoReferenceRole.LAST_FRAME in profile.accepted_reference_roles
                for profile in operations.values()
            ),
            supports_first_last_frame=(
                VideoOperation.FIRST_LAST_TO_VIDEO in operations
            ),
            supports_reference_images=reference_profile is not None,
            max_reference_images=(
                reference_profile.max_reference_images
                if reference_profile is not None
                else 0
            ),
            supports_reference_video=any(
                VideoReferenceRole.SOURCE_VIDEO in profile.accepted_reference_roles
                for profile in operations.values()
            ),
            supports_video_continuation=(
                VideoOperation.VIDEO_CONTINUATION in operations
            ),
            supports_video_editing=VideoOperation.VIDEO_EDIT in operations,
            supports_audio_generation=self.supports_audio_generation,
            supports_disable_audio=self.supports_disable_audio,
            supported_durations=tuple(sorted(durations)) if not ranges else (),
            duration_range=next(iter(ranges)) if len(ranges) == 1 else None,
            supported_resolutions=tuple(sorted(resolutions)),
            supported_aspect_ratios=tuple(sorted(ratios)),
            supports_seed=self.supports_seed,
            supports_callback=self.supports_callback,
            supports_server_cancel=self.supports_server_cancel,
            task_expiration_seconds=self.task_expiration_seconds,
        )


_SEMANTIC_IMAGE_ROLES = frozenset(
    {
        VideoReferenceRole.STYLE_REFERENCE,
        VideoReferenceRole.MASTER_LOCATION,
        VideoReferenceRole.CHARACTER_REFERENCE,
        VideoReferenceRole.PREVIOUS_STATE,
    }
)
_FIRST = frozenset({VideoReferenceRole.FIRST_FRAME})
_FIRST_LAST = frozenset(
    {VideoReferenceRole.FIRST_FRAME, VideoReferenceRole.LAST_FRAME}
)
_SOURCE_VIDEO = frozenset({VideoReferenceRole.SOURCE_VIDEO})
_EDIT_REFERENCES = frozenset({VideoReferenceRole.SOURCE_VIDEO}) | _SEMANTIC_IMAGE_ROLES
_RATIOS = ("16:9", "9:16", "4:3", "3:4", "1:1")
_SEEDANCE_RATIOS = ("16:9", "9:16", "1:1", "adaptive")


def _operation(
    operation: VideoOperation,
    path: str,
    strategy: VideoPayloadStrategy,
    *,
    durations: tuple[int, ...] = (),
    duration_range: tuple[int, int] | None = None,
    resolutions: tuple[str, ...],
    aspect_ratios: tuple[str, ...] = (),
    roles: frozenset[VideoReferenceRole] = frozenset(),
    max_references: int = 0,
) -> VideoOperationProfile:
    return VideoOperationProfile(
        operation=operation,
        create_path=path,
        payload_strategy=strategy,
        supported_durations=durations,
        duration_range=duration_range,
        supported_resolutions=resolutions,
        supported_aspect_ratios=aspect_ratios,
        accepted_reference_roles=roles,
        max_reference_images=max_references,
    )


def _profile(
    provider: str,
    model: str,
    operations: tuple[VideoOperationProfile, ...],
    *,
    status_endpoint: str,
    documentation: tuple[str, ...],
    **kwargs: object,
) -> ProviderModelProfile:
    return ProviderModelProfile(
        provider_id=provider,
        model_id=model,
        operations=MappingProxyType({item.operation: item for item in operations}),
        status_endpoint=status_endpoint,
        official_documentation=documentation,
        verified_on="2026-09-18",
        **kwargs,  # type: ignore[arg-type]
    )


_VIDU_DOCS = (
    "https://platform.vidu.com/docs/text-to-video",
    "https://platform.vidu.com/docs/image-to-video",
    "https://platform.vidu.com/docs/start-end-to-video",
    "https://platform.vidu.com/docs/reference-to-video",
    "https://platform.vidu.com/docs/get-generation",
    "https://platform.vidu.com/docs/cancel-generation",
)
_WAN_I2V_DOCS = (
    "https://www.alibabacloud.com/help/en/model-studio/image-to-video-general-api-reference",
)
_WAN_T2V_DOCS = (
    "https://www.alibabacloud.com/help/en/model-studio/text-to-video-api-reference",
)
_SEEDANCE_API_DOCS = (
    "https://docs.byteplus.com/en/docs/ModelArk/1520757",
    "https://docs.byteplus.com/en/docs/ModelArk/1521309",
    "https://docs.byteplus.com/en/docs/ModelArk/1521720",
)


MODEL_PROFILES: Mapping[tuple[str, str], ProviderModelProfile] = MappingProxyType(
    {
        ("vidu", "viduq3-turbo"): _profile(
            "vidu",
            "viduq3-turbo",
            (
                _operation(
                    VideoOperation.TEXT_TO_VIDEO,
                    "text2video",
                    VideoPayloadStrategy.VIDU_TEXT,
                    duration_range=(1, 16),
                    resolutions=("540p", "720p", "1080p"),
                    aspect_ratios=_RATIOS,
                ),
                _operation(
                    VideoOperation.IMAGE_TO_VIDEO,
                    "img2video",
                    VideoPayloadStrategy.VIDU_IMAGE,
                    duration_range=(1, 16),
                    resolutions=("540p", "720p", "1080p"),
                    aspect_ratios=_RATIOS,
                    roles=_FIRST,
                    max_references=1,
                ),
                _operation(
                    VideoOperation.FIRST_LAST_TO_VIDEO,
                    "start-end2video",
                    VideoPayloadStrategy.VIDU_FIRST_LAST,
                    duration_range=(1, 16),
                    resolutions=("540p", "720p", "1080p"),
                    aspect_ratios=_RATIOS,
                    roles=_FIRST_LAST,
                    max_references=2,
                ),
                _operation(
                    VideoOperation.REFERENCE_TO_VIDEO,
                    "reference2video",
                    VideoPayloadStrategy.VIDU_REFERENCE_IMAGES,
                    duration_range=(3, 16),
                    resolutions=("540p", "720p", "1080p"),
                    aspect_ratios=_RATIOS,
                    roles=_SEMANTIC_IMAGE_ROLES,
                    max_references=7,
                ),
            ),
            status_endpoint="tasks/{id}/creations",
            documentation=_VIDU_DOCS,
            supports_audio_generation=True,
            supports_disable_audio=True,
            supports_seed=True,
            supports_callback=True,
            supports_server_cancel=True,
            cancel_path="tasks/{id}/cancel",
        ),
        ("wan", "wan2.7-i2v-2026-04-25"): _profile(
            "wan",
            "wan2.7-i2v-2026-04-25",
            (
                _operation(
                    VideoOperation.IMAGE_TO_VIDEO,
                    "services/aigc/video-generation/video-synthesis",
                    VideoPayloadStrategy.WAN_IMAGE,
                    duration_range=(2, 15),
                    resolutions=("720P", "1080P"),
                    roles=_FIRST,
                    max_references=1,
                ),
                _operation(
                    VideoOperation.FIRST_LAST_TO_VIDEO,
                    "services/aigc/video-generation/video-synthesis",
                    VideoPayloadStrategy.WAN_FIRST_LAST,
                    duration_range=(2, 15),
                    resolutions=("720P", "1080P"),
                    roles=_FIRST_LAST,
                    max_references=2,
                ),
                _operation(
                    VideoOperation.VIDEO_CONTINUATION,
                    "services/aigc/video-generation/video-synthesis",
                    VideoPayloadStrategy.WAN_CONTINUATION,
                    duration_range=(2, 15),
                    resolutions=("720P", "1080P"),
                    roles=_SOURCE_VIDEO,
                ),
            ),
            status_endpoint="tasks/{id}",
            documentation=_WAN_I2V_DOCS,
            supports_audio_generation=True,
            supports_seed=True,
            task_expiration_seconds=86_400,
        ),
        ("wan", "wan2.7-t2v-2026-06-12"): _profile(
            "wan",
            "wan2.7-t2v-2026-06-12",
            (
                _operation(
                    VideoOperation.TEXT_TO_VIDEO,
                    "services/aigc/video-generation/video-synthesis",
                    VideoPayloadStrategy.WAN_TEXT,
                    duration_range=(2, 15),
                    resolutions=("720P", "1080P"),
                    aspect_ratios=_RATIOS,
                ),
            ),
            status_endpoint="tasks/{id}",
            documentation=_WAN_T2V_DOCS,
            supports_audio_generation=True,
            supports_seed=True,
            task_expiration_seconds=86_400,
        ),
        ("wan", "wan2.7-videoedit"): _profile(
            "wan",
            "wan2.7-videoedit",
            (
                _operation(
                    VideoOperation.VIDEO_EDIT,
                    "services/aigc/video-generation/video-synthesis",
                    VideoPayloadStrategy.WAN_EDIT,
                    duration_range=(2, 10),
                    resolutions=("720P", "1080P"),
                    aspect_ratios=_RATIOS,
                    roles=_EDIT_REFERENCES,
                    max_references=4,
                ),
            ),
            status_endpoint="tasks/{id}",
            documentation=(
                "https://www.alibabacloud.com/help/en/model-studio/wan-video-editing-api-reference",
            ),
            task_expiration_seconds=86_400,
        ),
        ("seedance", "seedance-1-0-lite-i2v-250428"): _profile(
            "seedance",
            "seedance-1-0-lite-i2v-250428",
            (
                _operation(
                    VideoOperation.IMAGE_TO_VIDEO,
                    "contents/generations/tasks",
                    VideoPayloadStrategy.SEEDANCE_IMAGE,
                    durations=(5, 10),
                    resolutions=("480p", "720p", "1080p"),
                    aspect_ratios=_SEEDANCE_RATIOS,
                    roles=_FIRST,
                    max_references=1,
                ),
                _operation(
                    VideoOperation.FIRST_LAST_TO_VIDEO,
                    "contents/generations/tasks",
                    VideoPayloadStrategy.SEEDANCE_FIRST_LAST,
                    durations=(5, 10),
                    resolutions=("480p", "720p", "1080p"),
                    aspect_ratios=_SEEDANCE_RATIOS,
                    roles=_FIRST_LAST,
                    max_references=2,
                ),
                _operation(
                    VideoOperation.REFERENCE_TO_VIDEO,
                    "contents/generations/tasks",
                    VideoPayloadStrategy.SEEDANCE_REFERENCE_IMAGES,
                    durations=(5, 10),
                    resolutions=("480p", "720p", "1080p"),
                    aspect_ratios=_SEEDANCE_RATIOS,
                    roles=_SEMANTIC_IMAGE_ROLES,
                    max_references=4,
                ),
            ),
            status_endpoint="contents/generations/tasks/{id}",
            documentation=(
                "https://docs.byteplus.com/en/docs/ModelArk/1553576",
                *_SEEDANCE_API_DOCS,
            ),
            supports_seed=True,
            supports_callback=True,
            supports_server_cancel=True,
            cancel_path="contents/generations/tasks/{id}",
        ),
        ("seedance", "seedance-1-0-pro-250528"): _profile(
            "seedance",
            "seedance-1-0-pro-250528",
            (
                _operation(
                    VideoOperation.TEXT_TO_VIDEO,
                    "contents/generations/tasks",
                    VideoPayloadStrategy.SEEDANCE_TEXT,
                    durations=(5, 10),
                    resolutions=("480p", "720p", "1080p"),
                    aspect_ratios=_SEEDANCE_RATIOS,
                ),
                _operation(
                    VideoOperation.IMAGE_TO_VIDEO,
                    "contents/generations/tasks",
                    VideoPayloadStrategy.SEEDANCE_IMAGE,
                    durations=(5, 10),
                    resolutions=("480p", "720p", "1080p"),
                    aspect_ratios=_SEEDANCE_RATIOS,
                    roles=_FIRST,
                    max_references=1,
                ),
                _operation(
                    VideoOperation.FIRST_LAST_TO_VIDEO,
                    "contents/generations/tasks",
                    VideoPayloadStrategy.SEEDANCE_FIRST_LAST,
                    durations=(5, 10),
                    resolutions=("480p", "720p", "1080p"),
                    aspect_ratios=_SEEDANCE_RATIOS,
                    roles=_FIRST_LAST,
                    max_references=2,
                ),
            ),
            status_endpoint="contents/generations/tasks/{id}",
            documentation=(
                "https://docs.byteplus.com/en/docs/ModelArk/1587798",
                *_SEEDANCE_API_DOCS,
            ),
            supports_seed=True,
            supports_callback=True,
            supports_server_cancel=True,
            cancel_path="contents/generations/tasks/{id}",
        ),
        ("seedance", "seedance-1-0-pro-fast-251015"): _profile(
            "seedance",
            "seedance-1-0-pro-fast-251015",
            (
                _operation(
                    VideoOperation.TEXT_TO_VIDEO,
                    "contents/generations/tasks",
                    VideoPayloadStrategy.SEEDANCE_TEXT,
                    durations=(5, 10),
                    resolutions=("480p", "720p", "1080p"),
                    aspect_ratios=_SEEDANCE_RATIOS,
                ),
                _operation(
                    VideoOperation.IMAGE_TO_VIDEO,
                    "contents/generations/tasks",
                    VideoPayloadStrategy.SEEDANCE_IMAGE,
                    durations=(5, 10),
                    resolutions=("480p", "720p", "1080p"),
                    aspect_ratios=_SEEDANCE_RATIOS,
                    roles=_FIRST,
                    max_references=1,
                ),
            ),
            status_endpoint="contents/generations/tasks/{id}",
            documentation=(
                "https://docs.byteplus.com/en/docs/ModelArk/1901652",
                *_SEEDANCE_API_DOCS,
            ),
            supports_seed=True,
            supports_callback=True,
            supports_server_cancel=True,
            cancel_path="contents/generations/tasks/{id}",
        ),
    }
)


def get_verified_video_model_profile(
    provider_id: str, model_id: str
) -> ProviderModelProfile | None:
    """Return an exact verified profile; unknown IDs intentionally return None."""

    return MODEL_PROFILES.get((provider_id.strip().casefold(), model_id.strip()))


def unsupported_video_model_profile(
    provider_id: str, model_id: str
) -> ProviderModelProfile:
    """Describe an unknown model without authorizing any paid capability."""

    return ProviderModelProfile(
        provider_id=provider_id.strip().casefold(),
        model_id=model_id.strip(),
        operations=MappingProxyType({}),
        status_endpoint="",
        official_documentation=(),
        verified_on="",
        verified=False,
    )
