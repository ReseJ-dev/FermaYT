"""Crash-safe asynchronous video task lifecycle and immutable local assets."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.costs import (
    CostCertainty,
    PricingUnit,
    UsageStatus,
    find_pricing,
    record_provider_usage,
)
from app.errors import (
    VIDEO_BUDGET_EXCEEDED,
    VIDEO_CAPABILITY_UNSUPPORTED,
    VIDEO_DOWNLOAD_FAILED,
    VIDEO_POLL_TIMEOUT,
    VIDEO_SUBMISSION_TIMEOUT_UNKNOWN,
    VIDEO_TASK_FAILED,
    VIDEO_VALIDATION_FAILED,
    VideoGenerationError,
)
from app.generators.video_prompt import VideoPromptBuilder
from app.media.probe import MediaProbeResult, probe_media
from app.persistence import (
    BeatVisualResult,
    GeneratedVideoAsset,
    Project,
    VideoGenerationAttempt,
)
from app.providers import get_video_provider
from app.services.visual_planning import require_current_project_visual_plan
from app.storage import ProjectMediaPaths
from app.video_providers import (
    RemoteVideoTaskState,
    VideoGenerationProvider,
    VideoGenerationRequest,
    VideoOperation,
    VideoReference,
    VideoReferenceRole,
    get_video_provider_capabilities,
    select_video_references,
)

VideoProviderResolver = Callable[[str, dict[str, Any]], VideoGenerationProvider]
VideoDownloader = Callable[[str, Path], Awaitable[Path]]
VideoProber = Callable[[str | Path], MediaProbeResult]

ACTIVE_VIDEO_STATUSES = frozenset(
    {"SUBMITTED", "QUEUED", "PROCESSING", "REMOTE_STATUS_UNKNOWN"}
)


async def generate_video_from_accepted_still(
    session: Session,
    project_id: str,
    beat_id: str,
    *,
    prompt_override: str | None = None,
    provider_resolver: VideoProviderResolver = get_video_provider,
    projects_root: str | Path = "data/projects",
    job_id: str | None = None,
    poll_interval: float = 5.0,
    max_poll_duration: float = 600.0,
    downloader: VideoDownloader | None = None,
    prober: VideoProber = probe_media,
) -> GeneratedVideoAsset:
    project = session.get(Project, project_id)
    if project is None:
        raise ValueError("Project not found")
    if project.video_generation_mode == "OFF":
        raise VideoGenerationError(
            VIDEO_CAPABILITY_UNSUPPORTED,
            "Video generation is OFF for this project",
        )
    plan = require_current_project_visual_plan(session, project_id).plan
    beat = next((item for item in plan.visual_beats if item.id == beat_id), None)
    if beat is None:
        raise ValueError("Visual beat not found")
    accepted = session.scalar(
        select(BeatVisualResult)
        .where(
            BeatVisualResult.project_id == project_id,
            BeatVisualResult.beat_id == beat_id,
            BeatVisualResult.is_accepted.is_(True),
            BeatVisualResult.generation_status == "SUCCEEDED",
        )
        .order_by(BeatVisualResult.created_at.desc())
        .limit(1)
    )
    if accepted is None or not accepted.output_path or not accepted.file_sha256:
        raise ValueError("Generate Video requires an accepted still image")
    source = Path(accepted.output_path)
    if not source.is_file() or _sha256(source) != accepted.file_sha256:
        raise ValueError("Accepted still lineage is missing or changed")
    provider = provider_resolver(
        project.video_provider,
        _project_video_provider_config(project),
    )
    prompt = VideoPromptBuilder().build_from_beat(
        plan,
        beat,
        manual_motion_prompt=prompt_override,
    )
    reference = VideoReference(
        reference_id=f"accepted:{accepted.id}",
        role=VideoReferenceRole.FIRST_FRAME,
        file_path=accepted.output_path,
        sha256=accepted.file_sha256,
    )
    request = VideoGenerationRequest(
        operation=VideoOperation.IMAGE_TO_VIDEO,
        prompt=prompt,
        duration_seconds=project.video_clip_duration,
        resolution=project.video_resolution,
        aspect_ratio=_aspect_ratio(project.width, project.height),
        references=(reference,),
        disable_audio=True,
    )
    return await execute_video_generation(
        session,
        project,
        provider,
        request,
        beat_id=beat_id,
        accepted_image_result_id=accepted.id,
        projects_root=projects_root,
        job_id=job_id or str(uuid4()),
        poll_interval=poll_interval,
        max_poll_duration=max_poll_duration,
        downloader=downloader,
        prober=prober,
    )


async def execute_video_generation(
    session: Session,
    project: Project,
    provider: VideoGenerationProvider,
    request: VideoGenerationRequest,
    *,
    beat_id: str | None,
    accepted_image_result_id: str | None,
    projects_root: str | Path,
    job_id: str,
    poll_interval: float = 5.0,
    max_poll_duration: float = 600.0,
    downloader: VideoDownloader | None = None,
    prober: VideoProber = probe_media,
) -> GeneratedVideoAsset:
    if poll_interval < 0 or max_poll_duration <= 0:
        raise ValueError("video polling limits are invalid")
    capabilities = get_video_provider_capabilities(provider)
    if not capabilities.supports(request.operation):
        raise VideoGenerationError(
            VIDEO_CAPABILITY_UNSUPPORTED,
            f"Selected {provider.provider_id}/{provider.model} does not support "
            f"{request.operation.value}",
            provider=provider.provider_id,
        )
    model_profile = getattr(provider, "model_profile", None)
    operation_profile = (
        model_profile.operation_profile(request.operation)
        if model_profile is not None
        else None
    )
    references = select_video_references(
        request.operation,
        request.references,
        capabilities,
        accepted_roles=(
            operation_profile.accepted_reference_roles
            if operation_profile is not None
            else None
        ),
        ignored_roles=(
            operation_profile.ignored_reference_roles
            if operation_profile is not None
            else None
        ),
        max_reference_images=(
            operation_profile.max_reference_images
            if operation_profile is not None
            else None
        ),
    )
    request = VideoGenerationRequest(
        operation=request.operation,
        prompt=request.prompt,
        duration_seconds=request.duration_seconds,
        resolution=request.resolution,
        aspect_ratio=request.aspect_ratio,
        references=references,
        seed=request.seed,
        disable_audio=request.disable_audio,
        callback_url=request.callback_url,
    )
    _validate_request(request, capabilities, operation_profile)
    configuration_validator = getattr(provider, "validate_configuration", None)
    if callable(configuration_validator):
        configuration_validator()
    request_hash = _request_hash(provider, request, project.id, beat_id)
    existing = session.scalar(
        select(VideoGenerationAttempt).where(
            VideoGenerationAttempt.request_hash == request_hash
        )
    )
    if existing is not None:
        if existing.asset is not None:
            return existing.asset
        if existing.status == "SUBMISSION_STATUS_UNKNOWN":
            raise VideoGenerationError(
                VIDEO_SUBMISSION_TIMEOUT_UNKNOWN,
                "The original submission may have been billed; explicit reconciliation is required",
                provider=existing.provider,
            )
        if existing.remote_task_id:
            return await resume_video_attempt(
                session,
                existing,
                provider,
                projects_root=projects_root,
                poll_interval=poll_interval,
                max_poll_duration=max_poll_duration,
                downloader=downloader,
                prober=prober,
            )
        raise VideoGenerationError(
            VIDEO_SUBMISSION_TIMEOUT_UNKNOWN,
            "A pre-existing attempt has no authoritative remote task ID",
            provider=existing.provider,
        )

    estimate, currency = _estimate_request(session, provider, request)
    _check_video_budget(session, project, estimate, currency)
    snapshot = _request_snapshot(request)
    capability_snapshot = capabilities.snapshot()
    if model_profile is not None:
        capability_snapshot["model_profile"] = model_profile.snapshot()
    attempt = VideoGenerationAttempt(
        project_id=project.id,
        job_id=job_id,
        beat_id=beat_id,
        accepted_image_result_id=accepted_image_result_id,
        provider=provider.provider_id,
        model=provider.model,
        operation=request.operation.value,
        capability_snapshot=capability_snapshot,
        request_hash=request_hash,
        request_snapshot=snapshot,
        prompt=request.prompt,
        reference_snapshot=snapshot["references"],
        source_image_path=next(
            (
                item.file_path
                for item in references
                if item.role is VideoReferenceRole.FIRST_FRAME
            ),
            None,
        ),
        source_video_path=next(
            (
                item.file_path
                for item in references
                if item.role is VideoReferenceRole.SOURCE_VIDEO
            ),
            None,
        ),
        requested_duration=request.duration_seconds,
        requested_resolution=request.resolution,
        requested_aspect_ratio=request.aspect_ratio,
        status="PENDING_SUBMISSION",
        estimated_cost=estimate,
        currency=currency,
        cost_certainty="ESTIMATED" if estimate is not None else "UNKNOWN",
    )
    session.add(attempt)
    session.commit()
    session.refresh(attempt)

    attempt.submission_started_at = datetime.now(UTC)
    session.commit()  # The paid POST must never precede this durable boundary.
    try:
        submission = await provider.submit(request)
    except VideoGenerationError as exc:
        if exc.code == VIDEO_SUBMISSION_TIMEOUT_UNKNOWN:
            attempt.status = "SUBMISSION_STATUS_UNKNOWN"
            attempt.cost_certainty = "UNKNOWN"
        else:
            attempt.status = "FAILED"
        attempt.error_code = exc.code
        attempt.error_message = exc.user_summary
        session.commit()
        raise
    attempt.remote_task_id = submission.remote_task_id
    attempt.status = "SUBMITTED"
    attempt.submitted_at = datetime.now(UTC)
    attempt.provider_metadata = dict(submission.provider_metadata or {})
    session.commit()  # Persist authoritative task ID before the first poll.
    return await resume_video_attempt(
        session,
        attempt,
        provider,
        projects_root=projects_root,
        poll_interval=poll_interval,
        max_poll_duration=max_poll_duration,
        downloader=downloader,
        prober=prober,
    )


async def resume_video_attempt(
    session: Session,
    attempt: VideoGenerationAttempt,
    provider: VideoGenerationProvider,
    *,
    projects_root: str | Path = "data/projects",
    poll_interval: float = 5.0,
    max_poll_duration: float = 600.0,
    downloader: VideoDownloader | None = None,
    prober: VideoProber = probe_media,
) -> GeneratedVideoAsset:
    if attempt.asset is not None:
        return attempt.asset
    if not attempt.remote_task_id:
        raise VideoGenerationError(
            VIDEO_SUBMISSION_TIMEOUT_UNKNOWN,
            "Cannot resume without an authoritative remote task ID",
            provider=attempt.provider,
        )
    if provider.provider_id != attempt.provider or provider.model != attempt.model:
        raise ValueError("Resume provider/model does not match persisted attempt")
    deadline = time.monotonic() + max_poll_duration
    result = None
    while time.monotonic() <= deadline:
        try:
            result = await provider.get_task_status(attempt.remote_task_id)
        except VideoGenerationError as exc:
            attempt.status = "REMOTE_STATUS_UNKNOWN"
            attempt.last_polled_at = datetime.now(UTC)
            attempt.error_code = exc.code
            attempt.error_message = exc.user_summary
            session.commit()
            raise
        attempt.last_polled_at = datetime.now(UTC)
        attempt.provider_metadata = dict(result.provider_metadata or {})
        attempt.usage_snapshot = dict(result.usage or {})
        if result.state is RemoteVideoTaskState.SUCCEEDED:
            attempt.status = "SUCCEEDED"
            attempt.remote_result_url = result.result_url
            session.commit()
            break
        if result.state in {
            RemoteVideoTaskState.FAILED,
            RemoteVideoTaskState.CANCELLED,
        }:
            attempt.status = (
                "FAILED"
                if result.state is RemoteVideoTaskState.FAILED
                else "CANCELLED_LOCAL"
            )
            attempt.error_code = result.error_code or VIDEO_TASK_FAILED
            attempt.error_message = result.error_message or "Remote video task failed"
            attempt.cost_certainty = "UNKNOWN"
            session.commit()
            raise VideoGenerationError(
                VIDEO_TASK_FAILED,
                attempt.error_message,
                provider=attempt.provider,
                remote_task_id=attempt.remote_task_id,
            )
        attempt.status = (
            "PROCESSING"
            if result.state is RemoteVideoTaskState.PROCESSING
            else "QUEUED"
        )
        session.commit()
        if poll_interval:
            await asyncio.sleep(poll_interval)
    else:
        attempt.status = "REMOTE_STATUS_UNKNOWN"
        attempt.error_code = VIDEO_POLL_TIMEOUT
        attempt.error_message = "Polling timed out; the remote task was not resubmitted"
        session.commit()
        raise VideoGenerationError(
            VIDEO_POLL_TIMEOUT,
            attempt.error_message,
            provider=attempt.provider,
            remote_task_id=attempt.remote_task_id,
        )
    assert result is not None
    if not result.result_url:
        attempt.status = "DOWNLOAD_FAILED"
        attempt.error_code = VIDEO_DOWNLOAD_FAILED
        attempt.error_message = "Successful task did not return a video URL"
        session.commit()
        raise VideoGenerationError(VIDEO_DOWNLOAD_FAILED, attempt.error_message)
    destination = ProjectMediaPaths(
        attempt.project_id, projects_root
    ).generated_video_path(attempt.id)
    try:
        local_path = await (downloader or _download_video)(
            result.result_url, destination
        )
    except Exception as exc:
        attempt.status = "DOWNLOAD_FAILED"
        attempt.error_code = VIDEO_DOWNLOAD_FAILED
        attempt.error_message = "Could not persist the temporary provider video URL"
        session.commit()
        raise VideoGenerationError(
            VIDEO_DOWNLOAD_FAILED, attempt.error_message
        ) from exc
    try:
        metadata = prober(local_path)
        _validate_downloaded_video(metadata, attempt.requested_duration)
    except Exception as exc:
        attempt.status = "VALIDATION_FAILED"
        attempt.error_code = VIDEO_VALIDATION_FAILED
        attempt.error_message = str(exc)[:1000]
        session.commit()
        raise VideoGenerationError(
            VIDEO_VALIDATION_FAILED, "Downloaded provider video is invalid"
        ) from exc
    digest = _sha256(local_path)
    asset = GeneratedVideoAsset(
        project_id=attempt.project_id,
        attempt_id=attempt.id,
        beat_id=attempt.beat_id,
        provider=attempt.provider,
        model=attempt.model,
        operation=attempt.operation,
        file_path=str(local_path),
        file_sha256=digest,
        duration=metadata.duration,
        width=metadata.width or 0,
        height=metadata.height or 0,
        fps=metadata.fps or 0,
        codec=metadata.codec,
        provider_has_audio=metadata.has_audio,
        mute_audio_default=True,
        source_lineage={
            "accepted_image_result_id": attempt.accepted_image_result_id,
            "references": attempt.reference_snapshot,
            "remote_task_id": attempt.remote_task_id,
        },
    )
    session.add(asset)
    attempt.output_path = str(local_path)
    attempt.output_sha256 = digest
    attempt.duration = metadata.duration
    attempt.width = metadata.width
    attempt.height = metadata.height
    attempt.fps = metadata.fps
    attempt.codec = metadata.codec
    attempt.provider_has_audio = metadata.has_audio
    attempt.completed_at = datetime.now(UTC)
    attempt.status = "SUCCEEDED"
    reconciled_cost, reconciled_currency = _reconcile_actual_cost(
        session, attempt, result
    )
    if reconciled_cost is not None:
        attempt.actual_cost = reconciled_cost
        attempt.currency = reconciled_currency or attempt.currency
        attempt.cost_certainty = "ACTUAL"
    session.commit()
    session.refresh(asset)
    _record_video_usage(session, attempt, result)
    return asset


async def resume_incomplete_video_attempts(
    session: Session,
    *,
    provider_resolver: VideoProviderResolver = get_video_provider,
    projects_root: str | Path = "data/projects",
    poll_interval: float = 5.0,
    max_poll_duration: float = 600.0,
) -> list[GeneratedVideoAsset]:
    attempts = list(
        session.scalars(
            select(VideoGenerationAttempt).where(
                VideoGenerationAttempt.status.in_(ACTIVE_VIDEO_STATUSES),
                VideoGenerationAttempt.remote_task_id.is_not(None),
            )
        )
    )
    assets: list[GeneratedVideoAsset] = []
    for attempt in attempts:
        provider = provider_resolver(
            attempt.provider,
            {"model": attempt.model},
        )
        try:
            assets.append(
                await resume_video_attempt(
                    session,
                    attempt,
                    provider,
                    projects_root=projects_root,
                    poll_interval=poll_interval,
                    max_poll_duration=max_poll_duration,
                )
            )
        except VideoGenerationError:
            continue
    return assets


def _validate_request(
    request: VideoGenerationRequest,
    capabilities: Any,
    operation_profile: Any | None = None,
) -> None:
    if not request.prompt.strip():
        raise ValueError("video prompt must not be empty")
    if (
        request.operation is VideoOperation.REFERENCE_TO_VIDEO
        and not request.references
    ):
        raise VideoGenerationError(
            VIDEO_CAPABILITY_UNSUPPORTED,
            "Reference-to-video requires at least one verified image reference",
        )
    durations = (
        operation_profile.supported_durations
        if operation_profile is not None
        else capabilities.supported_durations
    )
    duration_range = (
        operation_profile.duration_range
        if operation_profile is not None
        else capabilities.duration_range
    )
    resolutions = (
        operation_profile.supported_resolutions
        if operation_profile is not None
        else capabilities.supported_resolutions
    )
    aspect_ratios = (
        operation_profile.supported_aspect_ratios
        if operation_profile is not None
        else capabilities.supported_aspect_ratios
    )
    if (
        durations
        and request.duration_seconds not in durations
    ):
        raise VideoGenerationError(
            VIDEO_CAPABILITY_UNSUPPORTED, "Requested duration is unsupported"
        )
    if duration_range is not None:
        low, high = duration_range
        if not low <= request.duration_seconds <= high:
            raise VideoGenerationError(
                VIDEO_CAPABILITY_UNSUPPORTED, "Requested duration is unsupported"
            )
    if (
        resolutions
        and request.resolution not in resolutions
        and request.resolution.upper() not in resolutions
    ):
        raise VideoGenerationError(
            VIDEO_CAPABILITY_UNSUPPORTED, "Requested resolution is unsupported"
        )
    if aspect_ratios and request.aspect_ratio not in aspect_ratios:
        raise VideoGenerationError(
            VIDEO_CAPABILITY_UNSUPPORTED, "Requested aspect ratio is unsupported"
        )


def _validate_downloaded_video(metadata: MediaProbeResult, requested: int) -> None:
    if not metadata.has_video or metadata.duration <= 0:
        raise ValueError("MP4 has no readable video stream")
    if not metadata.width or not metadata.height or not metadata.fps:
        raise ValueError("MP4 video metadata is incomplete")
    if metadata.frame_count == 0:
        raise ValueError("MP4 contains zero frames")
    if (
        metadata.duration < min(0.5, requested * 0.25)
        or metadata.duration > requested * 2 + 1
    ):
        raise ValueError("MP4 duration is outside the safe requested range")


async def _download_video(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{uuid4()}.partial")
    try:
        async with (
            httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client,
            client.stream("GET", url) as response,
        ):
            response.raise_for_status()
            with partial.open("wb") as handle:
                async for chunk in response.aiter_bytes():
                    handle.write(chunk)
        if partial.stat().st_size <= 0:
            raise OSError("downloaded video is empty")
        partial.replace(destination)
        return destination
    finally:
        partial.unlink(missing_ok=True)


def _estimate_request(
    session: Session,
    provider: VideoGenerationProvider,
    request: VideoGenerationRequest,
) -> tuple[float | None, str | None]:
    pricing = find_pricing(
        session,
        provider.provider_id,
        provider.model,
        request.operation.value,
        PricingUnit.PER_SECOND,
    )
    if pricing is None:
        return None, None
    return round(float(pricing.price) * request.duration_seconds, 8), pricing.currency


def _check_video_budget(
    session: Session,
    project: Project,
    next_cost: float | None,
    currency: str | None,
) -> None:
    if project.video_budget_amount is None:
        return
    if next_cost is None or currency is None:
        raise VideoGenerationError(
            VIDEO_BUDGET_EXCEEDED,
            "Video price is not configured; a bounded paid submission cannot be made",
        )
    if currency != project.generation_budget_currency:
        raise VideoGenerationError(
            VIDEO_BUDGET_EXCEEDED,
            "Video pricing currency does not match project budget",
        )
    attempts = list(
        session.scalars(
            select(VideoGenerationAttempt).where(
                VideoGenerationAttempt.project_id == project.id,
                VideoGenerationAttempt.submission_started_at.is_not(None),
            )
        )
    )
    exposure = sum(
        float(
            item.actual_cost
            if item.actual_cost is not None
            else item.estimated_cost or 0
        )
        for item in attempts
    )
    if exposure + next_cost > float(project.video_budget_amount) + 1e-9:
        raise VideoGenerationError(
            VIDEO_BUDGET_EXCEEDED,
            "Estimated video request would exceed the project video budget",
            diagnostic={
                "spent_or_reserved": exposure,
                "next_estimated_cost": next_cost,
                "budget": float(project.video_budget_amount),
            },
        )


def _record_video_usage(
    session: Session, attempt: VideoGenerationAttempt, result: Any
) -> None:
    record_provider_usage(
        session,
        project_id=attempt.project_id,
        job_id=attempt.job_id,
        pipeline_stage="VIDEO_GENERATION",
        provider=attempt.provider,
        model=attempt.model,
        operation=attempt.operation,
        request_revision=attempt.request_hash,
        unit_type=PricingUnit.PER_SECOND,
        output_units=float(attempt.duration or attempt.requested_duration),
        status=UsageStatus.SUCCEEDED,
        beat_id=attempt.beat_id,
        actual_cost=(
            float(attempt.actual_cost) if attempt.actual_cost is not None else None
        ),
        cost_certainty=CostCertainty(attempt.cost_certainty),
    )


def _reconcile_actual_cost(
    session: Session, attempt: VideoGenerationAttempt, result: Any
) -> tuple[float | None, str | None]:
    if result.actual_cost is not None:
        return float(result.actual_cost), attempt.currency
    if result.consumed_credits is None:
        return None, attempt.currency
    pricing = find_pricing(
        session,
        attempt.provider,
        attempt.model,
        attempt.operation,
        PricingUnit.PER_CREDIT,
    )
    if pricing is None:
        return None, attempt.currency
    return (
        round(float(result.consumed_credits) * float(pricing.price), 8),
        pricing.currency,
    )


def _request_snapshot(request: VideoGenerationRequest) -> dict[str, Any]:
    return {
        "operation": request.operation.value,
        "prompt": request.prompt,
        "duration_seconds": request.duration_seconds,
        "resolution": request.resolution,
        "aspect_ratio": request.aspect_ratio,
        "seed": request.seed,
        "disable_audio": request.disable_audio,
        "references": [
            {
                "reference_id": item.reference_id,
                "role": item.role.value,
                "file_path": item.file_path,
                "sha256": item.sha256,
                "provider_order": index,
            }
            for index, item in enumerate(request.references)
        ],
    }


def _request_hash(
    provider: VideoGenerationProvider,
    request: VideoGenerationRequest,
    project_id: str,
    beat_id: str | None,
) -> str:
    payload = {
        "project_id": project_id,
        "beat_id": beat_id,
        "provider": provider.provider_id,
        "model": provider.model,
        "request": _request_snapshot(request),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _project_video_provider_config(project: Project) -> dict[str, Any]:
    return {"model": project.video_model or _default_model(project.video_provider)}


def _default_model(provider: str) -> str:
    return {
        "vidu": "viduq3-turbo",
        "wan": "wan2.7-i2v-2026-04-25",
        "seedance": "seedance-1-0-lite-i2v-250428",
    }[provider]


def _aspect_ratio(width: int, height: int) -> str:
    return "16:9" if width >= height else "9:16"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
