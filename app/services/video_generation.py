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
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from sqlalchemy import case, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.costs import (
    CostCertainty,
    PricingUnit,
    UsageStatus,
    find_pricing,
    record_provider_usage,
)
from app.errors import (
    VIDEO_AUTH_ERROR,
    VIDEO_BUDGET_EXCEEDED,
    VIDEO_CAPABILITY_UNSUPPORTED,
    VIDEO_DOWNLOAD_FAILED,
    VIDEO_POLL_TIMEOUT,
    VIDEO_PRICE_UNKNOWN,
    VIDEO_RECOVERY_CONFIGURATION_MISSING,
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
    {
        "SUBMITTED",
        "QUEUED",
        "PROCESSING",
        "REMOTE_STATUS_UNKNOWN",
        "REMOTE_SUCCEEDED",
    }
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
    execution_context = _provider_execution_context(provider)
    request_hash = _request_hash(
        provider,
        request,
        project.id,
        beat_id,
        execution_context,
    )
    _acquire_video_budget_transaction_lock(session, project.id)
    existing = session.scalar(
        select(VideoGenerationAttempt).where(
            VideoGenerationAttempt.request_hash == request_hash
        )
    )
    if existing is not None:
        session.commit()
        return await _reuse_existing_attempt(
            session,
            existing,
            provider,
            projects_root=projects_root,
            poll_interval=poll_interval,
            max_poll_duration=max_poll_duration,
            downloader=downloader,
            prober=prober,
        )

    estimate, currency = _estimate_request(session, provider, request)
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
        provider_execution_context=execution_context,
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
    try:
        with session.no_autoflush:
            _reserve_video_exposure(
                session,
                project,
                estimate=estimate,
                currency=currency,
            )
        session.commit()
    except VideoGenerationError:
        session.rollback()
        raise
    except IntegrityError:
        # Another worker won the unique request-hash race. The loser attaches
        # to the durable attempt and must never submit another paid task.
        session.rollback()
        existing = session.scalar(
            select(VideoGenerationAttempt).where(
                VideoGenerationAttempt.request_hash == request_hash
            )
        )
        if existing is None:
            raise
        return await _reuse_existing_attempt(
            session,
            existing,
            provider,
            projects_root=projects_root,
            poll_interval=poll_interval,
            max_poll_duration=max_poll_duration,
            downloader=downloader,
            prober=prober,
        )
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
    _require_matching_execution_context(session, attempt, provider)
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
            # Remote completion and durable local completion are distinct.
            attempt.status = "REMOTE_SUCCEEDED"
            attempt.remote_result_url = result.result_url
            _apply_provider_cost_reconciliation(session, attempt, result)
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
            if not _apply_provider_cost_reconciliation(session, attempt, result):
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
    downloader: VideoDownloader | None = None,
    prober: VideoProber = probe_media,
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
        try:
            recovery_config = _recovery_provider_config(session, attempt)
            provider = provider_resolver(
                attempt.provider,
                recovery_config,
            )
            configuration_validator = getattr(provider, "validate_configuration", None)
            if callable(configuration_validator):
                configuration_validator()
            assets.append(
                await resume_video_attempt(
                    session,
                    attempt,
                    provider,
                    projects_root=projects_root,
                    poll_interval=poll_interval,
                    max_poll_duration=max_poll_duration,
                    downloader=downloader,
                    prober=prober,
                )
            )
        except (TypeError, ValueError) as exc:
            _mark_recovery_configuration_missing(
                session,
                attempt,
                f"Persisted video provider configuration cannot be restored: {exc}",
            )
            continue
        except VideoGenerationError as exc:
            if exc.code == VIDEO_AUTH_ERROR:
                _mark_recovery_configuration_missing(
                    session,
                    attempt,
                    "Credentials required for exact video task recovery are unavailable",
                )
            continue
    return assets


async def _reuse_existing_attempt(
    session: Session,
    attempt: VideoGenerationAttempt,
    provider: VideoGenerationProvider,
    *,
    projects_root: str | Path,
    poll_interval: float,
    max_poll_duration: float,
    downloader: VideoDownloader | None,
    prober: VideoProber,
) -> GeneratedVideoAsset:
    if attempt.asset is not None:
        return attempt.asset
    if attempt.status == "RECOVERY_CONFIGURATION_MISSING":
        raise VideoGenerationError(
            VIDEO_RECOVERY_CONFIGURATION_MISSING,
            attempt.error_message or "Video recovery configuration is incomplete",
            provider=attempt.provider,
            remote_task_id=attempt.remote_task_id,
        )
    if attempt.status == "SUBMISSION_STATUS_UNKNOWN":
        raise VideoGenerationError(
            VIDEO_SUBMISSION_TIMEOUT_UNKNOWN,
            "The original submission may have been billed; explicit reconciliation is required",
            provider=attempt.provider,
        )
    if attempt.remote_task_id:
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
    raise VideoGenerationError(
        VIDEO_SUBMISSION_TIMEOUT_UNKNOWN,
        "A pre-existing attempt has no authoritative remote task ID",
        provider=attempt.provider,
    )


def _provider_execution_context(
    provider: VideoGenerationProvider,
) -> dict[str, Any]:
    snapshotter = getattr(provider, "execution_context_snapshot", None)
    if callable(snapshotter):
        context = snapshotter()
    else:
        # Protocol-only providers are useful for local/test integrations. They
        # remain resumable in-process, but persisted restart recovery fails
        # closed unless they expose an exact endpoint-bearing snapshot.
        context = {
            "schema_version": "video_provider_execution_context_v1",
            "provider_id": provider.provider_id,
            "model_id": provider.model,
            "api_variant": "protocol_only",
            "provider_config": {"model": provider.model},
        }
    _validate_execution_context(context, provider.provider_id, provider.model)
    return json.loads(json.dumps(context, sort_keys=True))


def _validate_execution_context(
    context: Any,
    provider_id: str,
    model: str,
) -> None:
    if not isinstance(context, dict):
        raise TypeError("provider execution context must be an object")
    if context.get("schema_version") != "video_provider_execution_context_v1":
        raise ValueError("provider execution context schema is unsupported")
    if context.get("provider_id") != provider_id or context.get("model_id") != model:
        raise ValueError("provider execution context identity does not match")
    provider_config = context.get("provider_config")
    if not isinstance(provider_config, dict) or provider_config.get("model") != model:
        raise ValueError("provider execution context is incomplete")
    _reject_secret_context_keys(context)
    endpoint = context.get("endpoint")
    if endpoint is not None:
        if not isinstance(endpoint, str):
            raise ValueError("provider endpoint must be a string")
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("provider endpoint is not a safe base URL")
        if provider_config.get("endpoint") != endpoint:
            raise ValueError("provider endpoint snapshot is inconsistent")


def _reject_secret_context_keys(value: Any) -> None:
    forbidden = ("api_key", "authorization", "bearer", "secret", "password", "token")
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).lower()
            if any(marker in normalized for marker in forbidden):
                raise ValueError("provider execution context contains a secret field")
            _reject_secret_context_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_secret_context_keys(nested)


def _recovery_provider_config(
    session: Session,
    attempt: VideoGenerationAttempt,
) -> dict[str, Any]:
    context = attempt.provider_execution_context
    try:
        _validate_execution_context(context, attempt.provider, attempt.model)
        assert isinstance(context, dict)
        endpoint = context.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("persisted provider endpoint is missing")
        stored_config = context["provider_config"]
        allowed = {
            "vidu": ("model", "endpoint"),
            "wan": ("model", "endpoint", "workspace_id", "region"),
            "seedance": ("model", "endpoint"),
        }.get(attempt.provider)
        if allowed is None:
            raise ValueError("persisted provider adapter is unsupported")
        config = {key: stored_config[key] for key in allowed if key in stored_config}
        if config.get("model") != attempt.model or config.get("endpoint") != endpoint:
            raise ValueError("persisted provider configuration is inconsistent")
        if attempt.provider == "wan" and not config.get("region"):
            raise ValueError("persisted Wan region is missing")
        return config
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        _mark_recovery_configuration_missing(
            session,
            attempt,
            f"Exact video recovery configuration is unavailable: {exc}",
        )
        raise VideoGenerationError(
            VIDEO_RECOVERY_CONFIGURATION_MISSING,
            attempt.error_message
            or "Exact video recovery configuration is unavailable",
            provider=attempt.provider,
            remote_task_id=attempt.remote_task_id,
        ) from exc


def _require_matching_execution_context(
    session: Session,
    attempt: VideoGenerationAttempt,
    provider: VideoGenerationProvider,
) -> None:
    try:
        current = _provider_execution_context(provider)
        persisted = attempt.provider_execution_context
        _validate_execution_context(persisted, attempt.provider, attempt.model)
        if current != persisted:
            raise ValueError(
                "runtime provider context differs from the persisted context"
            )
    except (TypeError, ValueError) as exc:
        _mark_recovery_configuration_missing(
            session,
            attempt,
            f"Exact video recovery configuration does not match: {exc}",
        )
        raise VideoGenerationError(
            VIDEO_RECOVERY_CONFIGURATION_MISSING,
            attempt.error_message
            or "Exact video recovery configuration does not match",
            provider=attempt.provider,
            remote_task_id=attempt.remote_task_id,
        ) from exc


def _mark_recovery_configuration_missing(
    session: Session,
    attempt: VideoGenerationAttempt,
    message: str,
) -> None:
    attempt.status = "RECOVERY_CONFIGURATION_MISSING"
    attempt.error_code = VIDEO_RECOVERY_CONFIGURATION_MISSING
    attempt.error_message = message[:1000]
    session.commit()


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
    if durations and request.duration_seconds not in durations:
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


def _reserve_video_exposure(
    session: Session,
    project: Project,
    *,
    estimate: float | None,
    currency: str | None,
) -> None:
    if estimate is None or currency is None:
        if not project.allow_unpriced_video_requests:
            raise VideoGenerationError(
                VIDEO_PRICE_UNKNOWN,
                "Video pricing is unavailable; explicitly allow unpriced video "
                "requests before paid submission",
                provider=project.video_provider,
            )
        result = session.execute(
            update(Project)
            .where(Project.id == project.id)
            .values(
                video_unknown_exposure_count=(Project.video_unknown_exposure_count + 1)
            )
        )
        if result.rowcount != 1:
            raise ValueError("Project not found while reserving video exposure")
        return

    amount = func.coalesce(Project.video_cost_exposure_amount, 0)
    conditions = [
        Project.id == project.id,
        (
            Project.video_cost_exposure_currency.is_(None)
            | (Project.video_cost_exposure_currency == currency)
        ),
        (
            Project.video_budget_amount.is_(None)
            | (Project.generation_budget_currency == currency)
        ),
        (
            Project.video_budget_amount.is_(None)
            | (amount + estimate <= Project.video_budget_amount)
        ),
    ]
    result = session.execute(
        update(Project)
        .where(*conditions)
        .values(
            video_cost_exposure_amount=amount + estimate,
            video_cost_exposure_currency=func.coalesce(
                Project.video_cost_exposure_currency, currency
            ),
        )
    )
    if result.rowcount == 1:
        return

    session.rollback()
    current = session.get(Project, project.id)
    if current is None:
        raise ValueError("Project not found while reserving video exposure")
    if (
        current.video_cost_exposure_currency is not None
        and current.video_cost_exposure_currency != currency
    ):
        raise VideoGenerationError(
            VIDEO_BUDGET_EXCEEDED,
            "Video pricing currency does not match existing project exposure",
        )
    if (
        current.video_budget_amount is not None
        and current.generation_budget_currency != currency
    ):
        raise VideoGenerationError(
            VIDEO_BUDGET_EXCEEDED,
            "Video pricing currency does not match project budget",
        )
    raise VideoGenerationError(
        VIDEO_BUDGET_EXCEEDED,
        "Estimated video request would exceed the project video budget",
        diagnostic={
            "spent_or_reserved": float(current.video_cost_exposure_amount),
            "next_estimated_cost": estimate,
            "budget": (
                float(current.video_budget_amount)
                if current.video_budget_amount is not None
                else None
            ),
        },
    )


def _acquire_video_budget_transaction_lock(
    session: Session,
    project_id: str,
) -> None:
    """Serialize deduplication, budget check, and reservation in the database."""
    if session.new or session.dirty or session.deleted:
        raise RuntimeError(
            "Video budget reservation requires a clean database transaction"
        )
    if session.in_transaction():
        session.commit()
    bind = session.get_bind()
    if bind.dialect.name == "sqlite":
        session.connection().exec_driver_sql("BEGIN IMMEDIATE")
        return
    locked = session.scalar(
        select(Project.id).where(Project.id == project_id).with_for_update()
    )
    if locked is None:
        raise ValueError("Project not found while locking video budget")


def _reconcile_video_exposure(
    session: Session,
    attempt: VideoGenerationAttempt,
    *,
    actual_cost: float,
    currency: str | None,
) -> None:
    if currency is None:
        return
    estimated = float(attempt.estimated_cost or 0)
    delta = actual_cost - estimated
    unknown_delta = -1 if attempt.estimated_cost is None else 0
    amount = func.coalesce(Project.video_cost_exposure_amount, 0) + delta
    unknown_count = Project.video_unknown_exposure_count + unknown_delta
    session.execute(
        update(Project)
        .where(Project.id == attempt.project_id)
        .values(
            video_cost_exposure_amount=case((amount < 0, 0), else_=amount),
            video_cost_exposure_currency=func.coalesce(
                Project.video_cost_exposure_currency, currency
            ),
            video_unknown_exposure_count=case(
                (unknown_count < 0, 0), else_=unknown_count
            ),
        )
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
    if result.actual_cost is not None and attempt.currency is not None:
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


def _apply_provider_cost_reconciliation(
    session: Session,
    attempt: VideoGenerationAttempt,
    result: Any,
) -> bool:
    if attempt.actual_cost is not None:
        return True
    reconciled_cost, reconciled_currency = _reconcile_actual_cost(
        session, attempt, result
    )
    if reconciled_cost is None:
        return False
    _reconcile_video_exposure(
        session,
        attempt,
        actual_cost=reconciled_cost,
        currency=reconciled_currency,
    )
    attempt.actual_cost = reconciled_cost
    attempt.currency = reconciled_currency or attempt.currency
    attempt.cost_certainty = "ACTUAL"
    return True


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
    execution_context: dict[str, Any],
) -> str:
    payload = {
        "project_id": project_id,
        "beat_id": beat_id,
        "provider": provider.provider_id,
        "model": provider.model,
        "provider_execution_context": execution_context,
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
