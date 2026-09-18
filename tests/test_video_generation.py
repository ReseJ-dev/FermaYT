"""Crash, billing, reference, and local-asset safety for AI video generation."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.clients.video_api import WanVideoProvider
from app.costs import PricingUnit, configure_provider_pricing
from app.database import create_session_factory, create_sqlite_engine, init_database
from app.errors import (
    VIDEO_BUDGET_EXCEEDED,
    VIDEO_CAPABILITY_UNSUPPORTED,
    VIDEO_POLL_TIMEOUT,
    VIDEO_SUBMISSION_TIMEOUT_UNKNOWN,
    VideoGenerationError,
)
from app.media.probe import MediaProbeResult
from app.persistence import ProviderUsageRecord, VideoGenerationAttempt
from app.repositories import create_project
from app.services.video_generation import execute_video_generation
from app.video_providers import (
    RemoteVideoTaskState,
    VideoGenerationRequest,
    VideoOperation,
    VideoProviderCapabilities,
    VideoReference,
    VideoReferenceRole,
    VideoSubmission,
    VideoTaskResult,
    select_video_references,
)


@pytest.fixture
def session(tmp_path: Path) -> Iterator[Session]:
    engine = create_sqlite_engine(tmp_path / "video.db")
    init_database(engine)
    factory = create_session_factory(engine)
    with factory() as value:
        yield value
    engine.dispose()


class FakeVideoProvider:
    provider_id = "vidu"
    model = "fake-video"
    capabilities = VideoProviderCapabilities(
        supports_image_to_video=True,
        supports_first_frame=True,
        supports_first_last_frame=True,
        supports_last_frame=True,
        supports_reference_images=True,
        max_reference_images=3,
        supports_disable_audio=True,
        duration_range=(1, 10),
        supported_resolutions=("720p",),
    )

    def __init__(self, states: list[VideoTaskResult] | None = None) -> None:
        self.submit_calls = 0
        self.poll_calls = 0
        self.states = states or [
            VideoTaskResult(RemoteVideoTaskState.PROCESSING),
            VideoTaskResult(
                RemoteVideoTaskState.SUCCEEDED,
                result_url="https://provider.invalid/temporary.mp4",
                actual_cost=0.12,
            ),
        ]

    async def submit(self, request: VideoGenerationRequest) -> VideoSubmission:
        self.submit_calls += 1
        return VideoSubmission("remote-task-1", provider_metadata={"accepted": True})

    async def get_task_status(self, remote_task_id: str) -> VideoTaskResult:
        assert remote_task_id == "remote-task-1"
        index = min(self.poll_calls, len(self.states) - 1)
        self.poll_calls += 1
        return self.states[index]


class AmbiguousProvider(FakeVideoProvider):
    async def submit(self, request: VideoGenerationRequest) -> VideoSubmission:
        self.submit_calls += 1
        raise VideoGenerationError(
            VIDEO_SUBMISSION_TIMEOUT_UNKNOWN,
            "submission timed out",
            provider=self.provider_id,
        )


def _project(session: Session, **changes: object):
    values = {
        "name": "AI video",
        "story_text": "A miner turns toward the ventilation duct.",
        "video_generation_mode": "MANUAL",
        "video_provider": "vidu",
        "video_model": "fake-video",
        "generation_budget_currency": "USD",
    }
    values.update(changes)
    return create_project(session, **values)


def _request(tmp_path: Path) -> VideoGenerationRequest:
    source = tmp_path / "accepted.png"
    source.write_bytes(b"accepted-image")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return VideoGenerationRequest(
        operation=VideoOperation.IMAGE_TO_VIDEO,
        prompt="Keep the miner stable; slowly turn toward the duct.",
        duration_seconds=5,
        resolution="720p",
        aspect_ratio="16:9",
        references=(
            VideoReference(
                "accepted:1", VideoReferenceRole.FIRST_FRAME, str(source), digest
            ),
        ),
    )


async def _download(_: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"local-immutable-mp4")
    return destination


def _probe(_: str | Path) -> MediaProbeResult:
    return MediaProbeResult(5.0, True, True, 1280, 720, 24.0, "h264", 120)


def _execute(session: Session, project, provider, request, tmp_path: Path, **kwargs):
    downloader = kwargs.pop("downloader", _download)
    prober = kwargs.pop("prober", _probe)
    return asyncio.run(
        execute_video_generation(
            session,
            project,
            provider,
            request,
            beat_id="beat-1",
            accepted_image_result_id=None,
            projects_root=tmp_path / "projects",
            job_id="job-1",
            poll_interval=0,
            downloader=downloader,
            prober=prober,
            **kwargs,
        )
    )


def test_task_id_is_persisted_then_result_is_downloaded_locally(
    session: Session, tmp_path: Path
) -> None:
    project = _project(session)
    provider = FakeVideoProvider()

    asset = _execute(session, project, provider, _request(tmp_path), tmp_path)

    attempt = session.scalar(select(VideoGenerationAttempt))
    assert attempt is not None
    assert attempt.remote_task_id == "remote-task-1"
    assert attempt.status == "SUCCEEDED"
    assert provider.submit_calls == 1
    assert provider.poll_calls == 2
    assert Path(asset.file_path).read_bytes() == b"local-immutable-mp4"
    assert asset.file_path != "https://provider.invalid/temporary.mp4"
    assert asset.provider_has_audio is True
    assert asset.mute_audio_default is True
    assert asset.source_lineage["references"][0]["reference_id"] == "accepted:1"
    usage = session.scalar(select(ProviderUsageRecord))
    assert usage is not None
    assert float(usage.actual_cost) == pytest.approx(0.12)


def test_poll_timeout_resumes_same_task_without_second_submission(
    session: Session, tmp_path: Path
) -> None:
    project = _project(session)
    provider = FakeVideoProvider([VideoTaskResult(RemoteVideoTaskState.PROCESSING)])
    request = _request(tmp_path)
    with pytest.raises(VideoGenerationError) as first:
        _execute(
            session,
            project,
            provider,
            request,
            tmp_path,
            max_poll_duration=0.000001,
        )
    assert first.value.code == VIDEO_POLL_TIMEOUT
    assert provider.submit_calls == 1

    provider.states = [
        VideoTaskResult(
            RemoteVideoTaskState.SUCCEEDED,
            result_url="https://provider.invalid/recovered.mp4",
        )
    ]
    asset = _execute(session, project, provider, request, tmp_path)
    assert asset.file_path.endswith(".mp4")
    assert provider.submit_calls == 1


def test_download_failure_reuses_same_remote_task_without_second_submission(
    session: Session, tmp_path: Path
) -> None:
    project = _project(session)
    provider = FakeVideoProvider(
        [
            VideoTaskResult(
                RemoteVideoTaskState.SUCCEEDED,
                result_url="https://provider.invalid/recovered.mp4",
            )
        ]
    )
    request = _request(tmp_path)

    async def fail_download(_: str, __: Path) -> Path:
        raise OSError("temporary network failure")

    with pytest.raises(VideoGenerationError) as first:
        _execute(
            session,
            project,
            provider,
            request,
            tmp_path,
            downloader=fail_download,
        )
    assert first.value.code == "VIDEO_DOWNLOAD_FAILED"

    asset = _execute(session, project, provider, request, tmp_path)

    assert Path(asset.file_path).exists()
    assert provider.submit_calls == 1


def test_ffprobe_failure_reuses_same_remote_task_without_second_submission(
    session: Session, tmp_path: Path
) -> None:
    project = _project(session)
    provider = FakeVideoProvider(
        [
            VideoTaskResult(
                RemoteVideoTaskState.SUCCEEDED,
                result_url="https://provider.invalid/recovered.mp4",
            )
        ]
    )
    request = _request(tmp_path)

    def fail_probe(_: str | Path) -> MediaProbeResult:
        raise ValueError("ffprobe could not parse the file")

    with pytest.raises(VideoGenerationError) as first:
        _execute(
            session,
            project,
            provider,
            request,
            tmp_path,
            prober=fail_probe,
        )
    assert first.value.code == "VIDEO_VALIDATION_FAILED"

    asset = _execute(session, project, provider, request, tmp_path)

    assert asset.duration == pytest.approx(5.0)
    assert provider.submit_calls == 1


def test_ambiguous_submission_is_never_automatically_resubmitted(
    session: Session, tmp_path: Path
) -> None:
    project = _project(session)
    provider = AmbiguousProvider()
    request = _request(tmp_path)
    for _ in range(2):
        with pytest.raises(VideoGenerationError) as raised:
            _execute(session, project, provider, request, tmp_path)
        assert raised.value.code == VIDEO_SUBMISSION_TIMEOUT_UNKNOWN
    assert provider.submit_calls == 1
    attempt = session.scalar(select(VideoGenerationAttempt))
    assert attempt is not None
    assert attempt.status == "SUBMISSION_STATUS_UNKNOWN"
    assert attempt.cost_certainty == "UNKNOWN"


def test_budget_blocks_before_provider_submission(
    session: Session, tmp_path: Path
) -> None:
    project = _project(session, video_budget_amount=0.10)
    configure_provider_pricing(
        session,
        provider="vidu",
        model="fake-video",
        operation="IMAGE_TO_VIDEO",
        pricing_unit=PricingUnit.PER_SECOND,
        price=0.03,
        currency="USD",
        version="test-v1",
        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
    )
    provider = FakeVideoProvider()
    with pytest.raises(VideoGenerationError) as raised:
        _execute(session, project, provider, _request(tmp_path), tmp_path)
    assert raised.value.code == VIDEO_BUDGET_EXCEEDED
    assert provider.submit_calls == 0
    assert session.scalar(select(VideoGenerationAttempt)) is None


def test_unsupported_operation_fails_before_submission(
    session: Session, tmp_path: Path
) -> None:
    project = _project(session)
    provider = FakeVideoProvider()
    request = _request(tmp_path)
    request = VideoGenerationRequest(
        operation=VideoOperation.VIDEO_EDIT,
        prompt=request.prompt,
        duration_seconds=5,
        resolution="720p",
        aspect_ratio="16:9",
    )
    with pytest.raises(VideoGenerationError) as raised:
        _execute(session, project, provider, request, tmp_path)
    assert raised.value.code == VIDEO_CAPABILITY_UNSUPPORTED
    assert provider.submit_calls == 0


def test_first_last_are_never_displaced_by_optional_references(tmp_path: Path) -> None:
    digest = "a" * 64
    refs = tuple(
        VideoReference(name, role, str(tmp_path / name), digest)
        for name, role in (
            ("style", VideoReferenceRole.STYLE_REFERENCE),
            ("last", VideoReferenceRole.LAST_FRAME),
            ("character", VideoReferenceRole.CHARACTER_REFERENCE),
            ("first", VideoReferenceRole.FIRST_FRAME),
        )
    )
    capabilities = VideoProviderCapabilities(
        supports_first_last_frame=True,
        supports_reference_images=True,
        max_reference_images=2,
    )
    selected = select_video_references(
        VideoOperation.FIRST_LAST_TO_VIDEO, refs, capabilities
    )
    assert [item.role for item in selected] == [
        VideoReferenceRole.FIRST_FRAME,
        VideoReferenceRole.LAST_FRAME,
    ]


def test_reference_limit_prefers_identity_then_state_then_location() -> None:
    digest = "b" * 64
    refs = tuple(
        VideoReference(role.value, role, f"/{role.value}.png", digest)
        for role in (
            VideoReferenceRole.STYLE_REFERENCE,
            VideoReferenceRole.MASTER_LOCATION,
            VideoReferenceRole.PREVIOUS_STATE,
            VideoReferenceRole.CHARACTER_REFERENCE,
        )
    )
    capabilities = VideoProviderCapabilities(
        supports_reference_images=True, max_reference_images=2
    )
    selected = select_video_references(
        VideoOperation.REFERENCE_TO_VIDEO, refs, capabilities
    )
    assert [item.role for item in selected] == [
        VideoReferenceRole.CHARACTER_REFERENCE,
        VideoReferenceRole.PREVIOUS_STATE,
    ]


def test_unknown_paid_model_fails_before_attempt_or_submission(
    session: Session, tmp_path: Path
) -> None:
    project = _project(session, video_provider="wan", video_model="wan3-edit-future")
    provider = WanVideoProvider(
        api_key="secret",
        model="wan3-edit-future",
        endpoint="https://wan.invalid/api/v1",
    )
    with pytest.raises(VideoGenerationError) as raised:
        _execute(session, project, provider, _request(tmp_path), tmp_path)
    assert raised.value.code == VIDEO_CAPABILITY_UNSUPPORTED
    assert session.scalar(select(VideoGenerationAttempt)) is None


@pytest.mark.parametrize(
    ("duration", "resolution"),
    ((1, "720p"), (5, "480p")),
)
def test_exact_wan_profile_rejects_invalid_duration_or_resolution_before_submission(
    session: Session,
    tmp_path: Path,
    duration: int,
    resolution: str,
) -> None:
    project = _project(
        session, video_provider="wan", video_model="wan2.7-i2v-2026-04-25"
    )
    provider = WanVideoProvider(
        api_key="secret",
        model="wan2.7-i2v-2026-04-25",
        endpoint="https://wan.invalid/api/v1",
    )
    original = _request(tmp_path)
    request = VideoGenerationRequest(
        operation=original.operation,
        prompt=original.prompt,
        duration_seconds=duration,
        resolution=resolution,
        aspect_ratio=original.aspect_ratio,
        references=original.references,
    )
    with pytest.raises(VideoGenerationError) as raised:
        _execute(session, project, provider, request, tmp_path)
    assert raised.value.code == VIDEO_CAPABILITY_UNSUPPORTED
    assert session.scalar(select(VideoGenerationAttempt)) is None
