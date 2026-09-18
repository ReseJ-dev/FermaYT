"""Restart and duplicate-submission safety for persisted remote video tasks."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import create_session_factory, create_sqlite_engine, init_database
from app.errors import (
    VIDEO_RECOVERY_CONFIGURATION_MISSING,
    VIDEO_SUBMISSION_TIMEOUT_UNKNOWN,
    VideoGenerationError,
)
from app.media.probe import MediaProbeResult
from app.persistence import GeneratedVideoAsset, Project, VideoGenerationAttempt
from app.repositories import create_project
from app.services.video_generation import (
    execute_video_generation,
    resume_incomplete_video_attempts,
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


class SimulatedProcessExit(BaseException):
    """An uncatchable-by-service boundary used to model process loss."""


class RecoverableWanProvider:
    provider_id = "wan"
    model = "recovery-test-model"
    capabilities = VideoProviderCapabilities(
        supports_image_to_video=True,
        supports_first_frame=True,
        supports_disable_audio=True,
        duration_range=(1, 10),
        supported_resolutions=("720p",),
    )

    def __init__(
        self,
        counters: dict[str, int],
        *,
        model: str = model,
        endpoint: str,
        workspace_id: str,
        region: str,
        crash_on_poll: bool = False,
    ) -> None:
        self.counters = counters
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.workspace_id = workspace_id
        self.region = region
        self.crash_on_poll = crash_on_poll

    def execution_context_snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "video_provider_execution_context_v1",
            "provider_id": self.provider_id,
            "model_id": self.model,
            "endpoint": self.endpoint,
            "workspace_id": self.workspace_id,
            "region": self.region,
            "api_variant": "dashscope_workspace_video_synthesis_v1",
            "provider_config": {
                "model": self.model,
                "endpoint": self.endpoint,
                "workspace_id": self.workspace_id,
                "region": self.region,
            },
        }

    async def submit(self, request: VideoGenerationRequest) -> VideoSubmission:
        del request
        self.counters["submit"] = self.counters.get("submit", 0) + 1
        return VideoSubmission("persisted-remote-task")

    async def get_task_status(self, remote_task_id: str) -> VideoTaskResult:
        assert remote_task_id == "persisted-remote-task"
        self.counters["poll"] = self.counters.get("poll", 0) + 1
        if self.crash_on_poll:
            raise SimulatedProcessExit
        return VideoTaskResult(
            RemoteVideoTaskState.SUCCEEDED,
            result_url="https://provider.invalid/temporary-video.mp4",
        )


@pytest.fixture
def recovery_config() -> dict[str, str]:
    return {
        "endpoint": "https://custom-video.example.test/api/v1",
        "workspace_id": "workspace-recovery-42",
        "region": "eu-test-9",
    }


def _project(session: Session):
    return create_project(
        session,
        name="Recovery",
        story_text="A miner waits for the lift.",
        video_generation_mode="MANUAL",
        video_provider="wan",
        video_model=RecoverableWanProvider.model,
    )


def _request(root: Path) -> VideoGenerationRequest:
    source = root / "accepted.png"
    source.write_bytes(b"accepted")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return VideoGenerationRequest(
        operation=VideoOperation.IMAGE_TO_VIDEO,
        prompt="The miner turns toward the lift.",
        duration_seconds=5,
        resolution="720p",
        aspect_ratio="16:9",
        references=(
            VideoReference(
                "accepted:recovery",
                VideoReferenceRole.FIRST_FRAME,
                str(source),
                digest,
            ),
        ),
    )


async def _download(_: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"validated-mp4")
    return destination


def _probe(_: str | Path) -> MediaProbeResult:
    return MediaProbeResult(5.0, True, False, 1280, 720, 24.0, "h264", 120)


def _run(
    session: Session,
    project: Any,
    provider: RecoverableWanProvider,
    request: VideoGenerationRequest,
    root: Path,
    *,
    downloader: Any = _download,
) -> GeneratedVideoAsset:
    return asyncio.run(
        execute_video_generation(
            session,
            project,
            provider,
            request,
            beat_id="beat-recovery",
            accepted_image_result_id=None,
            projects_root=root / "projects",
            job_id="job-recovery",
            poll_interval=0,
            downloader=downloader,
            prober=_probe,
        )
    )


def test_fresh_process_recovers_exact_custom_context_without_second_post(
    tmp_path: Path,
    recovery_config: dict[str, str],
) -> None:
    database = tmp_path / "restart.db"
    counters: dict[str, int] = {}
    engine_a = create_sqlite_engine(database)
    init_database(engine_a)
    factory_a = create_session_factory(engine_a)
    with factory_a() as session_a:
        project = _project(session_a)
        request = _request(tmp_path)
        provider_a = RecoverableWanProvider(
            counters, **recovery_config, crash_on_poll=True
        )
        with pytest.raises(SimulatedProcessExit):
            _run(session_a, project, provider_a, request, tmp_path)
        attempt = session_a.scalar(select(VideoGenerationAttempt))
        assert attempt is not None
        assert attempt.status == "SUBMITTED"
        assert attempt.remote_task_id == "persisted-remote-task"
        assert attempt.provider_execution_context is not None
    engine_a.dispose()

    captured_configs: list[dict[str, Any]] = []
    engine_b = create_sqlite_engine(database)
    init_database(engine_b)
    factory_b = create_session_factory(engine_b)
    with factory_b() as session_b:

        def resolver(name: str, config: dict[str, Any]) -> RecoverableWanProvider:
            assert name == "wan"
            captured_configs.append(dict(config))
            return RecoverableWanProvider(counters, **config)

        assets = asyncio.run(
            resume_incomplete_video_attempts(
                session_b,
                provider_resolver=resolver,
                projects_root=tmp_path / "projects",
                poll_interval=0,
                downloader=_download,
                prober=_probe,
            )
        )
        assert len(assets) == 1
        assert session_b.scalar(select(GeneratedVideoAsset)) is not None
    engine_b.dispose()

    assert captured_configs == [
        {"model": RecoverableWanProvider.model, **recovery_config}
    ]
    assert counters["submit"] == 1
    assert counters["poll"] == 2


def test_missing_recovery_context_fails_closed_without_second_post(
    tmp_path: Path,
    recovery_config: dict[str, str],
) -> None:
    database = tmp_path / "missing-context.db"
    counters: dict[str, int] = {}
    engine = create_sqlite_engine(database)
    init_database(engine)
    factory = create_session_factory(engine)
    with factory() as session:
        project = _project(session)
        provider = RecoverableWanProvider(
            counters, **recovery_config, crash_on_poll=True
        )
        with pytest.raises(SimulatedProcessExit):
            _run(session, project, provider, _request(tmp_path), tmp_path)
        attempt = session.scalar(select(VideoGenerationAttempt))
        assert attempt is not None
        attempt.provider_execution_context = None
        session.commit()
    engine.dispose()

    engine = create_sqlite_engine(database)
    init_database(engine)
    factory = create_session_factory(engine)
    with factory() as session:
        assets = asyncio.run(
            resume_incomplete_video_attempts(
                session,
                provider_resolver=lambda *_: pytest.fail("resolver must not run"),
                poll_interval=0,
            )
        )
        assert assets == []
        attempt = session.scalar(select(VideoGenerationAttempt))
        assert attempt is not None
        assert attempt.status == "RECOVERY_CONFIGURATION_MISSING"
        assert attempt.error_code == VIDEO_RECOVERY_CONFIGURATION_MISSING
    engine.dispose()
    assert counters["submit"] == 1


def test_crash_after_remote_success_resumes_download_without_second_post(
    tmp_path: Path,
    recovery_config: dict[str, str],
) -> None:
    database = tmp_path / "remote-succeeded.db"
    counters: dict[str, int] = {}
    engine = create_sqlite_engine(database)
    init_database(engine)
    factory = create_session_factory(engine)

    async def process_exit_during_download(_: str, __: Path) -> Path:
        raise SimulatedProcessExit

    with factory() as session:
        project = _project(session)
        provider = RecoverableWanProvider(counters, **recovery_config)
        with pytest.raises(SimulatedProcessExit):
            _run(
                session,
                project,
                provider,
                _request(tmp_path),
                tmp_path,
                downloader=process_exit_during_download,
            )
        attempt = session.scalar(select(VideoGenerationAttempt))
        assert attempt is not None
        assert attempt.status == "REMOTE_SUCCEEDED"
        assert attempt.remote_result_url
    engine.dispose()

    engine = create_sqlite_engine(database)
    init_database(engine)
    factory = create_session_factory(engine)
    with factory() as session:

        def resolver(_: str, config: dict[str, Any]) -> RecoverableWanProvider:
            return RecoverableWanProvider(counters, **config)

        assets = asyncio.run(
            resume_incomplete_video_attempts(
                session,
                provider_resolver=resolver,
                projects_root=tmp_path / "projects",
                poll_interval=0,
                downloader=_download,
                prober=_probe,
            )
        )
        assert len(assets) == 1
    engine.dispose()
    assert counters["submit"] == 1


def test_process_crash_before_post_never_submits_on_restart(
    tmp_path: Path,
    recovery_config: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "before-post.db"
    counters: dict[str, int] = {}
    engine = create_sqlite_engine(database)
    init_database(engine)
    factory = create_session_factory(engine)
    request = _request(tmp_path)
    with factory() as session:
        project = _project(session)
        provider = RecoverableWanProvider(counters, **recovery_config)
        real_commit = session.commit
        commits = 0

        def crash_on_paid_boundary_commit() -> None:
            nonlocal commits
            commits += 1
            if commits == 2:
                raise SimulatedProcessExit
            real_commit()

        monkeypatch.setattr(session, "commit", crash_on_paid_boundary_commit)
        with pytest.raises(SimulatedProcessExit):
            _run(session, project, provider, request, tmp_path)
    engine.dispose()

    engine = create_sqlite_engine(database)
    init_database(engine)
    factory = create_session_factory(engine)
    with factory() as session:
        project = session.scalar(select(Project))
        assert project is not None
        provider = RecoverableWanProvider(counters, **recovery_config)
        with pytest.raises(VideoGenerationError) as raised:
            _run(session, project, provider, request, tmp_path)
        assert raised.value.code == VIDEO_SUBMISSION_TIMEOUT_UNKNOWN
    engine.dispose()
    assert counters.get("submit", 0) == 0


def test_integrity_race_reloads_winner_without_raw_error_or_second_post(
    tmp_path: Path,
    recovery_config: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "race.db"
    counters: dict[str, int] = {}
    engine = create_sqlite_engine(database)
    init_database(engine)
    factory = create_session_factory(engine)
    with factory() as session:
        project = _project(session)
        provider = RecoverableWanProvider(counters, **recovery_config)
        request = _request(tmp_path)
        real_commit = session.commit
        inserted_winner = False

        def commit_with_competing_winner() -> None:
            nonlocal inserted_winner
            pending = next(
                (
                    item
                    for item in session.new
                    if isinstance(item, VideoGenerationAttempt)
                ),
                None,
            )
            if pending is not None and not inserted_winner:
                inserted_winner = True
                with Session(bind=engine, expire_on_commit=False) as competitor:
                    competitor.add(
                        VideoGenerationAttempt(
                            project_id=pending.project_id,
                            job_id="competing-job",
                            beat_id=pending.beat_id,
                            provider=pending.provider,
                            model=pending.model,
                            operation=pending.operation,
                            capability_snapshot=pending.capability_snapshot,
                            provider_execution_context=(
                                pending.provider_execution_context
                            ),
                            request_hash=pending.request_hash,
                            request_snapshot=pending.request_snapshot,
                            prompt=pending.prompt,
                            reference_snapshot=pending.reference_snapshot,
                            source_image_path=pending.source_image_path,
                            requested_duration=pending.requested_duration,
                            requested_resolution=pending.requested_resolution,
                            requested_aspect_ratio=pending.requested_aspect_ratio,
                            status="SUBMITTED",
                            remote_task_id="persisted-remote-task",
                            submission_started_at=datetime.now(UTC),
                            submitted_at=datetime.now(UTC),
                        )
                    )
                    competitor.commit()
            real_commit()

        monkeypatch.setattr(session, "commit", commit_with_competing_winner)
        asset = _run(session, project, provider, request, tmp_path)

        assert Path(asset.file_path).exists()
        assert counters.get("submit", 0) == 0
        assert counters["poll"] == 1
        assert len(list(session.scalars(select(VideoGenerationAttempt)))) == 1
    engine.dispose()
