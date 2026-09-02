"""Tests for the local background job manager."""

import asyncio
from pathlib import Path

import pytest

from app.jobs import (
    GenerationJobManager,
    GenerationJobStatus,
    GenerationJobType,
)


def test_create_job_persists_queued_metadata(tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = GenerationJobManager(tmp_path / "app.db")
        job = await manager.create_job(
            "project-1",
            GenerationJobType.GENERATE_IMAGES,
        )

        stored = await manager.get_job(job.id)

        assert stored is not None
        assert stored.project_id == "project-1"
        assert stored.type is GenerationJobType.GENERATE_IMAGES
        assert stored.status is GenerationJobStatus.QUEUED
        assert stored.progress == 0
        assert stored.error is None
        assert stored.created_at.tzinfo is not None

    asyncio.run(scenario())


def test_enqueue_runs_job_and_persists_progress(tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = GenerationJobManager(tmp_path / "app.db")
        operation_started = asyncio.Event()
        allow_completion = asyncio.Event()

        async def operation(job_id: str) -> None:
            await manager.update_progress(job_id, 40, "Generating asset")
            operation_started.set()
            await allow_completion.wait()

        job = await manager.enqueue(
            "project-1",
            GenerationJobType.GENERATE_ASSETS,
            operation,
        )
        await operation_started.wait()

        running = await manager.get_job(job.id)
        assert running is not None
        assert running.status is GenerationJobStatus.RUNNING
        assert running.progress == 40
        assert running.message == "Generating asset"

        allow_completion.set()
        completed = await manager.wait(job.id)

        assert completed is not None
        assert completed.status is GenerationJobStatus.COMPLETED
        assert completed.progress == 100
        assert completed.message == "Completed"

    asyncio.run(scenario())


def test_failed_operation_is_persisted(tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = GenerationJobManager(tmp_path / "app.db")

        async def operation(job_id: str) -> None:
            raise RuntimeError("Provider unavailable")

        job = await manager.enqueue(
            "project-1",
            GenerationJobType.GENERATE_AUDIO,
            operation,
        )
        failed = await manager.wait(job.id)

        assert failed is not None
        assert failed.status is GenerationJobStatus.FAILED
        assert failed.message == "Failed"
        assert failed.error == "Provider unavailable"

    asyncio.run(scenario())


def test_startup_marks_running_jobs_as_interrupted(tmp_path: Path) -> None:
    async def scenario() -> None:
        db_path = tmp_path / "app.db"
        first_manager = GenerationJobManager(db_path)
        operation_started = asyncio.Event()
        never_complete = asyncio.Event()

        async def operation(job_id: str) -> None:
            operation_started.set()
            await never_complete.wait()

        job = await first_manager.enqueue(
            "project-1",
            GenerationJobType.RENDER_VIDEO,
            operation,
        )
        await operation_started.wait()

        second_manager = GenerationJobManager(db_path)
        await second_manager.startup()
        recovered = await second_manager.get_job(job.id)

        assert recovered is not None
        assert recovered.status is GenerationJobStatus.FAILED
        assert recovered.message == "Interrupted"
        assert recovered.error == "Job interrupted by application restart"

        await first_manager.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("progress", [-1, 101])
def test_update_progress_rejects_out_of_range_value(
    tmp_path: Path,
    progress: int,
) -> None:
    async def scenario() -> None:
        manager = GenerationJobManager(tmp_path / "app.db")
        job = await manager.create_job(
            "project-1",
            GenerationJobType.GENERATE_IMAGES,
        )

        with pytest.raises(ValueError, match="between 0 and 100"):
            await manager.update_progress(job.id, progress)

    asyncio.run(scenario())


def test_pipeline_state_and_report_survive_manager_restart(tmp_path: Path) -> None:
    async def scenario() -> None:
        db_path = tmp_path / "app.db"
        manager = GenerationJobManager(db_path)
        job = await manager.create_job("project-1", GenerationJobType.GENERATE_VIDEO)
        await manager.update_pipeline_state(
            job.id,
            stage="GENERATING_VISUALS",
            progress=42,
            stage_progress=50,
            message="Generating visual beat 3 of 6",
            current_beat=3,
            total_beats=6,
        )
        await manager.set_pipeline_result(
            job.id,
            final_render_id="render-1",
            report={"visual_beats": 6, "final_mp4": "safe.mp4"},
        )

        restarted = GenerationJobManager(db_path)
        stored = await restarted.get_job(job.id)
        latest = await restarted.get_latest_project_job("project-1")
        assert stored is not None and latest is not None
        assert stored.current_stage == "COMPLETED"
        assert stored.current_beat == 3
        assert stored.total_beats == 6
        assert stored.final_render_id == "render-1"
        assert stored.report == {"visual_beats": 6, "final_mp4": "safe.mp4"}
        assert latest.id == job.id

    asyncio.run(scenario())
