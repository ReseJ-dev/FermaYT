"""Persistent in-process background job management."""

import asyncio
import json
import re
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4


class GenerationJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerationJobType(str, Enum):
    GENERATE_IMAGES = "generate_images"
    GENERATE_AUDIO = "generate_audio"
    GENERATE_ASSETS = "generate_assets"
    RENDER_VIDEO = "render_video"
    GENERATE_VIDEO = "generate_video"


@dataclass(slots=True)
class GenerationJob:
    id: str
    project_id: str
    type: GenerationJobType
    status: GenerationJobStatus
    progress: int
    message: str | None
    error: str | None
    created_at: datetime
    updated_at: datetime
    pipeline_version: str | None = None
    current_stage: str | None = None
    current_beat: int | None = None
    total_beats: int | None = None
    stage_progress: int = 0
    failed_stage: str | None = None
    failed_beat: str | None = None
    final_render_id: str | None = None
    report: dict[str, object] | None = None
    completed_at: datetime | None = None


JobOperation = Callable[[str], Awaitable[None]]


class GenerationJobManager:
    """Run local background tasks and persist their status in SQLite."""

    def __init__(self, db_path: str | Path = "data/app.db") -> None:
        self.db_path = Path(db_path)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._started = False

    async def startup(self) -> None:
        """Create storage and fail jobs interrupted by an earlier process."""
        if self._started:
            return
        await asyncio.to_thread(self._initialize_and_recover)
        self._started = True

    async def shutdown(self) -> None:
        """Cancel in-process tasks while leaving them recoverable on restart."""
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def create_job(
        self,
        project_id: str,
        job_type: GenerationJobType,
    ) -> GenerationJob:
        """Persist a new queued job."""
        await self.startup()
        job = GenerationJob(
            id=str(uuid4()),
            project_id=project_id,
            type=GenerationJobType(job_type),
            status=GenerationJobStatus.QUEUED,
            progress=0,
            message=None,
            error=None,
            created_at=_utc_now(),
            updated_at=_utc_now(),
        )
        await asyncio.to_thread(self._insert_job, job)
        return job

    async def get_job(self, job_id: str) -> GenerationJob | None:
        """Load a job by ID."""
        await self.startup()
        return await asyncio.to_thread(self._get_job, job_id)

    async def get_latest_project_job(self, project_id: str) -> GenerationJob | None:
        await self.startup()
        return await asyncio.to_thread(self._get_latest_project_job, project_id)

    async def get_active_project_job(self, project_id: str) -> GenerationJob | None:
        latest = await self.get_latest_project_job(project_id)
        if latest is not None and latest.status in {
            GenerationJobStatus.QUEUED,
            GenerationJobStatus.RUNNING,
        }:
            return latest
        return None

    async def enqueue(
        self,
        project_id: str,
        job_type: GenerationJobType,
        operation: JobOperation,
    ) -> GenerationJob:
        """Create a queued job and schedule its operation in-process."""
        job = await self.create_job(project_id, job_type)
        task = asyncio.create_task(
            self._run_job(job.id, operation),
            name=f"generation-job-{job.id}",
        )
        self._tasks[job.id] = task
        task.add_done_callback(lambda _: self._tasks.pop(job.id, None))
        return job

    async def update_progress(
        self,
        job_id: str,
        progress: int,
        message: str | None = None,
    ) -> None:
        """Persist progress for a running operation."""
        if not 0 <= progress <= 100:
            raise ValueError("job progress must be between 0 and 100")
        await asyncio.to_thread(
            self._update_progress,
            job_id,
            progress,
            message,
        )

    async def update_pipeline_state(
        self,
        job_id: str,
        *,
        stage: str,
        progress: int,
        stage_progress: int,
        message: str,
        current_beat: int | None = None,
        total_beats: int | None = None,
        failed_beat: str | None = None,
    ) -> None:
        if not 0 <= progress <= 100 or not 0 <= stage_progress <= 100:
            raise ValueError("pipeline progress must be between 0 and 100")
        await asyncio.to_thread(
            self._update_pipeline_state,
            job_id,
            stage,
            progress,
            stage_progress,
            message,
            current_beat,
            total_beats,
            failed_beat,
        )

    async def set_pipeline_result(
        self,
        job_id: str,
        *,
        final_render_id: str,
        report: dict[str, object],
    ) -> None:
        await asyncio.to_thread(
            self._set_pipeline_result,
            job_id,
            final_render_id,
            report,
        )

    async def wait(self, job_id: str) -> GenerationJob | None:
        """Wait for a locally running job and return its final state."""
        task = self._tasks.get(job_id)
        if task is not None:
            await asyncio.shield(task)
        return await self.get_job(job_id)

    async def _run_job(
        self,
        job_id: str,
        operation: JobOperation,
    ) -> None:
        await asyncio.to_thread(
            self._update_status,
            job_id,
            GenerationJobStatus.RUNNING,
            "Running",
            None,
            None,
        )
        try:
            await operation(job_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - job boundary must persist all failures
            error = _safe_job_error(exc)
            await asyncio.to_thread(
                self._update_status,
                job_id,
                GenerationJobStatus.FAILED,
                "Failed",
                error[:500],
                None,
            )
        else:
            await asyncio.to_thread(
                self._update_status,
                job_id,
                GenerationJobStatus.COMPLETED,
                "Completed",
                None,
                100,
            )

    def _initialize_and_recover(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        now = _serialize_datetime(_utc_now())
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS generation_jobs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL
                        CHECK (progress BETWEEN 0 AND 100),
                    message TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(generation_jobs)")
            }
            additions = {
                "pipeline_version": "TEXT",
                "current_stage": "TEXT",
                "current_beat": "INTEGER",
                "total_beats": "INTEGER",
                "stage_progress": "INTEGER NOT NULL DEFAULT 0",
                "failed_stage": "TEXT",
                "failed_beat": "TEXT",
                "final_render_id": "TEXT",
                "report_json": "TEXT",
                "completed_at": "TEXT",
            }
            for name, sql_type in additions.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE generation_jobs ADD COLUMN {name} {sql_type}"
                    )
            connection.execute(
                """
                UPDATE generation_jobs
                SET status = ?, message = ?, error = ?, failed_stage = current_stage,
                    completed_at = ?, updated_at = ?
                WHERE status = ?
                """,
                (
                    GenerationJobStatus.FAILED.value,
                    "Interrupted",
                    "Job interrupted by application restart",
                    now,
                    now,
                    GenerationJobStatus.RUNNING.value,
                ),
            )

    def _insert_job(self, job: GenerationJob) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO generation_jobs (
                    id, project_id, type, status, progress,
                    message, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.project_id,
                    job.type.value,
                    job.status.value,
                    job.progress,
                    job.message,
                    job.error,
                    _serialize_datetime(job.created_at),
                    _serialize_datetime(job.updated_at),
                ),
            )

    def _get_job(self, job_id: str) -> GenerationJob | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM generation_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return GenerationJob(
            id=row["id"],
            project_id=row["project_id"],
            type=GenerationJobType(row["type"]),
            status=GenerationJobStatus(row["status"]),
            progress=row["progress"],
            message=row["message"],
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            pipeline_version=row["pipeline_version"],
            current_stage=row["current_stage"],
            current_beat=row["current_beat"],
            total_beats=row["total_beats"],
            stage_progress=row["stage_progress"],
            failed_stage=row["failed_stage"],
            failed_beat=row["failed_beat"],
            final_render_id=row["final_render_id"],
            report=json.loads(row["report_json"]) if row["report_json"] else None,
            completed_at=(
                datetime.fromisoformat(row["completed_at"])
                if row["completed_at"]
                else None
            ),
        )

    def _get_latest_project_job(self, project_id: str) -> GenerationJob | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM generation_jobs WHERE project_id = ? AND type = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (project_id, GenerationJobType.GENERATE_VIDEO.value),
            ).fetchone()
        return self._get_job(row["id"]) if row is not None else None

    def _update_progress(
        self,
        job_id: str,
        progress: int,
        message: str | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE generation_jobs
                SET progress = ?, message = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    progress,
                    message,
                    _serialize_datetime(_utc_now()),
                    job_id,
                ),
            )

    def _update_pipeline_state(
        self,
        job_id: str,
        stage: str,
        progress: int,
        stage_progress: int,
        message: str,
        current_beat: int | None,
        total_beats: int | None,
        failed_beat: str | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE generation_jobs
                SET pipeline_version = ?, current_stage = ?, progress = ?,
                    stage_progress = ?, message = ?, current_beat = ?,
                    total_beats = ?, failed_beat = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    "project_video_pipeline_v1",
                    stage,
                    progress,
                    stage_progress,
                    message,
                    current_beat,
                    total_beats,
                    failed_beat,
                    _serialize_datetime(_utc_now()),
                    job_id,
                ),
            )

    def _set_pipeline_result(
        self,
        job_id: str,
        final_render_id: str,
        report: dict[str, object],
    ) -> None:
        now = _serialize_datetime(_utc_now())
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE generation_jobs
                SET final_render_id = ?, report_json = ?, current_stage = ?,
                    stage_progress = 100, completed_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    final_render_id,
                    json.dumps(report, ensure_ascii=False, sort_keys=True),
                    "COMPLETED",
                    now,
                    now,
                    job_id,
                ),
            )

    def _update_status(
        self,
        job_id: str,
        status: GenerationJobStatus,
        message: str,
        error: str | None,
        progress: int | None,
    ) -> None:
        with self._connect() as connection:
            if progress is None:
                connection.execute(
                    """
                    UPDATE generation_jobs
                    SET status = ?, message = ?, error = ?,
                        failed_stage = CASE WHEN ? = 'failed' THEN current_stage ELSE failed_stage END,
                        completed_at = CASE WHEN ? = 'failed' THEN ? ELSE completed_at END,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        status.value,
                        message,
                        error,
                        status.value,
                        status.value,
                        _serialize_datetime(_utc_now()),
                        _serialize_datetime(_utc_now()),
                        job_id,
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE generation_jobs
                    SET status = ?, progress = ?, message = ?, error = ?,
                        completed_at = CASE WHEN ? = 'completed' THEN ? ELSE completed_at END,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        status.value,
                        progress,
                        message,
                        error,
                        status.value,
                        _serialize_datetime(_utc_now()),
                        _serialize_datetime(_utc_now()),
                        job_id,
                    ),
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _serialize_datetime(value: datetime) -> str:
    return value.isoformat()


def _safe_job_error(error: Exception) -> str:
    message = str(error).strip() or type(error).__name__
    message = re.sub(
        r"(?i)(bearer|authorization|api[_ -]?key)(?:\s*[:=]?\s*)[^\s,;]+",
        r"\1 [redacted]",
        message,
    )
    return message[:500]
