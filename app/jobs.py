"""Persistent in-process background job management."""

import asyncio
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
        except Exception as exc:
            error = str(exc).strip() or type(exc).__name__
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
            connection.execute(
                """
                UPDATE generation_jobs
                SET status = ?, message = ?, error = ?, updated_at = ?
                WHERE status = ?
                """,
                (
                    GenerationJobStatus.FAILED.value,
                    "Interrupted",
                    "Job interrupted by application restart",
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
        )

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
                    SET status = ?, message = ?, error = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        status.value,
                        message,
                        error,
                        _serialize_datetime(_utc_now()),
                        job_id,
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE generation_jobs
                    SET status = ?, progress = ?, message = ?, error = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        status.value,
                        progress,
                        message,
                        error,
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
