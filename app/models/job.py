"""Video generation job domain model."""

from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field

from app.models.scene import Scene


def _generate_uuid() -> str:
    return str(uuid4())


def _utc_now() -> datetime:
    return datetime.now(UTC)


class VideoJobStatus(str, Enum):
    """Possible states of a video generation job."""

    CREATED = "created"
    GENERATING_IMAGE = "generating_image"
    GENERATING_AUDIO = "generating_audio"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoJob(BaseModel):
    """State of a video generation job."""

    id: str = Field(default_factory=_generate_uuid)
    prompt: str
    text: str

    status: VideoJobStatus = VideoJobStatus.CREATED
    scenes: list[Scene] = Field(default_factory=list)

    final_video_path: str | None = None
    error: str | None = None

    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    def set_status(self, status: VideoJobStatus) -> None:
        self.status = status
        self.updated_at = _utc_now()

    def fail(self, error_message: str) -> None:
        self.status = VideoJobStatus.FAILED
        self.error = error_message
        self.updated_at = _utc_now()

    def complete(self, video_path: str) -> None:
        self.status = VideoJobStatus.COMPLETED
        self.final_video_path = video_path
        self.error = None
        self.updated_at = _utc_now()

    def add_scene(self, scene: Scene) -> None:
        self.scenes.append(scene)
        self.updated_at = _utc_now()
