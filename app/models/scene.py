"""Scene domain model."""

from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def _generate_uuid() -> str:
    return str(uuid4())


class Scene(BaseModel):
    """A single scene in a generated video."""

    id: str = Field(default_factory=_generate_uuid)
    text: str
    image_prompt: str

    image_path: str | None = None
    audio_path: str | None = None
    video_path: str | None = None

    duration: float | None = Field(default=None, gt=0)

    @field_validator("text", "image_prompt")
    @classmethod
    def validate_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value
