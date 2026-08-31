"""Generated scene draft DTO."""

from pydantic import BaseModel, ConfigDict, field_validator


class SceneDraft(BaseModel):
    """Intermediate Story provider output before persistence."""

    model_config = ConfigDict(extra="forbid")

    text: str
    image_prompt: str

    @field_validator("text", "image_prompt")
    @classmethod
    def validate_and_trim_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized
