"""Structured result of judging an image as a quickly readable video frame."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class VisualQAResult(str, Enum):
    PASS = "PASS"
    REGENERATE = "REGENERATE"


class VisualQAProblemCategory(str, Enum):
    STORY_ACCURACY = "STORY_ACCURACY"
    CONTINUITY = "CONTINUITY"
    STYLE_DRIFT = "STYLE_DRIFT"
    COMPOSITION = "COMPOSITION"
    VIDEO_READABILITY = "VIDEO_READABILITY"


class VisualQADecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: VisualQAResult
    problem_categories: list[VisualQAProblemCategory]
    reasons: list[str]
    correction_instruction: str | None = None

    @field_validator("reasons")
    @classmethod
    def normalize_reasons(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("QA reasons must not be empty")
        return normalized

    @field_validator("correction_instruction")
    @classmethod
    def normalize_correction(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_result_details(self) -> "VisualQADecision":
        if self.result is VisualQAResult.PASS:
            if self.problem_categories or self.reasons or self.correction_instruction:
                raise ValueError("PASS must not contain problems or correction")
            return self
        if not self.problem_categories:
            raise ValueError("REGENERATE requires problem_categories")
        if not self.reasons:
            raise ValueError("REGENERATE requires reasons")
        if self.correction_instruction is None:
            raise ValueError("REGENERATE requires correction_instruction")
        return self
