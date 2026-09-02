"""Structured result of judging an image as a quickly readable video frame."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VisualQAResult(str, Enum):
    PASS = "PASS"
    REGENERATE = "REGENERATE"
    PASS_WITH_WARNING = "PASS_WITH_WARNING"


class VisualQAProblemCategory(str, Enum):
    MISSING_REQUIRED_OBJECT = "MISSING_REQUIRED_OBJECT"
    WRONG_PHYSICAL_STATE = "WRONG_PHYSICAL_STATE"
    WRONG_CHARACTER = "WRONG_CHARACTER"
    CHARACTER_DRIFT = "CHARACTER_DRIFT"
    LOCATION_DRIFT = "LOCATION_DRIFT"
    OBJECT_DRIFT = "OBJECT_DRIFT"
    STYLE_DRIFT_REALISM = "STYLE_DRIFT_REALISM"
    STYLE_DRIFT_DETAIL = "STYLE_DRIFT_DETAIL"
    STYLE_DRIFT_CHILDISH = "STYLE_DRIFT_CHILDISH"
    EDIT_CHANGED_TOO_MUCH = "EDIT_CHANGED_TOO_MUCH"
    COMPOSITION_UNCLEAR = "COMPOSITION_UNCLEAR"
    IMPORTANT_ACTION_TOO_SMALL = "IMPORTANT_ACTION_TOO_SMALL"
    EXCESSIVE_CLUTTER = "EXCESSIVE_CLUTTER"
    UNWANTED_TEXT = "UNWANTED_TEXT"
    WRONG_CAMERA = "WRONG_CAMERA"
    REFERENCE_NOT_RESPECTED = "REFERENCE_NOT_RESPECTED"
    OTHER = "OTHER"

    # Kept for backward compatibility with the Stage 1 QA prompt and records.
    STORY_ACCURACY = "STORY_ACCURACY"
    CONTINUITY = "CONTINUITY"
    VISUAL_PROGRESSION = "VISUAL_PROGRESSION"
    STYLE_DRIFT = "STYLE_DRIFT"
    COMPOSITION = "COMPOSITION"
    VIDEO_READABILITY = "VIDEO_READABILITY"


class VisualQASeverity(str, Enum):
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


class VisualQAScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    story_clarity: float = Field(default=1.0, ge=0, le=1)
    continuity: float = Field(default=1.0, ge=0, le=1)
    style: float = Field(default=1.0, ge=0, le=1)
    composition: float = Field(default=1.0, ge=0, le=1)
    operation_correctness: float = Field(default=1.0, ge=0, le=1)


class VisualQADecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: VisualQAResult
    scores: VisualQAScores = Field(default_factory=VisualQAScores)
    problem_categories: list[VisualQAProblemCategory]
    reasons: list[str]
    correction_instruction: str | None = None
    severity: VisualQASeverity | None = None

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
            if self.severity is not None:
                raise ValueError("PASS must not contain severity")
            if min(self.scores.model_dump().values()) < 0.5:
                raise ValueError("PASS scores contradict the acceptance result")
            return self
        if self.result is VisualQAResult.PASS_WITH_WARNING:
            if not self.problem_categories or not self.reasons:
                raise ValueError("PASS_WITH_WARNING requires warning details")
            if self.severity is VisualQASeverity.CRITICAL:
                raise ValueError("critical problems cannot PASS_WITH_WARNING")
            hard_categories = {
                VisualQAProblemCategory.MISSING_REQUIRED_OBJECT,
                VisualQAProblemCategory.WRONG_PHYSICAL_STATE,
                VisualQAProblemCategory.WRONG_CHARACTER,
                VisualQAProblemCategory.LOCATION_DRIFT,
                VisualQAProblemCategory.STORY_ACCURACY,
                VisualQAProblemCategory.CONTINUITY,
                VisualQAProblemCategory.STYLE_DRIFT_REALISM,
                VisualQAProblemCategory.EDIT_CHANGED_TOO_MUCH,
                VisualQAProblemCategory.COMPOSITION_UNCLEAR,
                VisualQAProblemCategory.VIDEO_READABILITY,
            }
            if set(self.problem_categories) & hard_categories:
                raise ValueError("hard failures cannot PASS_WITH_WARNING")
            return self
        if not self.problem_categories:
            raise ValueError("REGENERATE requires problem_categories")
        if not self.reasons:
            raise ValueError("REGENERATE requires reasons")
        if self.correction_instruction is None:
            raise ValueError("REGENERATE requires correction_instruction")
        if self.severity is None:
            self.severity = VisualQASeverity.MAJOR
        return self
