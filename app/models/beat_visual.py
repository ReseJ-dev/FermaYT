"""Structured non-rendered metadata for visual beat assets."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BeatVisualGenerationStatus(str, Enum):
    PENDING = "PENDING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class BeatVisualQAStatus(str, Enum):
    NOT_RUN = "NOT_RUN"
    PASS = "PASS"
    REGENERATE = "REGENERATE"
    PASS_WITH_WARNING = "PASS_WITH_WARNING"
    WARNING = "WARNING"  # Backward-compatible value used by older records.


class ManualVisualQAOverride(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    REGENERATE_REQUESTED = "REGENERATE_REQUESTED"


class TransformType(str, Enum):
    ZOOM_IN = "ZOOM_IN"
    ZOOM_OUT = "ZOOM_OUT"
    PAN = "PAN"
    CROP = "CROP"
    FOCUS = "FOCUS"


class OverlayType(str, Enum):
    ARROW = "ARROW"
    ROUTE = "ROUTE"
    X_MARK = "X_MARK"
    CIRCLE = "CIRCLE"
    HIGHLIGHT = "HIGHLIGHT"
    TEXT_LABEL = "TEXT_LABEL"
    MEASUREMENT = "MEASUREMENT"
    DEPTH = "DEPTH"
    DISTANCE = "DISTANCE"


class TransformSpecification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: TransformType
    target: str = Field(min_length=1)
    start_scale: float | None = Field(default=None, gt=0)
    end_scale: float | None = Field(default=None, gt=0)
    crop: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def validate_crop(self) -> "TransformSpecification":
        if self.crop is not None:
            left, top, right, bottom = self.crop
            if not all(0 <= value <= 1 for value in self.crop):
                raise ValueError("crop coordinates must be normalized")
            if left >= right or top >= bottom:
                raise ValueError("crop coordinates must describe a positive area")
        return self


class OverlaySpecification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: OverlayType
    instruction: str = Field(min_length=1)
    semantic_anchor: str = Field(min_length=1)
