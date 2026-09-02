"""Normalized narration timing and executable visual timeline schemas."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.beat_visual import OverlayType, TransformType


class TimestampSource(str, Enum):
    NATIVE_WORD_TIMESTAMPS = "NATIVE_WORD_TIMESTAMPS"
    NATIVE_CHARACTER_TIMESTAMPS = "NATIVE_CHARACTER_TIMESTAMPS"
    NATIVE_SENTENCE_TIMESTAMPS = "NATIVE_SENTENCE_TIMESTAMPS"
    ALIGNER = "ALIGNER"
    PROPORTIONAL_FALLBACK = "PROPORTIONAL_FALLBACK"


class NarrationTimingGranularity(str, Enum):
    WORD = "WORD"
    CHARACTER = "CHARACTER"
    PHRASE = "PHRASE"
    SENTENCE = "SENTENCE"


class BeatAlignmentMethod(str, Enum):
    EXACT = "EXACT"
    NORMALIZED_EXACT = "NORMALIZED_EXACT"
    MONOTONIC_TOKEN = "MONOTONIC_TOKEN"
    FUZZY = "FUZZY"
    PROPORTIONAL_FALLBACK = "PROPORTIONAL_FALLBACK"


class NarrationTimingItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    source_start_char: int | None = Field(default=None, ge=0)
    source_end_char: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> "NarrationTimingItem":
        if self.end <= self.start:
            raise ValueError("narration timing end must be after start")
        if (
            self.source_start_char is not None
            and self.source_end_char is not None
            and self.source_end_char <= self.source_start_char
        ):
            raise ValueError("source character range must be positive")
        return self


class NarrationTimingTrack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: TimestampSource
    granularity: NarrationTimingGranularity
    confidence: float = Field(ge=0, le=1)
    items: list[NarrationTimingItem] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_monotonic_items(self) -> "NarrationTimingTrack":
        previous_start = -1.0
        for item in self.items:
            if item.start < previous_start:
                raise ValueError("narration timings must be monotonic")
            previous_start = item.start
        return self


class TimestampedNarrationOutput(BaseModel):
    """Optional richer result implemented by timestamp-capable TTS providers."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    audio: str | bytes
    timing: NarrationTimingTrack


class NormalizedPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class NormalizedTransform(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: TransformType
    start_scale: float | None = Field(default=None, gt=0)
    end_scale: float | None = Field(default=None, gt=0)
    focus: NormalizedPoint | None = None
    from_point: NormalizedPoint | None = Field(default=None, alias="from")
    to: NormalizedPoint | None = None
    crop: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def validate_execution_shape(self) -> "NormalizedTransform":
        if self.crop is not None:
            left, top, right, bottom = self.crop
            if not all(0 <= item <= 1 for item in self.crop):
                raise ValueError("crop coordinates must be normalized")
            if left >= right or top >= bottom:
                raise ValueError("crop must describe a positive area")
        if self.type is TransformType.PAN and (
            self.from_point is None or self.to is None
        ):
            raise ValueError("PAN requires from and to coordinates")
        return self


class NormalizedOverlay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: OverlayType
    instruction: str = Field(min_length=1)
    semantic_anchor: str = Field(min_length=1)
    start: NormalizedPoint | None = None
    end: NormalizedPoint | None = None
    center: NormalizedPoint | None = None
    points: list[NormalizedPoint] | None = Field(default=None, min_length=2)
    appear_offset: float = Field(default=0.4, ge=0)
