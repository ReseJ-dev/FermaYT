"""Typed deterministic configuration for executing a ProjectTimeline."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RenderImageFit(str, Enum):
    COVER = "COVER"
    CONTAIN = "CONTAIN"


class RenderTransition(str, Enum):
    CUT = "CUT"
    CROSSFADE = "CROSSFADE"


class ProjectRenderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "project_render_config_v1"
    width: int = Field(default=1920, gt=0)
    height: int = Field(default=1080, gt=0)
    fps: int = Field(default=30, gt=0, le=120)
    video_codec: str = Field(default="libx264", min_length=1)
    audio_codec: str = Field(default="aac", min_length=1)
    pixel_format: str = Field(default="yuv420p", min_length=1)
    image_fit_mode: RenderImageFit = RenderImageFit.COVER
    background_color: str = "#111111"
    default_transition: RenderTransition = RenderTransition.CUT
    output_container: str = "mp4"
    max_motion_scale: float = Field(default=1.15, ge=1.0, le=1.5)

    @field_validator("width", "height")
    @classmethod
    def require_even_dimensions(cls, value: int) -> int:
        if value % 2:
            raise ValueError("render dimensions must be even")
        return value

    @field_validator("background_color")
    @classmethod
    def validate_background_color(cls, value: str) -> str:
        normalized = value.strip()
        if (
            len(normalized) != 7
            or not normalized.startswith("#")
            or any(character not in "0123456789abcdefABCDEF" for character in normalized[1:])
        ):
            raise ValueError("background_color must be a #RRGGBB value")
        return normalized.lower()

    @field_validator("output_container")
    @classmethod
    def support_mp4_only(cls, value: str) -> str:
        if value.lower() != "mp4":
            raise ValueError("only MP4 output is supported")
        return "mp4"


class OverlayStyleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stroke_color: str = "#f4c542"
    text_color: str = "#ffffff"
    fill_color: str = "#f4c542"
    stroke_width: int = Field(default=8, gt=0)
    font_size: int = Field(default=42, gt=0)
    safe_margin: float = Field(default=0.05, ge=0, le=0.25)
    opacity: float = Field(default=0.28, ge=0, le=1)
