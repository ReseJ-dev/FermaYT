"""Persisted Project and Scene entities."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from sqlalchemy.types import TypeDecorator

from app.database import Base


def _generate_uuid() -> str:
    return str(uuid4())


def _utc_now() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """Store UTC datetimes in SQLite and restore timezone information."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(
        self,
        value: datetime | None,
        dialect: Any,
    ) -> datetime | None:
        del dialect
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("datetime must be timezone-aware")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(
        self,
        value: datetime | None,
        dialect: Any,
    ) -> datetime | None:
        del dialect
        if value is None:
            return None
        return value.replace(tzinfo=UTC)


class Project(Base):
    """Long-lived user project persisted in SQLite."""

    __tablename__ = "projects"
    __table_args__ = (
        CheckConstraint(
            "scene_count IS NULL OR scene_count BETWEEN 1 AND 50",
            name="ck_projects_scene_count",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=_generate_uuid,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    story_text: Mapped[str] = mapped_column(Text, nullable=False)
    global_image_style_prompt: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
    )
    scene_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        default=None,
    )

    image_provider: Mapped[str] = mapped_column(String(50), nullable=False)
    image_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    tts_provider: Mapped[str] = mapped_column(String(50), nullable=False)
    tts_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    tts_voice: Mapped[str] = mapped_column(String(100), nullable=False)
    tts_language: Mapped[str] = mapped_column(String(100), nullable=False)

    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    fps: Mapped[int] = mapped_column(Integer, nullable=False)
    image_fit: Mapped[str] = mapped_column(String(20), nullable=False)

    final_video_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        default=_utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        default=_utc_now,
        onupdate=_utc_now,
        nullable=False,
    )
    rendered_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
        default=None,
    )

    scenes: Mapped[list[Scene]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="Scene.position",
    )

    @validates("story_text")
    def validate_story_text(self, key: str, value: str) -> str:
        del key
        normalized = value.strip()
        if not normalized:
            raise ValueError("story_text must not be empty")
        return normalized

    @validates("global_image_style_prompt")
    def normalize_style_prompt(
        self,
        key: str,
        value: str | None,
    ) -> str | None:
        del key
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @validates("scene_count")
    def validate_scene_count(
        self,
        key: str,
        value: int | None,
    ) -> int | None:
        del key
        if value is not None and not 1 <= value <= 50:
            raise ValueError("scene_count must be between 1 and 50")
        return value


class Scene(Base):
    """A scene persisted as part of a Project."""

    __tablename__ = "scenes"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "position",
            name="uq_scenes_project_position",
        ),
        CheckConstraint("position >= 0", name="ck_scenes_position"),
        CheckConstraint(
            "duration IS NULL OR duration > 0",
            name="ck_scenes_duration",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=_generate_uuid,
    )
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    image_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    audio_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    video_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        default=_utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        default=_utc_now,
        onupdate=_utc_now,
        nullable=False,
    )

    project: Mapped[Project] = relationship(back_populates="scenes")
