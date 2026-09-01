"""Persisted Project and Scene entities."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
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
    master_scene_assets: Mapped[list[MasterSceneAsset]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="MasterSceneAsset.master_scene_id",
    )
    style_reference_assets: Mapped[list[StyleReferenceAsset]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="StyleReferenceAsset.style_id",
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


class ApplicationSettings(Base):
    """Global non-secret defaults for newly created projects."""

    __tablename__ = "application_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    image_provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="seedream",
    )
    tts_provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="qwen",
    )
    qwen_image_endpoint: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        default=_utc_now,
        onupdate=_utc_now,
        nullable=False,
    )

    @validates("image_provider")
    def validate_image_provider(self, key: str, value: str) -> str:
        del key
        if value not in {"seedream", "qwen"}:
            raise ValueError("unsupported image provider")
        return value

    @validates("tts_provider")
    def validate_tts_provider(self, key: str, value: str) -> str:
        del key
        if value not in {"qwen", "elevenlabs"}:
            raise ValueError("unsupported TTS provider")
        return value

    @validates("qwen_image_endpoint")
    def normalize_qwen_endpoint(
        self,
        key: str,
        value: str | None,
    ) -> str | None:
        del key
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


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


class MasterSceneAsset(Base):
    """Immutable generated continuity anchor for one recurring environment."""

    __tablename__ = "master_scene_assets"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "master_scene_id",
            name="uq_master_scene_assets_project_master",
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
    master_scene_id: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    style_version: Mapped[str] = mapped_column(String(64), nullable=False)
    generation_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reference_hashes: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        default=_utc_now,
        nullable=False,
    )

    project: Mapped[Project] = relationship(back_populates="master_scene_assets")


class StyleReferenceAsset(Base):
    """Explicitly approved immutable style reference for a project style version."""

    __tablename__ = "style_reference_assets"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "style_id",
            name="uq_style_reference_assets_project_style",
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
    style_id: Mapped[str] = mapped_column(String(64), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        default=_utc_now,
        nullable=False,
    )

    project: Mapped[Project] = relationship(back_populates="style_reference_assets")


@event.listens_for(MasterSceneAsset, "before_update")
def _prevent_master_scene_update(
    mapper: object,
    connection: object,
    target: MasterSceneAsset,
) -> None:
    del mapper, connection, target
    raise ValueError("Master scene assets are immutable")


@event.listens_for(StyleReferenceAsset, "before_update")
def _prevent_style_reference_update(
    mapper: object,
    connection: object,
    target: StyleReferenceAsset,
) -> None:
    del mapper, connection, target
    raise ValueError("Style reference assets are immutable")
