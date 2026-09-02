"""Persisted Project and Scene entities."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
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
    visual_plan: Mapped[ProjectVisualPlan | None] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        uselist=False,
    )
    visual_execution_plans: Mapped[list[ProjectVisualExecutionPlan]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="ProjectVisualExecutionPlan.created_at",
    )
    beat_visual_results: Mapped[list[BeatVisualResult]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="BeatVisualResult.created_at",
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


class ProjectVisualPlan(Base):
    """Current validated semantic visual plan for one Project."""

    __tablename__ = "project_visual_plans"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            name="uq_project_visual_plans_project",
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
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    visual_director_version: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    story_text_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    plan_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
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

    project: Mapped[Project] = relationship(back_populates="visual_plan")
    execution_plans: Mapped[list[ProjectVisualExecutionPlan]] = relationship(
        back_populates="visual_plan",
        cascade="all, delete-orphan",
        order_by="ProjectVisualExecutionPlan.created_at",
    )


class ProjectVisualExecutionPlan(Base):
    """Provider-specific operation resolution derived from a semantic plan."""

    __tablename__ = "project_visual_execution_plans"
    __table_args__ = (
        UniqueConstraint(
            "resolution_revision",
            name="uq_project_visual_execution_plans_revision",
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
    visual_plan_id: Mapped[str] = mapped_column(
        ForeignKey("project_visual_plans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    visual_plan_schema_version: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    visual_plan_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    capability_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
    )
    decision_input_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
    )
    resolution_revision: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        default=_utc_now,
        nullable=False,
    )

    project: Mapped[Project] = relationship(
        back_populates="visual_execution_plans"
    )
    visual_plan: Mapped[ProjectVisualPlan] = relationship(
        back_populates="execution_plans"
    )
    decisions: Mapped[list[VisualOperationDecisionRecord]] = relationship(
        back_populates="execution_plan",
        cascade="all, delete-orphan",
        order_by="VisualOperationDecisionRecord.position",
    )
    beat_visual_results: Mapped[list[BeatVisualResult]] = relationship(
        back_populates="execution_plan",
        cascade="all, delete-orphan",
        order_by="BeatVisualResult.created_at",
    )


class VisualOperationDecisionRecord(Base):
    """Durable resolved operation for one semantic visual beat."""

    __tablename__ = "visual_operation_decisions"
    __table_args__ = (
        UniqueConstraint(
            "execution_plan_id",
            "beat_id",
            name="uq_visual_operation_decisions_plan_beat",
        ),
        UniqueConstraint(
            "execution_plan_id",
            "position",
            name="uq_visual_operation_decisions_plan_position",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=_generate_uuid,
    )
    execution_plan_id: Mapped[str] = mapped_column(
        ForeignKey("project_visual_execution_plans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    beat_id: Mapped[str] = mapped_column(String(255), nullable=False)
    preferred_operation: Mapped[str] = mapped_column(String(32), nullable=False)
    resolved_operation: Mapped[str] = mapped_column(String(32), nullable=False)
    fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False)
    fallback_from: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reason: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    source_visual_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    source_image_paths: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        default=_utc_now,
        nullable=False,
    )

    execution_plan: Mapped[ProjectVisualExecutionPlan] = relationship(
        back_populates="decisions"
    )


class BeatVisualResult(Base):
    """One durable attempt to execute a resolved visual beat operation."""

    __tablename__ = "beat_visual_results"
    __table_args__ = (
        UniqueConstraint(
            "execution_plan_id",
            "beat_id",
            "generation_revision",
            "attempt",
            name="uq_beat_visual_results_revision_attempt",
        ),
        CheckConstraint("attempt >= 1", name="ck_beat_visual_results_attempt"),
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
    visual_plan_id: Mapped[str] = mapped_column(
        ForeignKey("project_visual_plans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    visual_plan_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    execution_plan_id: Mapped[str] = mapped_column(
        ForeignKey("project_visual_execution_plans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    beat_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    resolved_operation: Mapped[str] = mapped_column(String(32), nullable=False)
    source_result_id: Mapped[str | None] = mapped_column(
        ForeignKey("beat_visual_results.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    source_master_asset_id: Mapped[str | None] = mapped_column(
        ForeignKey("master_scene_assets.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    master_scene_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    prompt_used: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    style_version: Mapped[str] = mapped_column(String(64), nullable=False)
    reference_snapshot: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    generation_status: Mapped[str] = mapped_column(String(32), nullable=False)
    qa_status: Mapped[str] = mapped_column(String(32), nullable=False)
    qa_result: Mapped[str | None] = mapped_column(String(32), nullable=True)
    qa_scores: Mapped[dict[str, float] | None] = mapped_column(JSON, nullable=True)
    qa_problem_categories: Mapped[list[str] | None] = mapped_column(
        JSON,
        nullable=True,
    )
    qa_reasons: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    qa_correction_instruction: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    qa_provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    qa_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    qa_attempt: Mapped[int | None] = mapped_column(Integer, nullable=True)
    qa_revision: Mapped[str | None] = mapped_column(String(64), nullable=True)
    qa_prompt_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    qa_warning: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_accepted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    accepted_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
    )
    manual_qa_override: Mapped[str | None] = mapped_column(String(32), nullable=True)
    manual_qa_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    manual_qa_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    transform_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )
    overlay_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )
    semantic_state_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
    )
    generation_revision: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
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

    project: Mapped[Project] = relationship(back_populates="beat_visual_results")
    execution_plan: Mapped[ProjectVisualExecutionPlan] = relationship(
        back_populates="beat_visual_results"
    )
    source_result: Mapped[BeatVisualResult | None] = relationship(
        remote_side="BeatVisualResult.id",
        foreign_keys=[source_result_id],
    )
    qa_evaluations: Mapped[list[BeatVisualQAEvaluation]] = relationship(
        back_populates="candidate",
        cascade="all, delete-orphan",
        order_by="BeatVisualQAEvaluation.created_at",
    )


class BeatVisualQAEvaluation(Base):
    """Immutable automated QA evaluation for one generated candidate revision."""

    __tablename__ = "beat_visual_qa_evaluations"
    __table_args__ = (
        UniqueConstraint(
            "candidate_result_id",
            "qa_revision",
            name="uq_beat_visual_qa_candidate_revision",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=_generate_uuid,
    )
    candidate_result_id: Mapped[str] = mapped_column(
        ForeignKey("beat_visual_results.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    qa_revision: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    qa_attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    result: Mapped[str] = mapped_column(String(32), nullable=False)
    scores: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    problem_categories: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    reasons: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    correction_instruction: Mapped[str | None] = mapped_column(Text, nullable=True)
    severity: Mapped[str | None] = mapped_column(String(32), nullable=True)
    decision_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        default=_utc_now,
        nullable=False,
    )

    candidate: Mapped[BeatVisualResult] = relationship(
        back_populates="qa_evaluations"
    )


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
