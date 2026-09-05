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
    Numeric,
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

    planning_provider: Mapped[str] = mapped_column(
        String(50), nullable=False, default="dashscope"
    )
    planning_model: Mapped[str] = mapped_column(
        String(255), nullable=False, default="qwen-plus"
    )
    visual_qa_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    visual_qa_provider: Mapped[str] = mapped_column(
        String(50), nullable=False, default="dashscope"
    )
    visual_qa_model: Mapped[str] = mapped_column(
        String(255), nullable=False, default="qwen-vl-max"
    )
    style_id: Mapped[str] = mapped_column(
        String(64), nullable=False, default="rough_explainer_v1"
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
    narration_assets: Mapped[list[ProjectNarrationAsset]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="ProjectNarrationAsset.created_at",
    )
    narration_alignments: Mapped[list[ProjectNarrationAlignment]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="ProjectNarrationAlignment.created_at",
    )
    timelines: Mapped[list[ProjectTimeline]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="ProjectTimeline.created_at",
    )
    video_renders: Mapped[list[ProjectVideoRender]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="ProjectVideoRender.created_at",
    )
    provider_usage_records: Mapped[list[ProviderUsageRecord]] = relationship(
        back_populates="project",
        passive_deletes=True,
        order_by="ProviderUsageRecord.created_at",
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

    @validates("planning_provider", "visual_qa_provider")
    def validate_structured_provider(self, key: str, value: str) -> str:
        del key
        normalized = value.strip().lower()
        if normalized != "dashscope":
            raise ValueError("unsupported structured AI provider")
        return normalized

    @validates("planning_model", "visual_qa_model", "style_id")
    def validate_pipeline_identifier(self, key: str, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{key} must not be empty")
        return normalized


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


class ProjectNarrationAsset(Base):
    """One immutable successful story-level narration audio revision."""

    __tablename__ = "project_narration_assets"
    __table_args__ = (
        UniqueConstraint(
            "generation_revision",
            name="uq_project_narration_assets_revision",
        ),
        CheckConstraint("duration > 0", name="ck_project_narration_duration"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    story_text_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    voice: Mapped[str] = mapped_column(String(100), nullable=False)
    language: Mapped[str | None] = mapped_column(String(100), nullable=True)
    capability_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    audio_path: Mapped[str] = mapped_column(Text, nullable=False)
    audio_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    duration: Mapped[float] = mapped_column(Float, nullable=False)
    generation_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    timestamp_source: Mapped[str] = mapped_column(String(50), nullable=False)
    timing_granularity: Mapped[str] = mapped_column(String(32), nullable=False)
    timing_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    timing_data: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    timing_warnings: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=_utc_now, nullable=False
    )

    project: Mapped[Project] = relationship(back_populates="narration_assets")
    alignments: Mapped[list[ProjectNarrationAlignment]] = relationship(
        back_populates="narration_asset"
    )
    timelines: Mapped[list[ProjectTimeline]] = relationship(
        back_populates="narration_asset"
    )


class ProjectNarrationAlignment(Base):
    """Runtime mapping of semantic beats onto one narration audio revision."""

    __tablename__ = "project_narration_alignments"
    __table_args__ = (
        UniqueConstraint(
            "alignment_revision",
            name="uq_project_narration_alignments_revision",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    visual_plan_id: Mapped[str] = mapped_column(
        ForeignKey("project_visual_plans.id", ondelete="CASCADE"), nullable=False
    )
    visual_plan_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    narration_asset_id: Mapped[str] = mapped_column(
        ForeignKey("project_narration_assets.id", ondelete="CASCADE"), nullable=False
    )
    alignment_version: Mapped[str] = mapped_column(String(64), nullable=False)
    alignment_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    warnings: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    average_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=_utc_now, nullable=False
    )

    project: Mapped[Project] = relationship(back_populates="narration_alignments")
    narration_asset: Mapped[ProjectNarrationAsset] = relationship(
        back_populates="alignments"
    )
    beat_timings: Mapped[list[VisualBeatTiming]] = relationship(
        back_populates="alignment",
        cascade="all, delete-orphan",
        order_by="VisualBeatTiming.position",
    )
    timelines: Mapped[list[ProjectTimeline]] = relationship(
        back_populates="alignment"
    )


class VisualBeatTiming(Base):
    __tablename__ = "visual_beat_timings"
    __table_args__ = (
        UniqueConstraint(
            "alignment_id", "beat_id", name="uq_visual_beat_timings_alignment_beat"
        ),
        UniqueConstraint(
            "alignment_id", "position", name="uq_visual_beat_timings_alignment_position"
        ),
        CheckConstraint("audio_start >= 0", name="ck_visual_beat_timing_start"),
        CheckConstraint("audio_end > audio_start", name="ck_visual_beat_timing_range"),
        CheckConstraint(
            "source_start_char >= 0",
            name="ck_visual_beat_timing_source_start",
        ),
        CheckConstraint(
            "source_end_char > source_start_char",
            name="ck_visual_beat_timing_source_range",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    alignment_id: Mapped[str] = mapped_column(
        ForeignKey("project_narration_alignments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    beat_id: Mapped[str] = mapped_column(String(255), nullable=False)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_start_char: Mapped[int] = mapped_column(Integer, nullable=False)
    source_end_char: Mapped[int] = mapped_column(Integer, nullable=False)
    audio_start: Mapped[float] = mapped_column(Float, nullable=False)
    audio_end: Mapped[float] = mapped_column(Float, nullable=False)
    alignment_method: Mapped[str] = mapped_column(String(50), nullable=False)
    alignment_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    warning: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=_utc_now, nullable=False
    )

    alignment: Mapped[ProjectNarrationAlignment] = relationship(
        back_populates="beat_timings"
    )


class ProjectTimeline(Base):
    """Versioned executable edit specification; it contains no rendered video."""

    __tablename__ = "project_timelines"
    __table_args__ = (
        UniqueConstraint("timeline_revision", name="uq_project_timelines_revision"),
        CheckConstraint("duration > 0", name="ck_project_timeline_duration"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    visual_plan_id: Mapped[str] = mapped_column(
        ForeignKey("project_visual_plans.id", ondelete="CASCADE"), nullable=False
    )
    visual_plan_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    execution_plan_id: Mapped[str] = mapped_column(
        ForeignKey("project_visual_execution_plans.id", ondelete="CASCADE"),
        nullable=False,
    )
    execution_plan_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    narration_asset_id: Mapped[str] = mapped_column(
        ForeignKey("project_narration_assets.id", ondelete="CASCADE"), nullable=False
    )
    alignment_id: Mapped[str] = mapped_column(
        ForeignKey("project_narration_alignments.id", ondelete="CASCADE"), nullable=False
    )
    rhythm_version: Mapped[str] = mapped_column(String(64), nullable=False)
    timeline_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    duration: Mapped[float] = mapped_column(Float, nullable=False)
    warnings: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=_utc_now, nullable=False
    )

    project: Mapped[Project] = relationship(back_populates="timelines")
    narration_asset: Mapped[ProjectNarrationAsset] = relationship(
        back_populates="timelines"
    )
    alignment: Mapped[ProjectNarrationAlignment] = relationship(
        back_populates="timelines"
    )
    entries: Mapped[list[TimelineEntry]] = relationship(
        back_populates="timeline",
        cascade="all, delete-orphan",
        order_by="TimelineEntry.position",
    )


class TimelineEntry(Base):
    __tablename__ = "timeline_entries"
    __table_args__ = (
        UniqueConstraint("timeline_id", "beat_id", name="uq_timeline_entries_beat"),
        UniqueConstraint("timeline_id", "position", name="uq_timeline_entries_position"),
        CheckConstraint("start_time >= 0", name="ck_timeline_entry_start"),
        CheckConstraint("end_time > start_time", name="ck_timeline_entry_range"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    timeline_id: Mapped[str] = mapped_column(
        ForeignKey("project_timelines.id", ondelete="CASCADE"), nullable=False, index=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    beat_id: Mapped[str] = mapped_column(String(255), nullable=False)
    beat_visual_result_id: Mapped[str | None] = mapped_column(
        ForeignKey("beat_visual_results.id", ondelete="SET NULL"), nullable=True
    )
    operation: Mapped[str] = mapped_column(String(32), nullable=False)
    asset_path: Mapped[str] = mapped_column(Text, nullable=False)
    source_result_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    master_scene_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    transform_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    overlay_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    transition_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=_utc_now, nullable=False
    )

    timeline: Mapped[ProjectTimeline] = relationship(back_populates="entries")


class ProjectVideoRender(Base):
    """One persisted attempt to deterministically execute a ProjectTimeline."""

    __tablename__ = "project_video_renders"
    __table_args__ = (
        UniqueConstraint(
            "render_revision", "attempt", name="uq_project_video_render_attempt"
        ),
        CheckConstraint("attempt >= 1", name="ck_project_video_render_attempt"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    timeline_id: Mapped[str] = mapped_column(
        ForeignKey("project_timelines.id", ondelete="CASCADE"), nullable=False, index=True
    )
    timeline_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    render_config_version: Mapped[str] = mapped_column(String(64), nullable=False)
    render_config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    renderer_version: Mapped[str] = mapped_column(String(64), nullable=False)
    render_revision: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    output_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    has_video: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    has_audio: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    safe_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    failed_entry_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    failed_beat_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    failed_stage: Mapped[str | None] = mapped_column(String(100), nullable=True)
    diagnostics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=_utc_now, nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=_utc_now, onupdate=_utc_now, nullable=False
    )

    project: Mapped[Project] = relationship(back_populates="video_renders")


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


class ProviderPricing(Base):
    """Immutable, versioned price used to value provider usage."""

    __tablename__ = "provider_pricing"
    __table_args__ = (
        UniqueConstraint(
            "provider", "model", "operation", "pricing_unit", "version",
            name="uq_provider_pricing_version",
        ),
        CheckConstraint("price >= 0", name="ck_provider_pricing_non_negative"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    provider: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    operation: Mapped[str] = mapped_column(String(100), nullable=False)
    pricing_unit: Mapped[str] = mapped_column(String(50), nullable=False)
    price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    effective_from: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=_utc_now, nullable=False
    )


class ProviderUsageRecord(Base):
    """Immutable ledger entry for one provider request or explicit free reuse."""

    __tablename__ = "provider_usage_records"
    __table_args__ = (
        UniqueConstraint(
            "job_id", "pipeline_stage", "request_revision",
            name="uq_provider_usage_request",
        ),
        CheckConstraint("input_units >= 0", name="ck_provider_usage_input_units"),
        CheckConstraint("output_units >= 0", name="ck_provider_usage_output_units"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    job_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    pipeline_stage: Mapped[str] = mapped_column(String(100), nullable=False)
    beat_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    master_scene_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    operation: Mapped[str] = mapped_column(String(100), nullable=False)
    request_revision: Mapped[str] = mapped_column(String(255), nullable=False)
    estimated_cost: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)
    actual_cost: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)
    currency: Mapped[str | None] = mapped_column(String(3), nullable=True)
    pricing_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    input_units: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    output_units: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    unit_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    cost_certainty: Mapped[str] = mapped_column(String(32), nullable=False)
    is_qa_retry: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=_utc_now, nullable=False
    )

    project: Mapped[Project] = relationship(back_populates="provider_usage_records")


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


@event.listens_for(ProviderPricing, "before_update")
@event.listens_for(ProviderPricing, "before_delete")
def _prevent_pricing_mutation(mapper: object, connection: object, target: object) -> None:
    del mapper, connection, target
    raise ValueError("Provider pricing versions are immutable")


@event.listens_for(ProviderUsageRecord, "before_update")
def _prevent_usage_mutation(mapper: object, connection: object, target: object) -> None:
    del mapper, connection, target
    raise ValueError("Provider usage records are immutable")
