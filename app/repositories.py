"""Minimal repositories for persisted projects and scenes."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.beat_visual import ManualVisualQAOverride
from app.models.visual_qa import VisualQADecision, VisualQAResult
from app.persistence import (
    ApplicationSettings,
    BeatVisualQAEvaluation,
    BeatVisualResult,
    MasterSceneAsset,
    Project,
    ProjectNarrationAlignment,
    ProjectNarrationAsset,
    ProjectTimeline,
    ProjectVideoRender,
    ProjectVisualExecutionPlan,
    ProjectVisualPlan,
    Scene,
    StyleReferenceAsset,
    VisualBeatTiming,
    VisualOperationDecisionRecord,
)


def get_successful_video_render(
    session: Session,
    render_revision: str,
) -> ProjectVideoRender | None:
    return session.scalar(
        select(ProjectVideoRender)
        .where(
            ProjectVideoRender.render_revision == render_revision,
            ProjectVideoRender.status == "SUCCEEDED",
        )
        .order_by(ProjectVideoRender.attempt.desc())
    )


def list_project_video_renders(
    session: Session,
    project_id: str,
) -> list[ProjectVideoRender]:
    return list(
        session.scalars(
            select(ProjectVideoRender)
            .where(ProjectVideoRender.project_id == project_id)
            .order_by(ProjectVideoRender.created_at, ProjectVideoRender.id)
        )
    )


def create_pending_video_render(
    session: Session,
    **values: Any,
) -> ProjectVideoRender:
    revision = str(values["render_revision"])
    previous_attempt = session.scalar(
        select(func.max(ProjectVideoRender.attempt)).where(
            ProjectVideoRender.render_revision == revision
        )
    )
    record = ProjectVideoRender(
        **values,
        attempt=(previous_attempt or 0) + 1,
        status="PENDING",
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def fail_video_render(
    session: Session,
    record: ProjectVideoRender,
    *,
    safe_error: str,
    failed_stage: str,
    failed_entry_id: str | None = None,
    failed_beat_id: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> ProjectVideoRender:
    record.status = "FAILED"
    record.safe_error = safe_error
    record.failed_stage = failed_stage
    record.failed_entry_id = failed_entry_id
    record.failed_beat_id = failed_beat_id
    record.diagnostics = diagnostics
    record.completed_at = datetime.now(UTC)
    record.updated_at = record.completed_at
    session.commit()
    session.refresh(record)
    return record


def succeed_video_render(
    session: Session,
    record: ProjectVideoRender,
    *,
    output_path: str,
    output_sha256: str,
    duration: float,
    width: int,
    height: int,
    fps: float,
    diagnostics: dict[str, Any],
) -> ProjectVideoRender:
    now = datetime.now(UTC)
    record.status = "SUCCEEDED"
    record.output_path = output_path
    record.output_sha256 = output_sha256
    record.duration = duration
    record.width = width
    record.height = height
    record.fps = fps
    record.has_video = True
    record.has_audio = True
    record.safe_error = None
    record.diagnostics = diagnostics
    record.completed_at = now
    record.updated_at = now
    record.project.final_video_path = output_path
    record.project.rendered_at = now
    record.project.updated_at = now
    session.commit()
    session.refresh(record)
    return record

PROJECT_UPDATE_FIELDS = frozenset(
    {
        "name",
        "story_text",
        "global_image_style_prompt",
        "scene_count",
        "planning_provider",
        "planning_model",
        "visual_qa_enabled",
        "visual_qa_provider",
        "visual_qa_model",
        "style_id",
        "image_provider",
        "image_model",
        "tts_provider",
        "tts_model",
        "tts_voice",
        "tts_language",
        "width",
        "height",
        "fps",
        "image_fit",
        "final_video_path",
        "rendered_at",
        "generation_budget_enabled",
        "generation_budget_amount",
        "generation_budget_currency",
        "generation_budget_warning_threshold",
    }
)
SCENE_UPDATE_FIELDS = frozenset(
    {
        "text",
        "image_prompt",
        "image_path",
        "audio_path",
        "video_path",
        "duration",
    }
)


def get_application_settings(session: Session) -> ApplicationSettings:
    settings = session.get(ApplicationSettings, 1)
    if settings is None:
        settings = ApplicationSettings(id=1)
        session.add(settings)
        session.commit()
        session.refresh(settings)
    return settings


def update_application_settings(
    session: Session,
    *,
    image_provider: str,
    tts_provider: str,
    qwen_image_endpoint: str | None,
) -> ApplicationSettings:
    settings = get_application_settings(session)
    settings.image_provider = image_provider
    settings.tts_provider = tts_provider
    settings.qwen_image_endpoint = qwen_image_endpoint
    settings.updated_at = datetime.now(UTC)
    session.commit()
    session.refresh(settings)
    return settings


def create_project(
    session: Session,
    *,
    name: str,
    story_text: str,
    global_image_style_prompt: str | None = None,
    scene_count: int | None = None,
    planning_provider: str = "dashscope",
    planning_model: str = "qwen-plus",
    visual_qa_enabled: bool = True,
    visual_qa_provider: str = "dashscope",
    visual_qa_model: str = "qwen-vl-max",
    style_id: str = "rough_explainer_v1",
    image_provider: str = "seedream",
    image_model: str | None = None,
    tts_provider: str = "qwen",
    tts_model: str | None = None,
    tts_voice: str = "Cherry",
    tts_language: str = "Russian",
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    image_fit: str = "cover",
    final_video_path: str | None = None,
    rendered_at: datetime | None = None,
    generation_budget_enabled: bool = False,
    generation_budget_amount: float | None = None,
    generation_budget_currency: str = "EUR",
    generation_budget_warning_threshold: float = 0.8,
) -> Project:
    _validate_project_budget(
        generation_budget_enabled,
        generation_budget_amount,
    )
    project = Project(
        name=name,
        story_text=story_text,
        global_image_style_prompt=global_image_style_prompt,
        scene_count=scene_count,
        planning_provider=planning_provider,
        planning_model=planning_model,
        visual_qa_enabled=visual_qa_enabled,
        visual_qa_provider=visual_qa_provider,
        visual_qa_model=visual_qa_model,
        style_id=style_id,
        image_provider=image_provider,
        image_model=image_model,
        tts_provider=tts_provider,
        tts_model=tts_model,
        tts_voice=tts_voice,
        tts_language=tts_language,
        width=width,
        height=height,
        fps=fps,
        image_fit=image_fit,
        final_video_path=final_video_path,
        rendered_at=rendered_at,
        generation_budget_enabled=generation_budget_enabled,
        generation_budget_amount=generation_budget_amount,
        generation_budget_currency=generation_budget_currency,
        generation_budget_warning_threshold=generation_budget_warning_threshold,
    )
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


def get_project(session: Session, project_id: str) -> Project | None:
    return session.get(Project, project_id)


def list_projects(session: Session) -> list[Project]:
    statement = select(Project).order_by(Project.created_at, Project.id)
    return list(session.scalars(statement))


def update_project(
    session: Session,
    project_id: str,
    **changes: Any,
) -> Project | None:
    project = get_project(session, project_id)
    if project is None:
        return None
    _validate_update_fields(changes, PROJECT_UPDATE_FIELDS)
    _validate_project_budget(
        bool(changes.get("generation_budget_enabled", project.generation_budget_enabled)),
        changes.get("generation_budget_amount", project.generation_budget_amount),
    )
    for field, value in changes.items():
        setattr(project, field, value)
    project.updated_at = datetime.now(UTC)
    session.commit()
    session.refresh(project)
    return project


def _validate_project_budget(enabled: bool, amount: object) -> None:
    if enabled and amount is None:
        raise ValueError("generation_budget_amount is required when budget is enabled")


def delete_project(session: Session, project_id: str) -> bool:
    project = get_project(session, project_id)
    if project is None:
        return False
    session.delete(project)
    session.commit()
    return True


def get_project_visual_plan_record(
    session: Session,
    project_id: str,
) -> ProjectVisualPlan | None:
    """Load the current persisted visual-plan record for a Project."""
    statement = select(ProjectVisualPlan).where(
        ProjectVisualPlan.project_id == project_id
    )
    return session.scalar(statement)


def save_project_visual_plan_record(
    session: Session,
    *,
    project_id: str,
    schema_version: str,
    visual_director_version: str,
    story_text_hash: str,
    plan_json: dict[str, Any],
) -> ProjectVisualPlan:
    """Atomically create or replace a Project's validated visual plan."""
    project = get_project(session, project_id)
    if project is None:
        raise ValueError(f"Project not found: {project_id}")

    record = get_project_visual_plan_record(session, project_id)
    if record is None:
        record = ProjectVisualPlan(
            project=project,
            schema_version=schema_version,
            visual_director_version=visual_director_version,
            story_text_hash=story_text_hash,
            plan_json=plan_json,
        )
        session.add(record)
    else:
        record.schema_version = schema_version
        record.visual_director_version = visual_director_version
        record.story_text_hash = story_text_hash
        record.plan_json = plan_json
        record.updated_at = datetime.now(UTC)

    try:
        session.commit()
    except Exception:
        session.rollback()
        raise
    session.refresh(record)
    return record


def get_visual_execution_plan_by_revision(
    session: Session,
    resolution_revision: str,
) -> ProjectVisualExecutionPlan | None:
    statement = select(ProjectVisualExecutionPlan).where(
        ProjectVisualExecutionPlan.resolution_revision == resolution_revision
    )
    return session.scalar(statement)


def get_visual_execution_plan(
    session: Session,
    execution_plan_id: str,
) -> ProjectVisualExecutionPlan | None:
    return session.get(ProjectVisualExecutionPlan, execution_plan_id)


def get_visual_operation_decision(
    session: Session,
    execution_plan_id: str,
    beat_id: str,
) -> VisualOperationDecisionRecord | None:
    statement = select(VisualOperationDecisionRecord).where(
        VisualOperationDecisionRecord.execution_plan_id == execution_plan_id,
        VisualOperationDecisionRecord.beat_id == beat_id,
    )
    return session.scalar(statement)


def list_project_visual_execution_plans(
    session: Session,
    project_id: str,
) -> list[ProjectVisualExecutionPlan]:
    statement = (
        select(ProjectVisualExecutionPlan)
        .where(ProjectVisualExecutionPlan.project_id == project_id)
        .order_by(ProjectVisualExecutionPlan.created_at, ProjectVisualExecutionPlan.id)
    )
    return list(session.scalars(statement))


def save_visual_execution_plan(
    session: Session,
    *,
    project_id: str,
    visual_plan: ProjectVisualPlan,
    visual_plan_revision: str,
    provider: str,
    model: str | None,
    capability_snapshot: dict[str, Any],
    decision_input_snapshot: dict[str, Any],
    resolution_revision: str,
    decisions: list[dict[str, Any]],
) -> ProjectVisualExecutionPlan:
    """Atomically persist a complete provider-specific resolution."""
    existing = get_visual_execution_plan_by_revision(session, resolution_revision)
    if existing is not None:
        return existing

    record = ProjectVisualExecutionPlan(
        project_id=project_id,
        visual_plan=visual_plan,
        visual_plan_schema_version=visual_plan.schema_version,
        visual_plan_revision=visual_plan_revision,
        provider=provider,
        model=model,
        capability_snapshot=capability_snapshot,
        decision_input_snapshot=decision_input_snapshot,
        resolution_revision=resolution_revision,
    )
    record.decisions = [
        VisualOperationDecisionRecord(**decision) for decision in decisions
    ]
    session.add(record)
    try:
        session.commit()
    except Exception:
        session.rollback()
        raise
    session.refresh(record)
    return record


def list_beat_visual_results(
    session: Session,
    project_id: str,
    *,
    execution_plan_id: str | None = None,
    beat_id: str | None = None,
    accepted_only: bool = False,
) -> list[BeatVisualResult]:
    statement = select(BeatVisualResult).where(
        BeatVisualResult.project_id == project_id
    )
    if execution_plan_id is not None:
        statement = statement.where(
            BeatVisualResult.execution_plan_id == execution_plan_id
        )
    if beat_id is not None:
        statement = statement.where(BeatVisualResult.beat_id == beat_id)
    if accepted_only:
        statement = statement.where(BeatVisualResult.is_accepted.is_(True))
    statement = statement.order_by(
        BeatVisualResult.created_at,
        BeatVisualResult.id,
    )
    return list(session.scalars(statement))


def get_beat_visual_result(
    session: Session,
    result_id: str,
) -> BeatVisualResult | None:
    return session.get(BeatVisualResult, result_id)


def get_successful_beat_visual_result(
    session: Session,
    *,
    execution_plan_id: str,
    beat_id: str,
    generation_revision: str,
) -> BeatVisualResult | None:
    statement = (
        select(BeatVisualResult)
        .where(
            BeatVisualResult.execution_plan_id == execution_plan_id,
            BeatVisualResult.beat_id == beat_id,
            BeatVisualResult.generation_revision == generation_revision,
            BeatVisualResult.generation_status == "SUCCEEDED",
            BeatVisualResult.is_accepted.is_(True),
        )
        .order_by(BeatVisualResult.attempt.desc())
    )
    return session.scalar(statement)


def next_beat_visual_attempt(
    session: Session,
    *,
    execution_plan_id: str,
    beat_id: str,
    generation_revision: str,
) -> int:
    last_attempt = session.scalar(
        select(func.max(BeatVisualResult.attempt)).where(
            BeatVisualResult.execution_plan_id == execution_plan_id,
            BeatVisualResult.beat_id == beat_id,
            BeatVisualResult.generation_revision == generation_revision,
        )
    )
    return 1 if last_attempt is None else last_attempt + 1


def create_beat_visual_result(
    session: Session,
    **values: Any,
) -> BeatVisualResult:
    result = BeatVisualResult(**values)
    session.add(result)
    session.commit()
    session.refresh(result)
    return result


def mark_beat_visual_result_succeeded(
    session: Session,
    result: BeatVisualResult,
    *,
    output_path: str,
    file_sha256: str,
    accept: bool = True,
) -> BeatVisualResult:
    result.output_path = output_path
    result.file_sha256 = file_sha256
    result.generation_status = "SUCCEEDED"
    result.is_accepted = accept
    result.accepted_at = datetime.now(UTC) if accept else None
    result.error = None
    result.updated_at = datetime.now(UTC)
    session.commit()
    session.refresh(result)
    return result


def get_beat_visual_qa_evaluation_by_revision(
    session: Session,
    qa_revision: str,
) -> BeatVisualQAEvaluation | None:
    return session.scalar(
        select(BeatVisualQAEvaluation)
        .where(BeatVisualQAEvaluation.qa_revision == qa_revision)
        .order_by(BeatVisualQAEvaluation.created_at)
    )


def create_beat_visual_qa_evaluation(
    session: Session,
    candidate: BeatVisualResult,
    *,
    qa_revision: str,
    prompt_version: str,
    provider: str,
    model: str | None,
    qa_attempt: int,
    decision: VisualQADecision,
) -> BeatVisualQAEvaluation:
    evaluation = BeatVisualQAEvaluation(
        candidate=candidate,
        qa_revision=qa_revision,
        prompt_version=prompt_version,
        provider=provider,
        model=model,
        qa_attempt=qa_attempt,
        result=decision.result.value,
        scores=decision.scores.model_dump(mode="json"),
        problem_categories=[item.value for item in decision.problem_categories],
        reasons=list(decision.reasons),
        correction_instruction=decision.correction_instruction,
        severity=decision.severity.value if decision.severity is not None else None,
        decision_snapshot=decision.model_dump(mode="json"),
    )
    session.add(evaluation)
    session.commit()
    session.refresh(evaluation)
    return evaluation


def apply_automated_visual_qa_decision(
    session: Session,
    result: BeatVisualResult,
    decision: VisualQADecision,
    *,
    qa_revision: str,
    prompt_version: str,
    provider: str,
    model: str | None,
    qa_attempt: int,
    warning: str | None = None,
) -> BeatVisualResult:
    accepted = decision.result in {
        VisualQAResult.PASS,
        VisualQAResult.PASS_WITH_WARNING,
    }
    result.qa_status = decision.result.value
    result.qa_result = decision.result.value
    result.qa_scores = decision.scores.model_dump(mode="json")
    result.qa_problem_categories = [
        item.value for item in decision.problem_categories
    ]
    result.qa_reasons = list(decision.reasons)
    result.qa_correction_instruction = decision.correction_instruction
    result.qa_provider = provider
    result.qa_model = model
    result.qa_attempt = qa_attempt
    result.qa_revision = qa_revision
    result.qa_prompt_version = prompt_version
    result.qa_warning = warning
    result.is_accepted = accepted
    result.accepted_at = datetime.now(UTC) if accepted else None
    result.error = None
    result.updated_at = datetime.now(UTC)
    session.commit()
    session.refresh(result)
    return result


def set_manual_visual_qa_override(
    session: Session,
    result: BeatVisualResult,
    override: ManualVisualQAOverride,
    *,
    reason: str,
) -> BeatVisualResult:
    normalized_reason = reason.strip()
    if not normalized_reason:
        raise ValueError("manual QA override reason must not be empty")
    if result.generation_status != "SUCCEEDED":
        raise ValueError("only a generated candidate can receive a QA override")
    result.manual_qa_override = override.value
    result.manual_qa_reason = normalized_reason
    result.manual_qa_at = datetime.now(UTC)
    result.is_accepted = override is ManualVisualQAOverride.ACCEPTED
    result.accepted_at = datetime.now(UTC) if result.is_accepted else None
    result.updated_at = datetime.now(UTC)
    session.commit()
    session.refresh(result)
    return result


def get_project_narration_asset_by_revision(
    session: Session,
    generation_revision: str,
) -> ProjectNarrationAsset | None:
    return session.scalar(
        select(ProjectNarrationAsset).where(
            ProjectNarrationAsset.generation_revision == generation_revision
        )
    )


def list_project_narration_assets(
    session: Session,
    project_id: str,
) -> list[ProjectNarrationAsset]:
    return list(
        session.scalars(
            select(ProjectNarrationAsset)
            .where(ProjectNarrationAsset.project_id == project_id)
            .order_by(ProjectNarrationAsset.created_at, ProjectNarrationAsset.id)
        )
    )


def create_project_narration_asset(
    session: Session,
    **values: Any,
) -> ProjectNarrationAsset:
    asset = ProjectNarrationAsset(**values)
    session.add(asset)
    session.commit()
    session.refresh(asset)
    return asset


def get_narration_alignment_by_revision(
    session: Session,
    alignment_revision: str,
) -> ProjectNarrationAlignment | None:
    return session.scalar(
        select(ProjectNarrationAlignment).where(
            ProjectNarrationAlignment.alignment_revision == alignment_revision
        )
    )


def create_narration_alignment(
    session: Session,
    *,
    beat_timings: list[dict[str, Any]],
    **values: Any,
) -> ProjectNarrationAlignment:
    alignment = ProjectNarrationAlignment(**values)
    alignment.beat_timings = [VisualBeatTiming(**item) for item in beat_timings]
    session.add(alignment)
    session.commit()
    session.refresh(alignment)
    return alignment


def get_project_timeline_by_revision(
    session: Session,
    timeline_revision: str,
) -> ProjectTimeline | None:
    return session.scalar(
        select(ProjectTimeline).where(
            ProjectTimeline.timeline_revision == timeline_revision
        )
    )


def list_project_timelines(
    session: Session,
    project_id: str,
) -> list[ProjectTimeline]:
    return list(
        session.scalars(
            select(ProjectTimeline)
            .where(ProjectTimeline.project_id == project_id)
            .order_by(ProjectTimeline.created_at, ProjectTimeline.id)
        )
    )


def create_project_timeline(
    session: Session,
    *,
    entries: list[dict[str, Any]],
    **values: Any,
) -> ProjectTimeline:
    from app.persistence import TimelineEntry

    timeline = ProjectTimeline(**values)
    timeline.entries = [TimelineEntry(**item) for item in entries]
    session.add(timeline)
    session.commit()
    session.refresh(timeline)
    return timeline


def mark_beat_visual_result_failed(
    session: Session,
    result: BeatVisualResult,
    *,
    error: str,
) -> BeatVisualResult:
    result.generation_status = "FAILED"
    result.is_accepted = False
    result.error = error
    result.updated_at = datetime.now(UTC)
    session.commit()
    session.refresh(result)
    return result


def create_master_scene_asset(
    session: Session,
    *,
    project_id: str,
    master_scene_id: str,
    file_path: str,
    file_sha256: str,
    style_version: str,
    generation_prompt: str,
    provider: str,
    model: str | None = None,
    seed: int | None = None,
    reference_hashes: list[str] | None = None,
) -> MasterSceneAsset:
    """Persist a master once; replacement must never happen implicitly."""
    project = get_project(session, project_id)
    if project is None:
        raise ValueError(f"Project not found: {project_id}")
    if get_master_scene_asset(session, project_id, master_scene_id) is not None:
        raise ValueError(f"Master scene already exists: {master_scene_id}")

    asset = MasterSceneAsset(
        project=project,
        master_scene_id=master_scene_id,
        file_path=file_path,
        file_sha256=file_sha256,
        style_version=style_version,
        generation_prompt=generation_prompt,
        provider=provider,
        model=model,
        seed=seed,
        reference_hashes=list(reference_hashes or ()),
    )
    session.add(asset)
    session.commit()
    session.refresh(asset)
    return asset


def get_master_scene_asset(
    session: Session,
    project_id: str,
    master_scene_id: str,
) -> MasterSceneAsset | None:
    statement = select(MasterSceneAsset).where(
        MasterSceneAsset.project_id == project_id,
        MasterSceneAsset.master_scene_id == master_scene_id,
    )
    return session.scalar(statement)


def list_master_scene_assets(
    session: Session,
    project_id: str,
) -> list[MasterSceneAsset]:
    statement = (
        select(MasterSceneAsset)
        .where(MasterSceneAsset.project_id == project_id)
        .order_by(MasterSceneAsset.master_scene_id)
    )
    return list(session.scalars(statement))


def create_style_reference_asset(
    session: Session,
    *,
    project_id: str,
    style_id: str,
    file_path: str,
    file_sha256: str,
) -> StyleReferenceAsset:
    """Persist an explicitly approved style reference without replacement."""
    project = get_project(session, project_id)
    if project is None:
        raise ValueError(f"Project not found: {project_id}")
    if get_style_reference_asset(session, project_id, style_id) is not None:
        raise ValueError(f"Style reference already exists: {style_id}")
    asset = StyleReferenceAsset(
        project=project,
        style_id=style_id,
        file_path=file_path,
        file_sha256=file_sha256,
    )
    session.add(asset)
    session.commit()
    session.refresh(asset)
    return asset


def get_style_reference_asset(
    session: Session,
    project_id: str,
    style_id: str,
) -> StyleReferenceAsset | None:
    statement = select(StyleReferenceAsset).where(
        StyleReferenceAsset.project_id == project_id,
        StyleReferenceAsset.style_id == style_id,
    )
    return session.scalar(statement)


def create_scene(
    session: Session,
    *,
    project_id: str,
    text: str,
    image_prompt: str,
    image_path: str | None = None,
    audio_path: str | None = None,
    video_path: str | None = None,
    duration: float | None = None,
) -> Scene:
    project = get_project(session, project_id)
    if project is None:
        raise ValueError(f"Project not found: {project_id}")
    last_position = session.scalar(
        select(func.max(Scene.position)).where(Scene.project_id == project_id)
    )
    position = 0 if last_position is None else last_position + 1
    scene = Scene(
        project=project,
        position=position,
        text=text,
        image_prompt=image_prompt,
        image_path=image_path,
        audio_path=audio_path,
        video_path=video_path,
        duration=duration,
    )
    session.add(scene)
    session.commit()
    session.refresh(scene)
    return scene


def get_scene(session: Session, scene_id: str) -> Scene | None:
    return session.get(Scene, scene_id)


def list_scenes(session: Session, project_id: str) -> list[Scene]:
    statement = (
        select(Scene)
        .where(Scene.project_id == project_id)
        .order_by(Scene.position, Scene.id)
    )
    return list(session.scalars(statement))


def update_scene(
    session: Session,
    scene_id: str,
    **changes: Any,
) -> Scene | None:
    scene = get_scene(session, scene_id)
    if scene is None:
        return None
    _validate_update_fields(changes, SCENE_UPDATE_FIELDS)
    for field, value in changes.items():
        setattr(scene, field, value)
    scene.updated_at = datetime.now(UTC)
    session.commit()
    session.refresh(scene)
    return scene


def delete_scene(session: Session, scene_id: str) -> bool:
    scene = get_scene(session, scene_id)
    if scene is None:
        return False
    project_id = scene.project_id
    session.delete(scene)
    session.flush()
    _normalize_scene_positions(session, project_id)
    session.commit()
    return True


def move_scene(
    session: Session,
    scene_id: str,
    direction: str,
) -> Scene | None:
    """Move a scene one position while preserving a gap-free order."""
    scene = get_scene(session, scene_id)
    if scene is None:
        return None
    offset = {"up": -1, "down": 1}.get(direction)
    if offset is None:
        raise ValueError("direction must be 'up' or 'down'")

    scenes = list_scenes(session, scene.project_id)
    current_index = next(
        index for index, item in enumerate(scenes) if item.id == scene.id
    )
    target_index = current_index + offset
    if not 0 <= target_index < len(scenes):
        return scene

    other = scenes[target_index]
    temporary_position = len(scenes)
    scene.position = temporary_position
    session.flush()
    other.position = current_index
    session.flush()
    scene.position = target_index
    session.commit()
    session.refresh(scene)
    return scene


def _normalize_scene_positions(session: Session, project_id: str) -> None:
    scenes = list_scenes(session, project_id)
    for position, scene in enumerate(scenes):
        scene.position = position
    session.flush()


def _validate_update_fields(
    changes: dict[str, Any],
    allowed_fields: frozenset[str],
) -> None:
    unknown_fields = changes.keys() - allowed_fields
    if unknown_fields:
        fields = ", ".join(sorted(unknown_fields))
        raise ValueError(f"Unsupported update fields: {fields}")
