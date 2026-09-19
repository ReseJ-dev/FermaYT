"""Resumable Stage 1–6 orchestration for one-click Project video generation."""

from __future__ import annotations

import asyncio
import shutil
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.budgets import ProjectBudgetGuard
from app.costs import (
    PricingUnit,
    UsageStatus,
    estimate_project_generation_cost,
    record_provider_usage,
    summarize_project_cost,
    usage_revision,
)
from app.generation_scope import GenerationScope, GenerationScopeType, PlanningScope
from app.generators.master_scene import generate_required_master_scenes
from app.models.render import ProjectRenderConfig
from app.persistence import ProviderUsageRecord
from app.pipeline.visual_qa import VisualQAService
from app.production_profiles import ProductionProfile
from app.providers import (
    ImageProvider,
    TTSProvider,
    get_image_provider_capabilities,
)
from app.repositories import (
    get_project,
    get_project_visual_plan_record,
    get_style_reference_asset,
    list_beat_visual_results,
    list_master_scene_assets,
    list_project_video_renders,
    set_active_project_visual_plan,
)
from app.services.narration import generate_project_narration
from app.services.narration_alignment import align_project_visual_beats
from app.services.project_render import render_project_video
from app.services.timeline import build_project_timeline, build_timeline_quality_report
from app.services.visual_asset_execution import (
    VisualBeatAssetExecutor,
    build_visual_qa_execution_summary,
)
from app.services.visual_operations import resolve_project_visual_operations
from app.services.visual_planning import (
    create_project_visual_plan,
    hash_story_text,
    is_visual_plan_record_compatible,
    load_project_visual_plan_state,
)
from app.style_contracts import get_image_style_contract
from app.utils.download import download_file

PIPELINE_VERSION = "project_video_pipeline_v1"


class ProjectPipelineStage(str, Enum):
    VALIDATING = "VALIDATING"
    PLANNING = "PLANNING"
    RESOLVING_VISUALS = "RESOLVING_VISUALS"
    GENERATING_MASTERS = "GENERATING_MASTERS"
    GENERATING_VISUALS = "GENERATING_VISUALS"
    VISUAL_QA = "VISUAL_QA"
    GENERATING_NARRATION = "GENERATING_NARRATION"
    ALIGNING_NARRATION = "ALIGNING_NARRATION"
    BUILDING_TIMELINE = "BUILDING_TIMELINE"
    RENDERING_VIDEO = "RENDERING_VIDEO"
    VALIDATING_VIDEO = "VALIDATING_VIDEO"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


ImageProviderResolver = Callable[[str, Mapping[str, Any] | None], ImageProvider]
TTSProviderResolver = Callable[[str, Mapping[str, Any] | None], TTSProvider]
PipelineProgress = Callable[
    [ProjectPipelineStage, int, int, str, int | None, int | None, str | None],
    Awaitable[None],
]


@dataclass(frozen=True, slots=True)
class ProjectPipelineDependencies:
    planning_client: Any
    image_provider_resolver: ImageProviderResolver
    tts_provider_resolver: TTSProviderResolver
    visual_qa_service: VisualQAService | None
    projects_root: str | Path = "data/projects"
    downloader: Callable[[str, str], Awaitable[str]] = download_file
    duration_probe: Callable[[str | Path], float] | None = None
    preflight_validator: Callable[[Any], None] | None = None


@dataclass(frozen=True, slots=True)
class ProjectPipelineReport:
    pipeline_version: str
    project_id: str
    story_revision: str
    visual_beats: int
    master_assets: int
    generated_or_edited_candidates: int
    qa_retries: int
    accepted_visual_assets: int
    narration_duration: float
    timeline_entries: int
    render_duration: float
    final_mp4: str
    final_render_id: str
    reused: dict[str, int | bool]
    estimated_cost_before_run: dict[str, Any]
    actual_run_cost: float | None
    historical_project_asset_cost: float | None
    qa_retry_cost: float | None
    cost_by_stage: dict[str, float]
    cost_by_provider: dict[str, float]
    cost_by_model: dict[str, float]
    cost_by_beat: dict[str, float]
    cost_currency: str | None
    unpriced_usage_records: int
    cost_run_id: str
    generation_budget: dict[str, object]
    production_profile: str
    production_profile_version: str
    paid_visual_beats: int
    free_visual_beats: int
    visual_qa_calls: int
    projected_final_cost: dict[str, Any] | None
    generation_scope: dict[str, str | float | int | None]
    semantic_visual_beats: int
    timeline_quality: dict[str, Any]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProjectStylePreviewReport:
    pipeline_version: str
    project_id: str
    story_revision: str
    visual_beats: int
    semantic_visual_beats: int
    master_assets: int
    generated_or_edited_candidates: int
    accepted_visual_assets: int
    preview_result_ids: tuple[str, ...]
    reused: dict[str, int | bool]
    estimated_cost_before_run: dict[str, Any]
    actual_run_cost: float | None
    historical_project_asset_cost: float | None
    qa_retry_cost: float | None
    cost_currency: str | None
    unpriced_usage_records: int
    cost_run_id: str
    generation_budget: dict[str, object]
    production_profile: str
    production_profile_version: str
    generation_scope: dict[str, str | float | int | None]
    final_render_id: None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


async def run_project_video_pipeline(
    session: Session,
    project_id: str,
    dependencies: ProjectPipelineDependencies,
    *,
    progress: PipelineProgress | None = None,
    job_id: str | None = None,
    budget_override: bool = False,
    production_profile: ProductionProfile | str = ProductionProfile.FINAL,
    generation_scope: GenerationScope | None = None,
    planning_run_id: str | None = None,
    allow_planning_retry_after_uncertain: bool = False,
    cancellation_requested: Callable[[], bool] | None = None,
) -> ProjectPipelineReport | ProjectStylePreviewReport:
    """Run every required current stage, reusing valid persisted revisions."""
    cost_run_id = job_id or f"direct-{uuid4()}"
    profile = ProductionProfile(production_profile)
    scope = generation_scope or GenerationScope.full()
    progress_callback = progress or _ignore_progress

    async def emit(
        stage: ProjectPipelineStage,
        overall: int,
        stage_progress: int,
        message: str,
        current_beat: int | None,
        total_beats: int | None,
        failed_beat: str | None,
    ) -> None:
        if cancellation_requested is not None and cancellation_requested():
            raise asyncio.CancelledError
        await progress_callback(
            stage,
            overall,
            stage_progress,
            message,
            current_beat,
            total_beats,
            failed_beat,
        )

    await emit(
        ProjectPipelineStage.VALIDATING, 2, 0, "Проверка проекта", None, None, None
    )
    project = get_project(session, project_id)
    if project is None:
        raise ValueError("Project not found")
    _validate_project_preflight(
        project,
        dependencies,
        require_video_tools=not scope.is_image_only,
    )
    budget_guard = ProjectBudgetGuard(session, project_id, override=budget_override)
    narration_kwargs: dict[str, Any] = {
        "provider_resolver": dependencies.tts_provider_resolver,
        "downloader": dependencies.downloader,
        "projects_root": dependencies.projects_root,
        "job_id": cost_run_id,
        "budget_guard": budget_guard,
    }
    if dependencies.duration_probe is not None:
        narration_kwargs["duration_probe"] = dependencies.duration_probe
    narration = None
    await emit(
        ProjectPipelineStage.VALIDATING, 5, 100, "Проект готов", None, None, None
    )

    planning_scope = PlanningScope.derive(project.story_text, scope)
    existing_plan = load_project_visual_plan_state(
        session, project_id, scope_key=planning_scope.scope_key
    )
    plan_record = get_project_visual_plan_record(
        session, project_id, scope_key=planning_scope.scope_key
    )
    plan_reused = (
        existing_plan is not None
        and existing_plan.is_current
        and plan_record is not None
        and is_visual_plan_record_compatible(plan_record)
    )
    await emit(
        ProjectPipelineStage.PLANNING,
        8,
        0,
        "Планирование визуальной истории",
        None,
        None,
        None,
    )
    if plan_reused:
        plan = existing_plan.plan
        assert plan_record is not None
        set_active_project_visual_plan(session, project_id, plan_record.id)
        record_provider_usage(
            session,
            project_id=project_id,
            job_id=cost_run_id,
            pipeline_stage="PLANNING",
            provider=project.planning_provider,
            model=project.planning_model,
            operation="PLANNING",
            request_revision=usage_revision(
                hash_story_text(project.story_text), "planning"
            ),
            unit_type=PricingUnit.PER_REQUEST,
            input_units=1,
            status=UsageStatus.CACHED,
        )
    else:
        plan = await create_project_visual_plan(
            session,
            project_id,
            dependencies.planning_client,
            job_id=cost_run_id,
            budget_guard=budget_guard,
            planning_run_id=planning_run_id,
            allow_after_uncertain=allow_planning_retry_after_uncertain,
            cancellation_requested=cancellation_requested,
            planning_scope=planning_scope,
        )
    semantic_total_beats = len(plan.visual_beats)
    selected_beat_ids = scope.select_beat_ids(plan)
    selected_beat_id_set = frozenset(selected_beat_ids)
    total_beats = len(selected_beat_ids)
    await emit(
        ProjectPipelineStage.PLANNING,
        15,
        100,
        f"Визуальный план: {semantic_total_beats} beats; выбрано {total_beats}",
        None,
        total_beats,
        None,
    )

    await emit(
        ProjectPipelineStage.RESOLVING_VISUALS,
        18,
        0,
        "Выбор visual operations",
        None,
        total_beats,
        None,
    )
    execution = resolve_project_visual_operations(
        session,
        project_id,
        provider_resolver=dependencies.image_provider_resolver,
        production_profile=profile,
    )
    provider = dependencies.image_provider_resolver(
        project.image_provider,
        {"model": project.image_model} if project.image_model else None,
    )
    capabilities = get_image_provider_capabilities(provider)
    estimate = estimate_project_generation_cost(
        session,
        project_id,
        production_profile=profile.value,
        beat_ids=selected_beat_id_set,
    )
    budget_guard.check_preflight(estimate)
    if scope.type is GenerationScopeType.FIRST_SECONDS:
        narration = await generate_project_narration(
            session, project_id, **narration_kwargs
        )
        alignment = align_project_visual_beats(session, project_id, narration)
        selected_beat_ids = scope.select_aligned_beat_ids(
            plan,
            {item.beat_id: item.audio_end for item in alignment.beat_timings},
        )
        selected_beat_id_set = frozenset(selected_beat_ids)
        total_beats = len(selected_beat_ids)
        estimate = estimate_project_generation_cost(
            session,
            project_id,
            production_profile=profile.value,
            beat_ids=selected_beat_id_set,
        )
        budget_guard.check_preflight(estimate)
    style_reference = get_style_reference_asset(session, project_id, project.style_id)
    await emit(
        ProjectPipelineStage.RESOLVING_VISUALS,
        22,
        100,
        "Visual operations сохранены",
        None,
        total_beats,
        None,
    )

    masters_before = len(list_master_scene_assets(session, project_id))
    await emit(
        ProjectPipelineStage.GENERATING_MASTERS,
        24,
        0,
        "Подготовка master scenes",
        None,
        total_beats,
        None,
    )
    if not capabilities.reference_generation:
        # Text-only providers receive the immutable master geometry through every
        # semantic beat prompt. Do not buy PNG masters they cannot attach later.
        masters = list_master_scene_assets(session, project_id)
    else:
        masters = await generate_required_master_scenes(
            session,
            project,
            plan,
            provider,
            projects_root=dependencies.projects_root,
            style_id=project.style_id,
            style_reference=style_reference,
            capabilities=capabilities,
            qa_service=dependencies.visual_qa_service,
            downloader=dependencies.downloader,
            job_id=cost_run_id,
            budget_guard=budget_guard,
            required_beat_ids=selected_beat_id_set,
        )
    execution = resolve_project_visual_operations(
        session,
        project_id,
        provider_resolver=dependencies.image_provider_resolver,
        production_profile=profile,
    )
    await emit(
        ProjectPipelineStage.GENERATING_MASTERS,
        30,
        100,
        f"Master scenes: {len(masters)}",
        None,
        total_beats,
        None,
    )

    accepted_before = {
        item.id
        for item in list_beat_visual_results(session, project_id, accepted_only=True)
    }
    executor = VisualBeatAssetExecutor(
        session,
        provider_resolver=dependencies.image_provider_resolver,
        downloader=dependencies.downloader,
        projects_root=dependencies.projects_root,
        style_id=project.style_id,
        qa_service=dependencies.visual_qa_service,
        job_id=cost_run_id,
        budget_guard=budget_guard,
        required_beat_ids=selected_beat_id_set,
    )
    results = []
    scoped_decisions = [
        decision
        for decision in execution.decisions
        if decision.beat_id in selected_beat_id_set
    ]
    for index, decision in enumerate(scoped_decisions, start=1):
        stage_progress = round((index - 1) / max(total_beats, 1) * 100)
        overall = 30 + round((index - 1) / max(total_beats, 1) * 30)
        await emit(
            ProjectPipelineStage.GENERATING_VISUALS,
            overall,
            stage_progress,
            f"Генерация visual beat {index} из {total_beats}",
            index,
            total_beats,
            None,
        )
        try:
            result = await executor.execute_beat(
                project_id, execution.id, decision.beat_id
            )
        except Exception:
            await emit(
                ProjectPipelineStage.GENERATING_VISUALS,
                overall,
                stage_progress,
                f"Ошибка visual beat {index} из {total_beats}",
                index,
                total_beats,
                decision.beat_id,
            )
            raise
        results.append(result)
    await emit(
        ProjectPipelineStage.VISUAL_QA,
        62,
        100,
        "Все visuals приняты QA",
        total_beats,
        total_beats,
        None,
    )

    if scope.is_image_only:
        operation_counts = Counter(item.resolved_operation for item in results)
        costs = summarize_project_cost(session, project_id, job_id=cost_run_id)
        report = ProjectStylePreviewReport(
            pipeline_version=PIPELINE_VERSION,
            project_id=project_id,
            story_revision=hash_story_text(project.story_text),
            visual_beats=total_beats,
            semantic_visual_beats=semantic_total_beats,
            master_assets=len(masters),
            generated_or_edited_candidates=sum(
                operation_counts[name]
                for name in ("NEW_IMAGE", "REFERENCE_GENERATION", "EDIT_EXISTING")
            ),
            accepted_visual_assets=sum(item.is_accepted for item in results),
            preview_result_ids=tuple(
                item.id for item in results if item.is_accepted and item.output_path
            ),
            reused={
                "visual_plan": plan_reused,
                "master_assets": min(masters_before, len(masters)),
                "accepted_visual_assets": sum(
                    result.id in accepted_before
                    or (
                        result.resolved_operation == "REUSE"
                        and result.source_result_id in accepted_before
                    )
                    for result in results
                ),
            },
            estimated_cost_before_run=estimate.as_dict(),
            actual_run_cost=costs.run_cost,
            historical_project_asset_cost=costs.historical_project_cost,
            qa_retry_cost=costs.qa_retry_cost,
            cost_currency=costs.currency,
            unpriced_usage_records=costs.unpriced_records,
            cost_run_id=cost_run_id,
            generation_budget=budget_guard.snapshot().as_dict(),
            production_profile=profile.value,
            production_profile_version=execution.production_profile_version,
            generation_scope=scope.snapshot(),
        )
        await emit(
            ProjectPipelineStage.COMPLETED,
            100,
            100,
            f"Превью стиля готово: {len(report.preview_result_ids)} кадра",
            total_beats,
            total_beats,
            None,
        )
        return report

    await emit(
        ProjectPipelineStage.GENERATING_NARRATION,
        66,
        0,
        "Генерация narration",
        None,
        total_beats,
        None,
    )
    if narration is None:
        narration = await generate_project_narration(
            session, project_id, **narration_kwargs
        )
    await emit(
        ProjectPipelineStage.GENERATING_NARRATION,
        73,
        100,
        f"Narration: {narration.duration:.1f} sec",
        None,
        total_beats,
        None,
    )

    await emit(
        ProjectPipelineStage.ALIGNING_NARRATION,
        76,
        0,
        "Синхронизация beats",
        None,
        total_beats,
        None,
    )
    align_project_visual_beats(session, project_id, narration)
    await emit(
        ProjectPipelineStage.ALIGNING_NARRATION,
        80,
        100,
        "Narration alignment готов",
        None,
        total_beats,
        None,
    )

    await emit(
        ProjectPipelineStage.BUILDING_TIMELINE,
        82,
        0,
        "Построение timeline",
        None,
        total_beats,
        None,
    )
    timeline = build_project_timeline(
        session,
        project_id,
        execution.id,
        narration.id,
        generation_scope=scope,
        selected_beat_ids=selected_beat_ids,
    )
    timeline_quality = build_timeline_quality_report(timeline)
    await emit(
        ProjectPipelineStage.BUILDING_TIMELINE,
        86,
        100,
        (
            f"Timeline: {timeline_quality.effective_screen_states} screen states; "
            f"longest unchanged hold {timeline_quality.longest_unchanged_hold:.2f}s; "
            f"{timeline_quality.paid_generations_per_minute:.2f} paid visuals/min"
        ),
        None,
        total_beats,
        None,
    )

    render_config = ProjectRenderConfig(
        version=(
            "draft_render_config_v2"
            if profile is ProductionProfile.DRAFT
            else "project_render_config_v2"
        ),
        width=(
            project.draft_width if profile is ProductionProfile.DRAFT else project.width
        ),
        height=(
            project.draft_height
            if profile is ProductionProfile.DRAFT
            else project.height
        ),
        fps=project.fps,
        image_fit_mode=project.image_fit.upper(),
    )
    previous_render_ids = {
        item.id for item in list_project_video_renders(session, project_id)
    }
    await emit(
        ProjectPipelineStage.RENDERING_VIDEO,
        88,
        0,
        "Рендеринг финального MP4",
        None,
        total_beats,
        None,
    )
    render = render_project_video(
        session,
        project_id,
        timeline.id,
        config=render_config,
        projects_root=dependencies.projects_root,
        production_profile=profile,
    )
    await emit(
        ProjectPipelineStage.VALIDATING_VIDEO,
        97,
        0,
        "Проверка финального MP4",
        None,
        total_beats,
        None,
    )
    if render.output_path is None or render.duration is None:
        raise RuntimeError("Final video render is incomplete")

    all_results = list_beat_visual_results(
        session, project_id, execution_plan_id=execution.id
    )
    qa_summary = build_visual_qa_execution_summary(all_results)
    operation_counts = Counter(item.resolved_operation for item in results)
    costs = summarize_project_cost(session, project_id, job_id=cost_run_id)
    paid_operation_names = {"NEW_IMAGE", "REFERENCE_GENERATION", "EDIT_EXISTING"}
    projected_final = None
    if profile is ProductionProfile.DRAFT:
        resolve_project_visual_operations(
            session,
            project_id,
            provider_resolver=dependencies.image_provider_resolver,
            production_profile=ProductionProfile.FINAL,
        )
        projected_final = estimate_project_generation_cost(
            session,
            project_id,
            production_profile=ProductionProfile.FINAL.value,
        ).as_dict()
    report = ProjectPipelineReport(
        pipeline_version=PIPELINE_VERSION,
        project_id=project_id,
        story_revision=hash_story_text(project.story_text),
        visual_beats=total_beats,
        master_assets=len(masters),
        generated_or_edited_candidates=sum(
            operation_counts[name]
            for name in ("NEW_IMAGE", "REFERENCE_GENERATION", "EDIT_EXISTING")
        ),
        qa_retries=qa_summary.regenerated,
        accepted_visual_assets=sum(item.is_accepted for item in all_results),
        narration_duration=narration.duration,
        timeline_entries=len(timeline.entries),
        render_duration=render.duration,
        final_mp4=render.output_path,
        final_render_id=render.id,
        reused={
            "visual_plan": plan_reused,
            "master_assets": min(masters_before, len(masters)),
            "accepted_visual_assets": sum(
                result.id in accepted_before
                or (
                    result.resolved_operation == "REUSE"
                    and result.source_result_id in accepted_before
                )
                for result in results
            ),
            "render": render.id in previous_render_ids,
        },
        estimated_cost_before_run=estimate.as_dict(),
        actual_run_cost=costs.run_cost,
        historical_project_asset_cost=costs.historical_project_cost,
        qa_retry_cost=costs.qa_retry_cost,
        cost_by_stage=costs.cost_by_stage,
        cost_by_provider=costs.cost_by_provider,
        cost_by_model=costs.cost_by_model,
        cost_by_beat=costs.cost_by_beat,
        cost_currency=costs.currency,
        unpriced_usage_records=costs.unpriced_records,
        cost_run_id=cost_run_id,
        generation_budget=budget_guard.snapshot().as_dict(),
        production_profile=profile.value,
        production_profile_version=execution.production_profile_version,
        paid_visual_beats=sum(operation_counts[name] for name in paid_operation_names),
        free_visual_beats=sum(
            operation_counts[name] for name in ("REUSE", "TRANSFORM", "OVERLAY")
        ),
        visual_qa_calls=int(
            session.scalar(
                select(func.count(ProviderUsageRecord.id)).where(
                    ProviderUsageRecord.project_id == project_id,
                    ProviderUsageRecord.job_id == cost_run_id,
                    ProviderUsageRecord.pipeline_stage == "VISUAL_QA",
                    ProviderUsageRecord.status.in_(
                        [
                            UsageStatus.SUCCEEDED.value,
                            UsageStatus.FAILED.value,
                        ]
                    ),
                )
            )
            or 0
        ),
        projected_final_cost=projected_final,
        generation_scope=scope.snapshot(),
        semantic_visual_beats=semantic_total_beats,
        timeline_quality=asdict(timeline_quality),
    )
    await emit(
        ProjectPipelineStage.COMPLETED,
        100,
        100,
        "Видео готово",
        total_beats,
        total_beats,
        None,
    )
    return report


def _validate_project_preflight(
    project: Any,
    dependencies: ProjectPipelineDependencies,
    *,
    require_video_tools: bool = True,
) -> None:
    if not project.story_text.strip():
        raise ValueError("Добавьте готовый текст истории")
    if project.planning_provider not in {"dashscope", "kimi"}:
        raise ValueError("Выбранный planning provider не поддерживается")
    if project.visual_qa_enabled and dependencies.visual_qa_service is None:
        raise ValueError("Visual QA включён, но vision provider не настроен")
    get_image_style_contract(project.style_id)
    if require_video_tools and (
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None
    ):
        raise ValueError("FFmpeg и ffprobe должны быть установлены")
    root = Path(dependencies.projects_root)
    root.mkdir(parents=True, exist_ok=True)
    probe = root / ".pipeline-write-test"
    try:
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as exc:
        raise ValueError("Project storage is not writable") from exc
    if dependencies.preflight_validator is not None:
        dependencies.preflight_validator(project)


async def _ignore_progress(
    stage: ProjectPipelineStage,
    progress: int,
    stage_progress: int,
    message: str,
    current_beat: int | None,
    total_beats: int | None,
    failed_beat: str | None,
) -> None:
    del stage, progress, stage_progress, message, current_beat, total_beats, failed_beat
    await asyncio.sleep(0)
