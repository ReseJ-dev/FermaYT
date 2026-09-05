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

from sqlalchemy.orm import Session

from app.costs import (
    PricingUnit,
    UsageStatus,
    estimate_project_generation_cost,
    record_provider_usage,
    summarize_project_cost,
    usage_revision,
)
from app.generators.master_scene import generate_required_master_scenes
from app.models.render import ProjectRenderConfig
from app.pipeline.visual_qa import VisualQAService
from app.providers import (
    ImageProvider,
    TTSProvider,
    get_image_provider_capabilities,
)
from app.repositories import (
    get_project,
    get_style_reference_asset,
    list_beat_visual_results,
    list_master_scene_assets,
    list_project_video_renders,
)
from app.services.narration import generate_project_narration
from app.services.narration_alignment import align_project_visual_beats
from app.services.project_render import render_project_video
from app.services.timeline import build_project_timeline
from app.services.visual_asset_execution import (
    VisualBeatAssetExecutor,
    build_visual_qa_execution_summary,
)
from app.services.visual_operations import resolve_project_visual_operations
from app.services.visual_planning import (
    create_project_visual_plan,
    hash_story_text,
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

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


async def run_project_video_pipeline(
    session: Session,
    project_id: str,
    dependencies: ProjectPipelineDependencies,
    *,
    progress: PipelineProgress | None = None,
    job_id: str | None = None,
) -> ProjectPipelineReport:
    """Run every required current stage, reusing valid persisted revisions."""
    cost_run_id = job_id or f"direct-{uuid4()}"
    emit = progress or _ignore_progress
    await emit(
        ProjectPipelineStage.VALIDATING, 2, 0, "Проверка проекта", None, None, None
    )
    project = get_project(session, project_id)
    if project is None:
        raise ValueError("Project not found")
    _validate_project_preflight(project, dependencies)
    await emit(
        ProjectPipelineStage.VALIDATING, 5, 100, "Проект готов", None, None, None
    )

    existing_plan = load_project_visual_plan_state(session, project_id)
    plan_reused = existing_plan is not None and existing_plan.is_current
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
        record_provider_usage(
            session, project_id=project_id, job_id=cost_run_id,
            pipeline_stage="PLANNING", provider=project.planning_provider,
            model=project.planning_model, operation="PLANNING",
            request_revision=usage_revision(hash_story_text(project.story_text), "planning"),
            unit_type=PricingUnit.PER_REQUEST, input_units=1,
            status=UsageStatus.CACHED,
        )
    else:
        plan = await create_project_visual_plan(
            session, project_id, dependencies.planning_client, job_id=cost_run_id
        )
    total_beats = len(plan.visual_beats)
    await emit(
        ProjectPipelineStage.PLANNING,
        15,
        100,
        f"Визуальный план: {total_beats} beats",
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
    )
    provider = dependencies.image_provider_resolver(
        project.image_provider,
        {"model": project.image_model} if project.image_model else None,
    )
    capabilities = get_image_provider_capabilities(provider)
    estimate = estimate_project_generation_cost(session, project_id)
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
    )
    execution = resolve_project_visual_operations(
        session,
        project_id,
        provider_resolver=dependencies.image_provider_resolver,
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
    )
    results = []
    for index, decision in enumerate(execution.decisions, start=1):
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

    await emit(
        ProjectPipelineStage.GENERATING_NARRATION,
        66,
        0,
        "Генерация narration",
        None,
        total_beats,
        None,
    )
    narration_kwargs: dict[str, Any] = {
        "provider_resolver": dependencies.tts_provider_resolver,
        "downloader": dependencies.downloader,
        "projects_root": dependencies.projects_root,
        "job_id": cost_run_id,
    }
    if dependencies.duration_probe is not None:
        narration_kwargs["duration_probe"] = dependencies.duration_probe
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
    timeline = build_project_timeline(session, project_id, execution.id, narration.id)
    await emit(
        ProjectPipelineStage.BUILDING_TIMELINE,
        86,
        100,
        f"Timeline entries: {len(timeline.entries)}",
        None,
        total_beats,
        None,
    )

    render_config = ProjectRenderConfig(
        width=project.width,
        height=project.height,
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
                result.id in accepted_before for result in results
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
    project: Any, dependencies: ProjectPipelineDependencies
) -> None:
    if not project.story_text.strip():
        raise ValueError("Добавьте готовый текст истории")
    if project.planning_provider != "dashscope":
        raise ValueError("Выбранный planning provider не поддерживается")
    if project.visual_qa_enabled and dependencies.visual_qa_service is None:
        raise ValueError("Visual QA включён, но vision provider не настроен")
    get_image_style_contract(project.style_id)
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
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
