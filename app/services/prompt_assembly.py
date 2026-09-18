"""Single-source image prompt compilation, override safety, and inspection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.clients.image_api import normalize_image_prompt_for_provider
from app.generators.image_prompt import (
    ImagePromptBuilder,
    sanitize_provider_visual_text,
)
from app.generators.style_reference import build_reference_role_instruction
from app.models.visual_plan import MasterScene, VisualBeat, VisualOperation, VisualPlan
from app.persistence import (
    BeatVisualResult,
    GeneratedVideoAsset,
    MasterSceneAsset,
    MasterSceneGenerationAttempt,
    Project,
    ProjectNarrationAlignment,
    ProjectVisualPlan,
    VideoGenerationAttempt,
    VisualPromptOverride,
)
from app.pipeline.visual_operation_engine import VisualOperationDecisionEngine
from app.pipeline.visual_qa import apply_visual_qa_correction
from app.provider_capabilities import ImageProviderCapabilities
from app.providers import (
    ImageReference,
    ImageReferenceRole,
    get_image_provider,
    get_image_provider_capabilities,
)
from app.repositories import (
    get_project,
    get_project_visual_plan_record,
    get_style_reference_asset,
)
from app.services.visual_asset_selection import (
    VisualAssetSource,
    select_visual_references,
)
from app.style_contracts import (
    DEFAULT_IMAGE_STYLE_ID,
    NO_VISIBLE_TEXT_INSTRUCTION,
    apply_image_style_contract,
    get_image_style_contract,
    prepare_image_prompt_for_provider,
)


class PromptTargetType(str, Enum):
    BEAT = "BEAT"
    MASTER_SCENE = "MASTER_SCENE"


class PromptOverrideState(str, Enum):
    AUTO = "AUTO"
    ACTIVE = "ACTIVE"
    STALE = "STALE"
    ORPHANED = "ORPHANED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    DISABLED = "DISABLED"


GENERATED_OPERATIONS = frozenset(
    {
        VisualOperation.NEW_IMAGE,
        VisualOperation.REFERENCE_GENERATION,
        VisualOperation.EDIT_EXISTING,
    }
)


@dataclass(frozen=True, slots=True)
class PromptAssembly:
    target_id: str
    target_type: str
    visual_plan_id: str
    visual_plan_revision: str
    semantic_requirement: dict[str, Any]
    semantic_fingerprint: str
    auto_scene_prompt: str | None
    manual_scene_override: str | None
    override_state: str
    override_id: str | None
    override_revision: int | None
    effective_scene_prompt: str | None
    operation: str
    operation_instructions: str | None
    references_used: list[dict[str, Any]]
    style_contract_version: str
    style_contract_snapshot: str
    qa_correction: str | None
    assembled_prompt_before_provider_transform: str | None
    provider_transformations: list[dict[str, object]]
    provider_compaction_report: dict[str, object] | None
    final_provider_prompt: str | None
    provider: str
    model: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PromptOverrideResolution:
    override: VisualPromptOverride | None
    state: PromptOverrideState


def visual_plan_revision(plan: VisualPlan) -> str:
    return _stable_hash(plan.model_dump(mode="json"))


def semantic_requirement_for_target(
    plan: VisualPlan,
    target_type: PromptTargetType | str,
    target_id: str,
) -> dict[str, Any]:
    target_type = PromptTargetType(target_type)
    if target_type is PromptTargetType.BEAT:
        beat = _beat(plan, target_id)
        return {
            "visual_purpose": beat.visual_purpose,
            "what_viewer_should_understand": beat.what_viewer_should_understand,
            "location_id": beat.location_id,
            "characters_visible": list(beat.characters_visible),
            "important_objects": list(beat.important_objects),
            "physical_state": beat.physical_state,
            "camera_framing": beat.camera_framing.value,
            "camera_view": beat.camera_view,
            "visual_focus": beat.visual_focus,
            "must_not_show": list(beat.must_not_show),
        }
    master = _master(plan, target_id)
    return {
        "location_id": master.location_id,
        "description": master.description,
        "environment_geometry": master.environment_geometry,
        "recurring_object_positions": master.recurring_object_positions,
        "basic_composition": master.basic_composition,
        "characters_visible": list(master.characters_visible),
        "important_objects": list(master.important_objects),
    }


def semantic_fingerprint(
    plan: VisualPlan,
    target_type: PromptTargetType | str,
    target_id: str,
) -> str:
    return _stable_hash(semantic_requirement_for_target(plan, target_type, target_id))


def resolve_prompt_override(
    session: Session,
    *,
    project_id: str,
    visual_plan_id: str,
    plan: VisualPlan,
    target_type: PromptTargetType | str,
    target_id: str,
) -> PromptOverrideResolution:
    target_type = PromptTargetType(target_type)
    revision = visual_plan_revision(plan)
    current_fingerprint = semantic_fingerprint(plan, target_type, target_id)
    exact = session.scalar(
        select(VisualPromptOverride).where(
            VisualPromptOverride.project_id == project_id,
            VisualPromptOverride.visual_plan_revision == revision,
            VisualPromptOverride.target_type == target_type.value,
            VisualPromptOverride.target_id == target_id,
        )
    )
    if exact is not None:
        if not exact.enabled:
            return PromptOverrideResolution(None, PromptOverrideState.AUTO)
        if exact.base_semantic_fingerprint != current_fingerprint:
            return PromptOverrideResolution(exact, PromptOverrideState.REVIEW_REQUIRED)
        return PromptOverrideResolution(exact, PromptOverrideState.ACTIVE)
    previous = session.scalar(
        select(VisualPromptOverride)
        .where(
            VisualPromptOverride.project_id == project_id,
            VisualPromptOverride.target_type == target_type.value,
            VisualPromptOverride.target_id == target_id,
        )
        .order_by(VisualPromptOverride.updated_at.desc())
        .limit(1)
    )
    if previous is None:
        return PromptOverrideResolution(None, PromptOverrideState.AUTO)
    if previous.base_semantic_fingerprint != current_fingerprint:
        return PromptOverrideResolution(previous, PromptOverrideState.REVIEW_REQUIRED)
    return PromptOverrideResolution(previous, PromptOverrideState.STALE)


def set_prompt_override(
    session: Session,
    project_id: str,
    *,
    target_type: PromptTargetType | str,
    target_id: str,
    scene_prompt_override: str,
) -> VisualPromptOverride:
    plan_record, plan = _current_plan(session, project_id)
    target_type = PromptTargetType(target_type)
    normalized = scene_prompt_override.strip()
    if not normalized:
        raise ValueError("scene_prompt_override must not be empty")
    fingerprint = semantic_fingerprint(plan, target_type, target_id)
    revision = visual_plan_revision(plan)
    record = session.scalar(
        select(VisualPromptOverride).where(
            VisualPromptOverride.project_id == project_id,
            VisualPromptOverride.visual_plan_revision == revision,
            VisualPromptOverride.target_type == target_type.value,
            VisualPromptOverride.target_id == target_id,
        )
    )
    if record is None:
        record = VisualPromptOverride(
            project_id=project_id,
            visual_plan_id=plan_record.id,
            visual_plan_revision=revision,
            target_type=target_type.value,
            target_id=target_id,
            scene_prompt_override=normalized,
            base_semantic_fingerprint=fingerprint,
            enabled=True,
            revision=1,
        )
        session.add(record)
    else:
        record.scene_prompt_override = normalized
        record.base_semantic_fingerprint = fingerprint
        record.enabled = True
        record.revision += 1
        record.updated_at = datetime.now(UTC)
    session.commit()
    session.refresh(record)
    return record


def clear_prompt_override(
    session: Session,
    project_id: str,
    *,
    target_type: PromptTargetType | str,
    target_id: str,
) -> VisualPromptOverride | None:
    plan_record, plan = _current_plan(session, project_id)
    resolution = resolve_prompt_override(
        session,
        project_id=project_id,
        visual_plan_id=plan_record.id,
        plan=plan,
        target_type=target_type,
        target_id=target_id,
    )
    record = resolution.override
    if record is None:
        return None
    record.enabled = False
    record.revision += 1
    record.updated_at = datetime.now(UTC)
    session.commit()
    session.refresh(record)
    return record


def compile_prompt_assembly(
    session: Session,
    *,
    project: Project,
    plan_record: ProjectVisualPlan | None,
    plan: VisualPlan,
    target_type: PromptTargetType | str,
    target_id: str,
    operation: VisualOperation | str,
    auto_scene_prompt: str | None,
    references: tuple[ImageReference, ...] = (),
    style_id: str = DEFAULT_IMAGE_STYLE_ID,
    provider: str | None = None,
    model: str | None = None,
    qa_correction: str | None = None,
) -> PromptAssembly:
    target_type = PromptTargetType(target_type)
    operation = VisualOperation(operation)
    requirement = semantic_requirement_for_target(plan, target_type, target_id)
    fingerprint = _stable_hash(requirement)
    resolution = (
        resolve_prompt_override(
            session,
            project_id=project.id,
            visual_plan_id=plan_record.id,
            plan=plan,
            target_type=target_type,
            target_id=target_id,
        )
        if plan_record is not None
        else PromptOverrideResolution(None, PromptOverrideState.AUTO)
    )
    plan_id = plan_record.id if plan_record is not None else "unpersisted"
    active_override = (
        resolution.override if resolution.state is PromptOverrideState.ACTIVE else None
    )
    operation_instructions = _operation_instructions(operation)
    reference_snapshot = [_reference_snapshot(item) for item in references]
    provider_id = (provider or project.image_provider).strip().lower()
    provider_model = model if model is not None else project.image_model
    style = get_image_style_contract(style_id)

    if operation not in GENERATED_OPERATIONS:
        return PromptAssembly(
            target_id=target_id,
            target_type=target_type.value,
            visual_plan_id=plan_id,
            visual_plan_revision=visual_plan_revision(plan),
            semantic_requirement=requirement,
            semantic_fingerprint=fingerprint,
            auto_scene_prompt=None,
            manual_scene_override=None,
            override_state=resolution.state.value,
            override_id=resolution.override.id if resolution.override else None,
            override_revision=(
                resolution.override.revision if resolution.override else None
            ),
            effective_scene_prompt=None,
            operation=operation.value,
            operation_instructions=operation_instructions,
            references_used=reference_snapshot,
            style_contract_version=style.style_id,
            style_contract_snapshot=style.render_for_image_provider(),
            qa_correction=None,
            assembled_prompt_before_provider_transform=None,
            provider_transformations=[],
            provider_compaction_report=None,
            final_provider_prompt=None,
            provider=provider_id,
            model=provider_model,
        )

    if auto_scene_prompt is None:
        raise ValueError("generated image operation requires auto_scene_prompt")
    auto_dynamic_prompt = _without_style_contract(auto_scene_prompt, style_id)
    manual = active_override.scene_prompt_override if active_override else None
    effective = manual or auto_dynamic_prompt
    if manual is None:
        assembled = auto_scene_prompt
    else:
        assembled = _assemble_manual_dynamic_prompt(
            requirement,
            manual,
            operation_instructions,
            references,
            project.global_image_style_prompt,
            style_id,
        )
    if qa_correction is not None:
        assembled = apply_visual_qa_correction(assembled, qa_correction, style_id)
    provider_ready = prepare_image_prompt_for_provider(assembled, style_id)
    compaction_report: dict[str, object] | None = None
    if provider_id == "zimage":
        from app.services.zimage_prompt_compiler import compile_zimage_semantic_prompt

        target = (
            _beat(plan, target_id)
            if target_type is PromptTargetType.BEAT
            else _master(plan, target_id)
        )
        compilation = compile_zimage_semantic_prompt(
            plan,
            target=target,
            operation=operation,
            manual_scene_override=manual,
            qa_correction=qa_correction,
            style_id=style_id,
        )
        final_prompt = compilation.prompt
        compaction_report = compilation.report
        transformations = [
            {
                "type": "ZIMAGE_LIMIT_NORMALIZATION",
                "strategy": "STRUCTURED_SEMANTIC_COMPILATION",
                "before_length": len(provider_ready),
                "after_length": len(final_prompt),
                **compilation.report,
            }
        ]
    else:
        final_prompt, transformations = normalize_image_prompt_for_provider(
            provider_ready, provider_id
        )
    if provider_ready != assembled:
        transformations.insert(
            0,
            {
                "type": "STYLE_CONTRACT_PROVIDER_RENDERING",
                "before_length": len(assembled),
                "after_length": len(provider_ready),
            },
        )
    return PromptAssembly(
        target_id=target_id,
        target_type=target_type.value,
        visual_plan_id=plan_id,
        visual_plan_revision=visual_plan_revision(plan),
        semantic_requirement=requirement,
        semantic_fingerprint=fingerprint,
        auto_scene_prompt=auto_dynamic_prompt,
        manual_scene_override=manual,
        override_state=resolution.state.value,
        override_id=active_override.id if active_override else None,
        override_revision=active_override.revision if active_override else None,
        effective_scene_prompt=effective,
        operation=operation.value,
        operation_instructions=operation_instructions,
        references_used=reference_snapshot,
        style_contract_version=style.style_id,
        style_contract_snapshot=style.render_for_image_provider(),
        qa_correction=qa_correction,
        assembled_prompt_before_provider_transform=assembled,
        provider_transformations=transformations,
        provider_compaction_report=compaction_report,
        final_provider_prompt=final_prompt,
        provider=provider_id,
        model=provider_model,
    )


def preview_generation_request(
    session: Session,
    project_id: str,
    *,
    target_type: PromptTargetType | str,
    target_id: str,
    provider: str | None = None,
    model: str | None = None,
    qa_correction: str | None = None,
    operation: VisualOperation | str | None = None,
    references: tuple[ImageReference, ...] | None = None,
) -> PromptAssembly:
    project = get_project(session, project_id)
    plan_record, plan = _current_plan(session, project_id)
    if project is None:
        raise ValueError("Project not found")
    target_type = PromptTargetType(target_type)
    if target_type is PromptTargetType.BEAT:
        beat = _beat(plan, target_id)
        capabilities = _preview_capabilities(project, provider, model)
        resolved = (
            VisualOperation(operation)
            if operation is not None
            else _preview_resolved_operation(plan, beat, capabilities)
        )
        if references is None:
            references = _preview_beat_references(
                session,
                project,
                beat,
                resolved,
                capabilities=capabilities,
            )
        auto = build_auto_beat_scene_prompt(
            project,
            plan,
            beat,
            resolved,
            references,
            style_id=project.style_id,
        )
    else:
        master = _master(plan, target_id)
        resolved = VisualOperation(operation or VisualOperation.NEW_IMAGE)
        if references is None:
            references = _preview_master_references(
                session,
                project,
                provider=provider,
                model=model,
            )
        if operation is None and references:
            resolved = VisualOperation.REFERENCE_GENERATION
        auto = build_auto_master_scene_prompt(
            master,
            project.global_image_style_prompt,
            references,
            style_id=project.style_id,
        )
    return compile_prompt_assembly(
        session,
        project=project,
        plan_record=plan_record,
        plan=plan,
        target_type=target_type,
        target_id=target_id,
        operation=resolved,
        auto_scene_prompt=auto,
        references=references,
        style_id=project.style_id,
        provider=provider,
        model=model,
        qa_correction=qa_correction,
    )


def get_prompt_detail(
    session: Session,
    project_id: str,
    *,
    target_type: PromptTargetType | str,
    target_id: str,
) -> dict[str, Any]:
    target_type = PromptTargetType(target_type)
    detail = preview_generation_request(
        session,
        project_id,
        target_type=target_type,
        target_id=target_id,
    ).as_dict()
    plan_record, plan = _current_plan(session, project_id)
    resolution = resolve_prompt_override(
        session,
        project_id=project_id,
        visual_plan_id=plan_record.id,
        plan=plan,
        target_type=target_type,
        target_id=target_id,
    )
    detail["stored_override"] = (
        {
            "id": resolution.override.id,
            "scene_prompt_override": resolution.override.scene_prompt_override,
            "state": resolution.state.value,
            "revision": resolution.override.revision,
        }
        if resolution.override is not None
        else None
    )
    if target_type is PromptTargetType.BEAT:
        beat = _beat(plan, target_id)
        records = list(
            session.scalars(
                select(BeatVisualResult)
                .where(
                    BeatVisualResult.project_id == project_id,
                    BeatVisualResult.beat_id == target_id,
                )
                .order_by(BeatVisualResult.created_at, BeatVisualResult.attempt)
            )
        )
        detail["attempts"] = [
            {
                "generation_attempt": record.attempt,
                "prompt_assembly": record.prompt_assembly_snapshot,
                "generated_asset": record.output_path,
                "generation_status": record.generation_status,
                "qa_result": record.qa_result,
                "qa_status": record.qa_status,
                "accepted": record.is_accepted,
            }
            for record in records
        ]
        latest = records[-1] if records else None
        detail["target_metadata"] = {
            "narration": beat.narration_segment,
            "summary": beat.what_viewer_should_understand,
            "source_visual_id": beat.source_visual_id,
            "source_asset": latest.source_path if latest is not None else None,
            "master_scene_id": beat.master_scene_id,
            "time_range": _beat_time_range(session, plan_record, target_id),
            "operation_detail": _operation_detail(beat),
        }
    else:
        master = _master(plan, target_id)
        attempts = list(
            session.scalars(
                select(MasterSceneGenerationAttempt)
                .where(
                    MasterSceneGenerationAttempt.project_id == project_id,
                    MasterSceneGenerationAttempt.master_scene_id == target_id,
                )
                .order_by(
                    MasterSceneGenerationAttempt.created_at,
                    MasterSceneGenerationAttempt.attempt,
                )
            )
        )
        asset = session.scalar(
            select(MasterSceneAsset).where(
                MasterSceneAsset.project_id == project_id,
                MasterSceneAsset.master_scene_id == target_id,
            )
        )
        detail["attempts"] = [
            {
                "generation_attempt": record.attempt,
                "prompt_assembly": record.prompt_assembly_snapshot,
                "generated_asset": record.output_path,
                "generation_status": record.generation_status,
                "qa_result": record.qa_result,
                "accepted": bool(
                    asset
                    and asset.prompt_assembly_snapshot
                    == record.prompt_assembly_snapshot
                ),
            }
            for record in attempts
        ]
        detail["generated_asset"] = asset.file_path if asset is not None else None
        detail["target_metadata"] = {
            "narration": None,
            "summary": master.description,
            "source_visual_id": None,
            "source_asset": asset.file_path if asset is not None else None,
            "master_scene_id": master.id,
            "time_range": None,
            "operation_detail": None,
        }
    return detail


def get_prompt_sheet(session: Session, project_id: str) -> list[dict[str, Any]]:
    plan_record, plan = _current_plan(session, project_id)
    project = get_project(session, project_id)
    if project is None:
        raise ValueError("Project not found")
    capabilities = _preview_capabilities(project, None, None)
    rows: list[dict[str, Any]] = []
    current_keys: set[tuple[str, str]] = set()
    latest_alignment = session.scalar(
        select(ProjectNarrationAlignment)
        .where(
            ProjectNarrationAlignment.project_id == project_id,
            ProjectNarrationAlignment.visual_plan_id == plan_record.id,
            ProjectNarrationAlignment.visual_plan_revision
            == visual_plan_revision(plan),
        )
        .order_by(ProjectNarrationAlignment.created_at.desc())
        .limit(1)
    )
    timings = (
        {
            timing.beat_id: {
                "start": timing.audio_start,
                "end": timing.audio_end,
            }
            for timing in latest_alignment.beat_timings
        }
        if latest_alignment is not None
        else {}
    )
    for target_type, targets in (
        (PromptTargetType.MASTER_SCENE, plan.possible_master_scenes),
        (PromptTargetType.BEAT, plan.visual_beats),
    ):
        for target in targets:
            current_keys.add((target_type.value, target.id))
            resolution = resolve_prompt_override(
                session,
                project_id=project_id,
                visual_plan_id=plan_record.id,
                plan=plan,
                target_type=target_type,
                target_id=target.id,
            )
            operation = (
                (
                    VisualOperation.REFERENCE_GENERATION.value
                    if _preview_master_references(
                        session,
                        project,
                        provider=None,
                        model=None,
                    )
                    else VisualOperation.NEW_IMAGE.value
                )
                if target_type is PromptTargetType.MASTER_SCENE
                else _preview_resolved_operation(plan, target, capabilities).value
            )
            latest_result = (
                session.scalar(
                    select(BeatVisualResult)
                    .where(
                        BeatVisualResult.project_id == project_id,
                        BeatVisualResult.beat_id == target.id,
                    )
                    .order_by(BeatVisualResult.created_at.desc())
                    .limit(1)
                )
                if target_type is PromptTargetType.BEAT
                else None
            )
            accepted_result = (
                session.scalar(
                    select(BeatVisualResult)
                    .where(
                        BeatVisualResult.project_id == project_id,
                        BeatVisualResult.beat_id == target.id,
                        BeatVisualResult.is_accepted.is_(True),
                    )
                    .order_by(BeatVisualResult.created_at.desc())
                    .limit(1)
                )
                if target_type is PromptTargetType.BEAT
                else None
            )
            master_asset = (
                session.scalar(
                    select(MasterSceneAsset).where(
                        MasterSceneAsset.project_id == project_id,
                        MasterSceneAsset.master_scene_id == target.id,
                    )
                )
                if target_type is PromptTargetType.MASTER_SCENE
                else None
            )
            latest_master_attempt = (
                session.scalar(
                    select(MasterSceneGenerationAttempt)
                    .where(
                        MasterSceneGenerationAttempt.project_id == project_id,
                        MasterSceneGenerationAttempt.master_scene_id == target.id,
                    )
                    .order_by(MasterSceneGenerationAttempt.created_at.desc())
                    .limit(1)
                )
                if target_type is PromptTargetType.MASTER_SCENE
                else None
            )
            latest_video_asset = (
                session.scalar(
                    select(GeneratedVideoAsset)
                    .join(VideoGenerationAttempt)
                    .where(
                        GeneratedVideoAsset.project_id == project_id,
                        GeneratedVideoAsset.beat_id == target.id,
                        VideoGenerationAttempt.accepted_image_result_id
                        == (
                            accepted_result.id
                            if accepted_result is not None
                            else "__missing_accepted_result__"
                        ),
                    )
                    .order_by(GeneratedVideoAsset.created_at.desc())
                    .limit(1)
                )
                if target_type is PromptTargetType.BEAT
                else None
            )
            latest_video_attempt = (
                session.scalar(
                    select(VideoGenerationAttempt)
                    .where(
                        VideoGenerationAttempt.project_id == project_id,
                        VideoGenerationAttempt.beat_id == target.id,
                    )
                    .order_by(VideoGenerationAttempt.created_at.desc())
                    .limit(1)
                )
                if target_type is PromptTargetType.BEAT
                else None
            )
            no_provider_call = operation in {
                VisualOperation.REUSE.value,
                VisualOperation.TRANSFORM.value,
                VisualOperation.OVERLAY.value,
            }
            rows.append(
                {
                    "target_type": target_type.value,
                    "target_id": target.id,
                    "semantic_fingerprint": semantic_fingerprint(
                        plan, target_type, target.id
                    ),
                    "override_state": resolution.state.value,
                    "prompt_mode": (
                        "NO_PROVIDER_CALL"
                        if no_provider_call
                        else "OVERRIDE"
                        if resolution.state is PromptOverrideState.ACTIVE
                        else resolution.state.value
                    ),
                    "override_id": resolution.override.id
                    if resolution.override
                    else None,
                    "operation": operation,
                    "summary": (
                        target.description
                        if target_type is PromptTargetType.MASTER_SCENE
                        else target.what_viewer_should_understand
                    ),
                    "narration": (
                        None
                        if target_type is PromptTargetType.MASTER_SCENE
                        else target.narration_segment
                    ),
                    "source_visual_id": (
                        None
                        if target_type is PromptTargetType.MASTER_SCENE
                        else target.source_visual_id
                    ),
                    "master_scene_id": (
                        target.id
                        if target_type is PromptTargetType.MASTER_SCENE
                        else target.master_scene_id
                    ),
                    "time_range": (
                        None
                        if target_type is PromptTargetType.MASTER_SCENE
                        else timings.get(target.id)
                    ),
                    "selectable": target_type is PromptTargetType.BEAT,
                    "accepted_asset_path": (
                        master_asset.file_path
                        if master_asset is not None
                        else (
                            accepted_result.output_path
                            if accepted_result is not None
                            else None
                        )
                    ),
                    "source_asset": (
                        latest_result.source_path if latest_result else None
                    ),
                    "provider": (
                        latest_result.provider
                        if latest_result is not None
                        else master_asset.provider
                        if master_asset is not None
                        else project.image_provider
                    ),
                    "model": (
                        latest_result.model
                        if latest_result is not None
                        else master_asset.model
                        if master_asset is not None
                        else project.image_model
                    ),
                    "operation_detail": (
                        None
                        if target_type is PromptTargetType.MASTER_SCENE
                        else _operation_detail(target)
                    ),
                    "generation_status": (
                        latest_result.generation_status
                        if latest_result
                        else latest_master_attempt.generation_status
                        if latest_master_attempt
                        else "SUCCEEDED"
                        if master_asset is not None
                        else None
                    ),
                    "qa_status": (
                        latest_result.qa_status
                        if latest_result
                        else latest_master_attempt.qa_result
                        if latest_master_attempt
                        else None
                    ),
                    "video_asset_path": (
                        latest_video_asset.file_path if latest_video_asset else None
                    ),
                    "video_asset_id": (
                        latest_video_asset.id if latest_video_asset else None
                    ),
                    "video_attempt": (
                        {
                            "id": latest_video_attempt.id,
                            "provider": latest_video_attempt.provider,
                            "model": latest_video_attempt.model,
                            "operation": latest_video_attempt.operation,
                            "remote_task_id": latest_video_attempt.remote_task_id,
                            "status": latest_video_attempt.status,
                            "estimated_cost": (
                                float(latest_video_attempt.estimated_cost)
                                if latest_video_attempt.estimated_cost is not None
                                else None
                            ),
                            "actual_cost": (
                                float(latest_video_attempt.actual_cost)
                                if latest_video_attempt.actual_cost is not None
                                else None
                            ),
                            "cost_certainty": latest_video_attempt.cost_certainty,
                        }
                        if latest_video_attempt
                        else None
                    ),
                }
            )
    overrides = list(
        session.scalars(
            select(VisualPromptOverride).where(
                VisualPromptOverride.project_id == project_id,
                VisualPromptOverride.enabled.is_(True),
            )
        )
    )
    for item in overrides:
        key = (item.target_type, item.target_id)
        if key not in current_keys:
            rows.append(
                {
                    "target_type": item.target_type,
                    "target_id": item.target_id,
                    "semantic_fingerprint": None,
                    "override_state": PromptOverrideState.ORPHANED.value,
                    "override_id": item.id,
                }
            )
    return rows


def _beat_time_range(
    session: Session,
    plan_record: ProjectVisualPlan,
    beat_id: str,
) -> dict[str, float] | None:
    """Return timing only when it belongs to this exact VisualPlan revision."""
    alignment = session.scalar(
        select(ProjectNarrationAlignment)
        .where(
            ProjectNarrationAlignment.project_id == plan_record.project_id,
            ProjectNarrationAlignment.visual_plan_id == plan_record.id,
            ProjectNarrationAlignment.visual_plan_revision
            == visual_plan_revision(VisualPlan.model_validate(plan_record.plan_json)),
        )
        .order_by(ProjectNarrationAlignment.created_at.desc())
        .limit(1)
    )
    if alignment is None:
        return None
    timing = next(
        (item for item in alignment.beat_timings if item.beat_id == beat_id),
        None,
    )
    if timing is None:
        return None
    return {"start": timing.audio_start, "end": timing.audio_end}


def _preview_master_references(
    session: Session,
    project: Project,
    *,
    provider: str | None,
    model: str | None,
) -> tuple[ImageReference, ...]:
    capabilities = _preview_capabilities(project, provider, model)
    if not capabilities.reference_generation:
        return ()
    style_reference = get_style_reference_asset(session, project.id, project.style_id)
    if style_reference is None:
        return ()
    from app.generators.style_reference import to_style_image_reference

    return (to_style_image_reference(style_reference),)


def _preview_beat_references(
    session: Session,
    project: Project,
    beat: VisualBeat,
    operation: VisualOperation,
    *,
    capabilities: ImageProviderCapabilities,
) -> tuple[ImageReference, ...]:
    if not capabilities.reference_generation:
        return ()
    style_reference = get_style_reference_asset(session, project.id, project.style_id)
    master_asset = session.scalar(
        select(MasterSceneAsset).where(
            MasterSceneAsset.project_id == project.id,
            MasterSceneAsset.master_scene_id == beat.master_scene_id,
        )
    )
    source_result = None
    if beat.source_visual_id is not None:
        source_result = session.scalar(
            select(BeatVisualResult)
            .where(
                BeatVisualResult.project_id == project.id,
                BeatVisualResult.beat_id == beat.source_visual_id,
                BeatVisualResult.is_accepted.is_(True),
            )
            .order_by(BeatVisualResult.created_at.desc())
            .limit(1)
        )
    source = (
        VisualAssetSource(
            asset_id=source_result.id,
            result_id=source_result.id,
            path=source_result.output_path,
            sha256=source_result.file_sha256,
            visual_id=source_result.beat_id,
            reason="explicit source beat selected for prompt preview",
        )
        if source_result is not None
        and source_result.output_path is not None
        and source_result.file_sha256 is not None
        else None
    )
    selected = [
        item.reference
        for item in select_visual_references(
            beat,
            operation,
            capabilities,
            style_reference=style_reference,
            master_asset=master_asset,
            source=source,
        )
    ]
    if master_asset is None and beat.master_scene_id is not None:
        selected.append(_planned_reference(f"master:{beat.master_scene_id}"))
    planned_source = None
    if (
        source is None
        and beat.source_visual_id is not None
        and operation is not VisualOperation.NEW_IMAGE
    ):
        planned_source = _planned_reference(f"beat:{beat.source_visual_id}")
        selected.append(planned_source)
    unique = list({item.sha256: item for item in selected}.values())
    limited = unique[: capabilities.max_reference_images]
    if (
        operation is VisualOperation.EDIT_EXISTING
        and planned_source is not None
        and all(item.sha256 != planned_source.sha256 for item in limited)
        and limited
    ):
        limited[-1] = planned_source
    return tuple(limited)


def _preview_capabilities(
    project: Project,
    provider: str | None,
    model: str | None,
) -> ImageProviderCapabilities:
    provider_id = provider or project.image_provider
    provider_model = model if model is not None else project.image_model
    config = {"model": provider_model} if provider_model else None
    return get_image_provider_capabilities(get_image_provider(provider_id, config))


def _planned_reference(reference_id: str) -> ImageReference:
    digest = hashlib.sha256(f"planned-reference:{reference_id}".encode()).hexdigest()
    return ImageReference(
        reference_id=reference_id,
        file_path=f"planned://{reference_id}",
        sha256=digest,
        role=ImageReferenceRole.CONTENT_CONTINUITY,
    )


def _preview_resolved_operation(
    plan: VisualPlan,
    beat: VisualBeat,
    capabilities: ImageProviderCapabilities,
) -> VisualOperation:
    beat_index = next(
        index
        for index, candidate in enumerate(plan.visual_beats)
        if candidate.id == beat.id
    )
    available_visuals = {
        master.id: f"planned://master/{master.id}"
        for master in plan.possible_master_scenes
    }
    available_visuals.update(
        {
            candidate.id: f"planned://beat/{candidate.id}"
            for candidate in plan.visual_beats[:beat_index]
        }
    )
    return (
        VisualOperationDecisionEngine()
        .decide(
            plan,
            beat_index,
            capabilities=capabilities,
            available_visuals=available_visuals,
        )
        .operation
    )


def _operation_detail(beat: VisualBeat) -> str | None:
    if beat.preferred_visual_operation is VisualOperation.OVERLAY:
        return beat.overlay_description
    if beat.preferred_visual_operation is VisualOperation.TRANSFORM:
        return f"{beat.camera_movement.value}: {beat.camera_view}"
    if beat.preferred_visual_operation is VisualOperation.REUSE:
        return f"Reuse source {beat.source_visual_id or 'automatically selected asset'}"
    return None


def _assemble_manual_dynamic_prompt(
    requirement: dict[str, Any],
    manual: str,
    operation_instructions: str | None,
    references: tuple[ImageReference, ...],
    project_style_prompt: str | None,
    style_id: str,
) -> str:
    parts = [
        NO_VISIBLE_TEXT_INSTRUCTION,
        _semantic_guard_instruction(requirement),
        f"Create an illustration showing {sanitize_provider_visual_text(manual)}.",
    ]
    if operation_instructions:
        parts.append(operation_instructions)
    if references:
        parts.append(build_reference_role_instruction(references))
    if project_style_prompt and project_style_prompt.strip():
        parts.append(
            "Also follow this project drawing direction, "
            f"{sanitize_provider_visual_text(project_style_prompt)}."
        )
    return apply_image_style_contract(" ".join(parts), style_id)


def _semantic_guard_instruction(requirement: dict[str, Any]) -> str:
    """Render semantic authority as natural direction, never planner field labels."""
    parts: list[str] = []
    understanding = requirement.get("what_viewer_should_understand")
    purpose = requirement.get("visual_purpose")
    if understanding:
        parts.append(f"The viewer must immediately understand {understanding}")
    if purpose:
        parts.append(f"the illustration must visually {purpose}")
    location = requirement.get("location_id")
    if location:
        parts.append(f"keep the scene in the established {location} location")
    characters = requirement.get("characters_visible") or ()
    if characters:
        parts.append(f"show {', '.join(str(item) for item in characters)}")
    objects = requirement.get("important_objects") or ()
    if objects:
        parts.append(f"include {', '.join(str(item) for item in objects)}")
    physical_state = requirement.get("physical_state")
    if physical_state:
        parts.append(f"depict {physical_state}")
    framing = requirement.get("camera_framing")
    camera_view = requirement.get("camera_view")
    if framing or camera_view:
        parts.append(
            f"frame it as {' '.join(str(item) for item in (framing, camera_view) if item)}"
        )
    focus = requirement.get("visual_focus")
    if focus:
        parts.append(f"guide attention to {focus}")
    excluded = requirement.get("must_not_show") or ()
    if excluded:
        parts.append(f"exclude {', '.join(str(item) for item in excluded)}")
    description = requirement.get("description")
    geometry = requirement.get("environment_geometry")
    positions = requirement.get("recurring_object_positions")
    composition = requirement.get("basic_composition")
    if description:
        parts.append(f"establish {description}")
    if geometry:
        parts.append(f"preserve this environment geometry, {geometry}")
    if positions:
        parts.append(f"keep recurring objects positioned as {positions}")
    if composition:
        parts.append(f"preserve this composition, {composition}")
    return "Preserve these story facts. " + ". ".join(parts) + "."


def _without_style_contract(prompt: str, style_id: str) -> str:
    rendered = get_image_style_contract(style_id).render_for_image_provider()
    normalized = prompt.strip()
    if normalized.endswith(rendered):
        return normalized[: -len(rendered)].rstrip()
    return normalized


def _operation_instructions(operation: VisualOperation) -> str | None:
    return {
        VisualOperation.NEW_IMAGE: "Create a new image for the desired final visual.",
        VisualOperation.REFERENCE_GENERATION: (
            "Create a new image while using the attached references only for their assigned roles."
        ),
        VisualOperation.EDIT_EXISTING: (
            "Edit the source image while preserving everything not required to change."
        ),
        VisualOperation.REUSE: "Reuse the selected accepted source asset unchanged.",
        VisualOperation.TRANSFORM: "Apply the persisted local transform; do not call an image provider.",
        VisualOperation.OVERLAY: "Apply the persisted deterministic overlay; do not call an image provider.",
    }[operation]


def _reference_snapshot(reference: ImageReference) -> dict[str, Any]:
    return {
        "reference_id": reference.reference_id,
        "file_path": reference.file_path,
        "sha256": reference.sha256,
        "role": reference.role.value,
    }


def build_auto_beat_scene_prompt(
    project: Project,
    plan: VisualPlan,
    beat: VisualBeat,
    operation: VisualOperation,
    references: tuple[ImageReference, ...],
    *,
    style_id: str,
) -> str | None:
    if operation not in GENERATED_OPERATIONS:
        return None
    builder = ImagePromptBuilder()
    if operation is VisualOperation.EDIT_EXISTING:
        return builder.build_edit(
            plan,
            beat,
            references=references,
            style_id=style_id,
            project_style_prompt=project.global_image_style_prompt,
        )
    return builder.build(
        plan,
        beat,
        operation,
        references=references,
        style_id=style_id,
        project_style_prompt=project.global_image_style_prompt,
    )


def build_auto_master_scene_prompt(
    master: MasterScene,
    global_style_prompt: str | None,
    references: tuple[ImageReference, ...],
    *,
    style_id: str,
) -> str:
    # Imported lazily to keep master generation and inspection on one builder.
    from app.generators.master_scene import build_master_scene_generation_prompt

    prompt = build_master_scene_generation_prompt(master, global_style_prompt)
    if references:
        prompt = f"{prompt}\n\n{build_reference_role_instruction(references)}"
    return apply_image_style_contract(prompt, style_id)


def _current_plan(
    session: Session, project_id: str
) -> tuple[ProjectVisualPlan, VisualPlan]:
    record = get_project_visual_plan_record(session, project_id)
    if record is None:
        raise ValueError("Project visual plan not found")
    return record, VisualPlan.model_validate(record.plan_json)


def _beat(plan: VisualPlan, target_id: str) -> VisualBeat:
    match = next((item for item in plan.visual_beats if item.id == target_id), None)
    if match is None:
        raise ValueError(f"Unknown visual beat: {target_id}")
    return match


def _master(plan: VisualPlan, target_id: str) -> MasterScene:
    match = next(
        (item for item in plan.possible_master_scenes if item.id == target_id), None
    )
    if match is None:
        raise ValueError(f"Unknown master scene: {target_id}")
    return match


def _stable_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()
