"""Execute resolved visual beats into a persisted continuity asset graph."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.errors import BeatVisualExecutionError, VisualOperationResolutionError
from app.generators.image_prompt import ImagePromptBuilder
from app.generators.master_scene import (
    ContinuityGenerationRequest,
    generate_continuity_image,
    generate_required_master_scenes,
)
from app.models.beat_visual import (
    BeatVisualGenerationStatus,
    BeatVisualQAStatus,
    OverlaySpecification,
    OverlayType,
    TransformSpecification,
    TransformType,
)
from app.models.visual_plan import (
    CameraMovement,
    VisualBeat,
    VisualOperation,
    VisualPlan,
)
from app.models.visual_qa import VisualQADecision, VisualQAResult
from app.persistence import (
    BeatVisualResult,
    MasterSceneAsset,
    Project,
    ProjectVisualExecutionPlan,
    StyleReferenceAsset,
    VisualOperationDecisionRecord,
)
from app.pipeline.visual_qa import (
    VisualQAContext,
    VisualQAService,
    apply_visual_qa_correction,
    build_visual_qa_request,
    is_hard_qa_failure,
    qa_candidate_penalty,
)
from app.provider_capabilities import ImageProviderCapabilities
from app.provider_diagnostics import diagnostic_from_exception
from app.providers import (
    ImageProvider,
    get_image_provider,
    get_image_provider_capabilities,
)
from app.repositories import (
    apply_automated_visual_qa_decision,
    create_beat_visual_qa_evaluation,
    create_beat_visual_result,
    get_beat_visual_qa_evaluation_by_revision,
    get_project,
    get_project_visual_plan_record,
    get_style_reference_asset,
    get_successful_beat_visual_result,
    get_visual_execution_plan,
    get_visual_operation_decision,
    list_beat_visual_results,
    list_master_scene_assets,
    mark_beat_visual_result_failed,
    mark_beat_visual_result_succeeded,
    next_beat_visual_attempt,
)
from app.services.visual_asset_selection import (
    SelectedVisualReference,
    select_source_asset,
    select_visual_references,
)
from app.services.visual_operations import resolve_project_visual_operations
from app.services.visual_planning import require_current_project_visual_plan
from app.storage import ProjectMediaPaths
from app.style_contracts import DEFAULT_IMAGE_STYLE_ID
from app.utils.download import download_file

logger = logging.getLogger(__name__)

ProviderResolver = Callable[[str, Mapping[str, Any] | None], ImageProvider]
Downloader = Callable[[str, str], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class VisualAssetExecutionSummary:
    beats: int
    operation_counts: Mapping[str, int]
    api_image_calls: int
    unique_visual_assets: int
    lineage: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VisualQAExecutionSummary:
    beats: int
    generated_candidates: int
    pass_first_attempt: int
    pass_with_warning: int
    regenerated: int
    continuity_failures: int
    style_drift_failures: int
    composition_failures: int
    average_attempts_per_accepted_generated_beat: float


@dataclass(frozen=True, slots=True)
class _ExecutionContext:
    project: Project
    plan: VisualPlan
    execution_plan: ProjectVisualExecutionPlan
    provider: ImageProvider
    capabilities: ImageProviderCapabilities
    master_assets: dict[str, MasterSceneAsset]
    style_reference: StyleReferenceAsset | None


class VisualBeatAssetExecutor:
    """Execute beats sequentially while preserving state, lineage, and retries."""

    def __init__(
        self,
        session: Session,
        *,
        provider_resolver: ProviderResolver = get_image_provider,
        downloader: Downloader = download_file,
        projects_root: str | Path = "data/projects",
        style_id: str = DEFAULT_IMAGE_STYLE_ID,
        qa_service: VisualQAService | None = None,
        max_visual_qa_attempts: int = 3,
    ) -> None:
        if not 1 <= max_visual_qa_attempts <= 5:
            raise ValueError("max_visual_qa_attempts must be between 1 and 5")
        self.session = session
        self.provider_resolver = provider_resolver
        self.downloader = downloader
        self.projects_root = projects_root
        self.style_id = style_id
        self.qa_service = qa_service
        self.max_visual_qa_attempts = max_visual_qa_attempts

    async def execute_project(
        self,
        project_id: str,
        execution_plan_id: str,
    ) -> list[BeatVisualResult]:
        """Generate required masters, then execute all resolved beats in order."""
        context = await self._prepare_context(project_id, execution_plan_id)
        results: list[BeatVisualResult] = []
        for decision in sorted(
            context.execution_plan.decisions,
            key=lambda item: item.position,
        ):
            results.append(await self._execute_decision(context, decision))
        summary = build_visual_asset_execution_summary(results)
        qa_summary = build_visual_qa_execution_summary(
            list_beat_visual_results(
                self.session,
                project_id,
                execution_plan_id=context.execution_plan.id,
            )
        )
        logger.info(
            "Visual asset graph execution completed",
            extra={
                "project_id": project_id,
                "beat_count": summary.beats,
                "operation_counts": dict(summary.operation_counts),
                "api_image_calls": summary.api_image_calls,
                "unique_visual_assets": summary.unique_visual_assets,
                "lineage": list(summary.lineage),
                "visual_qa": asdict(qa_summary),
            },
        )
        return results

    async def execute_beat(
        self,
        project_id: str,
        execution_plan_id: str,
        beat_id: str,
    ) -> BeatVisualResult:
        """Execute or retry one beat without restarting successful predecessors."""
        context = await self._prepare_context(project_id, execution_plan_id)
        decision = get_visual_operation_decision(
            self.session,
            context.execution_plan.id,
            beat_id,
        )
        if decision is None:
            raise ValueError(f"Visual beat decision not found: {beat_id}")
        return await self._execute_decision(context, decision)

    async def _prepare_context(
        self,
        project_id: str,
        execution_plan_id: str,
    ) -> _ExecutionContext:
        state = require_current_project_visual_plan(self.session, project_id)
        project = get_project(self.session, project_id)
        plan_record = get_project_visual_plan_record(self.session, project_id)
        execution_plan = get_visual_execution_plan(
            self.session,
            execution_plan_id,
        )
        if project is None or plan_record is None or execution_plan is None:
            raise ValueError("Project visual execution context is incomplete")
        if (
            execution_plan.project_id != project_id
            or execution_plan.visual_plan_id != plan_record.id
            or execution_plan.visual_plan_revision
            != _stable_hash(state.plan.model_dump(mode="json"))
        ):
            raise VisualOperationResolutionError(
                "Visual execution plan does not match the current semantic plan"
            )

        provider_options: dict[str, Any] = {}
        if project.image_model is not None:
            provider_options["model"] = project.image_model
        provider = self.provider_resolver(project.image_provider, provider_options)
        capabilities = get_image_provider_capabilities(provider)
        actual_model = getattr(provider, "model", project.image_model)
        if (
            execution_plan.provider != project.image_provider
            or execution_plan.model != actual_model
            or execution_plan.capability_snapshot != capabilities.snapshot()
        ):
            raise VisualOperationResolutionError(
                "Visual execution plan does not match the selected image provider"
            )

        style_reference = get_style_reference_asset(
            self.session,
            project_id,
            self.style_id,
        )
        await generate_required_master_scenes(
            self.session,
            project,
            state.plan,
            provider,
            projects_root=self.projects_root,
            style_id=self.style_id,
            style_reference=style_reference,
            capabilities=capabilities,
            downloader=self.downloader,
        )

        # Master availability is a Stage 2 resolution input. Re-resolve once after
        # masters exist instead of silently using decisions made without them.
        current_execution_plan = resolve_project_visual_operations(
            self.session,
            project_id,
            provider_resolver=lambda name, config: provider,
        )
        if current_execution_plan.id != execution_plan.id:
            logger.info(
                "Visual operations re-resolved after master generation",
                extra={
                    "project_id": project_id,
                    "previous_execution_plan_id": execution_plan.id,
                    "execution_plan_id": current_execution_plan.id,
                },
            )
        masters = {
            asset.master_scene_id: asset
            for asset in list_master_scene_assets(self.session, project_id)
        }
        return _ExecutionContext(
            project=project,
            plan=state.plan,
            execution_plan=current_execution_plan,
            provider=provider,
            capabilities=capabilities,
            master_assets=masters,
            style_reference=style_reference,
        )

    async def _execute_decision(
        self,
        context: _ExecutionContext,
        decision: VisualOperationDecisionRecord,
    ) -> BeatVisualResult:
        beat = _get_beat(context.plan, decision.beat_id)
        operation = VisualOperation(decision.resolved_operation)
        accepted_results = [
            result
            for result in list_beat_visual_results(
                self.session,
                context.project.id,
                accepted_only=True,
            )
            if result.visual_plan_id == context.execution_plan.visual_plan_id
        ]
        source = None
        if operation is not VisualOperation.NEW_IMAGE:
            source = select_source_asset(
                context.plan,
                beat,
                decision,
                accepted_results,
                context.master_assets,
            )
        master_asset = (
            context.master_assets.get(beat.master_scene_id)
            if beat.master_scene_id is not None
            else None
        )
        selected_references = select_visual_references(
            beat,
            operation,
            context.capabilities,
            style_reference=context.style_reference,
            master_asset=master_asset,
            source=source,
        )
        references = tuple(item.reference for item in selected_references)
        prompt = self._build_prompt(
            context.project,
            context.plan,
            beat,
            operation,
            references,
        )
        transform = _build_transform(beat) if operation is VisualOperation.TRANSFORM else None
        overlay = _build_overlay(beat) if operation is VisualOperation.OVERLAY else None
        semantic_state = _semantic_state_snapshot(beat, decision.position)
        reference_snapshot = _reference_snapshot(selected_references)
        generation_revision = _stable_hash(
            {
                "visual_plan_revision": context.execution_plan.visual_plan_revision,
                "execution_plan_revision": context.execution_plan.resolution_revision,
                "beat": semantic_state,
                "resolved_operation": operation.value,
                "source_asset_id": source.asset_id if source else None,
                "source_sha256": source.sha256 if source else None,
                "master_asset_id": master_asset.id if master_asset else None,
                "references": reference_snapshot,
                "style_version": self.style_id,
                "provider": context.execution_plan.provider,
                "model": context.execution_plan.model,
                "prompt": prompt,
                "transform": transform,
                "overlay": overlay,
                "visual_qa_policy": {
                    "enabled": self.qa_service is not None,
                    "provider": self.qa_service.provider if self.qa_service else None,
                    "model": self.qa_service.model if self.qa_service else None,
                    "prompt_version": (
                        self.qa_service.prompt_version if self.qa_service else None
                    ),
                    "max_attempts": self.max_visual_qa_attempts,
                },
            }
        )
        existing = get_successful_beat_visual_result(
            self.session,
            execution_plan_id=context.execution_plan.id,
            beat_id=beat.id,
            generation_revision=generation_revision,
        )
        if existing is not None and await _verify_result_file(existing):
            logger.info(
                "Beat visual asset reused by generation revision",
                extra={
                    "project_id": context.project.id,
                    "beat_id": beat.id,
                    "generation_revision": generation_revision,
                    "beat_visual_result_id": existing.id,
                },
            )
            return existing

        generated_operations = {
            VisualOperation.NEW_IMAGE,
            VisualOperation.REFERENCE_GENERATION,
            VisualOperation.EDIT_EXISTING,
        }
        should_run_qa = operation in generated_operations and self.qa_service is not None
        total_attempts = self.max_visual_qa_attempts if should_run_qa else 1
        correction: str | None = None
        rejected_candidates: list[tuple[BeatVisualResult, VisualQADecision]] = []

        for qa_attempt in range(1, total_attempts + 1):
            attempt = next_beat_visual_attempt(
                self.session,
                execution_plan_id=context.execution_plan.id,
                beat_id=beat.id,
                generation_revision=generation_revision,
            )
            if operation in generated_operations:
                planned_output_path = str(
                    ProjectMediaPaths(
                        context.project.id,
                        self.projects_root,
                    ).visual_beat_path(beat.id, generation_revision, attempt)
                )
            else:
                assert source is not None
                planned_output_path = source.path
            prompt_used = apply_visual_qa_correction(
                prompt or "",
                correction,
                self.style_id,
            ) if correction is not None else prompt
            result = create_beat_visual_result(
                self.session,
                project_id=context.project.id,
                visual_plan_id=context.execution_plan.visual_plan_id,
                visual_plan_revision=context.execution_plan.visual_plan_revision,
                execution_plan_id=context.execution_plan.id,
                beat_id=beat.id,
                resolved_operation=operation.value,
                source_result_id=source.result_id if source else None,
                source_master_asset_id=(
                    source.master_asset_id if source is not None else None
                ),
                source_path=source.path if source is not None else None,
                output_path=planned_output_path,
                file_sha256=None,
                master_scene_id=beat.master_scene_id,
                prompt_used=prompt_used,
                provider=context.execution_plan.provider,
                model=context.execution_plan.model,
                style_version=self.style_id,
                reference_snapshot=reference_snapshot,
                generation_status=BeatVisualGenerationStatus.PENDING.value,
                qa_status=BeatVisualQAStatus.NOT_RUN.value,
                is_accepted=False,
                error=None,
                transform_metadata=(
                    transform.model_dump(mode="json")
                    if transform is not None
                    else None
                ),
                overlay_metadata=(
                    overlay.model_dump(mode="json") if overlay is not None else None
                ),
                semantic_state_snapshot=semantic_state,
                generation_revision=generation_revision,
                attempt=attempt,
            )
            try:
                if operation in generated_operations:
                    request = ContinuityGenerationRequest(
                        operation=operation,
                        prompt=prompt_used or "",
                        master_scene_id=beat.master_scene_id,
                        master_image_path=(
                            master_asset.file_path
                            if master_asset is not None
                            else None
                        ),
                        style_version=self.style_id,
                        references=references,
                    )
                    output_path = await generate_continuity_image(
                        request,
                        planned_output_path,
                        context.provider,
                        downloader=self.downloader,
                    )
                    file_sha256 = await asyncio.to_thread(_sha256_file, output_path)
                else:
                    assert source is not None
                    output_path = planned_output_path
                    file_sha256 = source.sha256
            except Exception as exc:
                summary = f"Failed to execute visual beat {beat.id}"
                provider_operation = (
                    "edit"
                    if operation is VisualOperation.EDIT_EXISTING
                    else "reference"
                    if references
                    else "generate"
                )
                diagnostic = diagnostic_from_exception(
                    exc,
                    provider=context.execution_plan.provider,
                    model=context.execution_plan.model,
                    operation=provider_operation,
                    request_stage="visual_beat_generation",
                ).with_context(
                    request_stage="visual_beat_generation",
                    beat_id=beat.id,
                )
                mark_beat_visual_result_failed(
                    self.session,
                    result,
                    error=_safe_error(exc),
                )
                logger.error("%s", diagnostic.format(summary))
                raise BeatVisualExecutionError(
                    f"{summary}: {_safe_error(exc)}",
                    diagnostic=diagnostic,
                    user_summary=summary,
                ) from exc

            result = mark_beat_visual_result_succeeded(
                self.session,
                result,
                output_path=output_path,
                file_sha256=file_sha256,
                accept=not should_run_qa,
            )
            if not should_run_qa:
                return result

            assert self.qa_service is not None
            qa_context = _build_qa_context(
                context,
                beat,
                operation,
                prompt_used or "",
                source_path=source.path if source is not None else None,
                master_asset=master_asset,
                style_id=self.style_id,
            )
            qa_revision = _build_qa_revision(
                result,
                qa_context,
                self.qa_service,
                source_sha256=source.sha256 if source is not None else None,
                master_sha256=(
                    master_asset.file_sha256 if master_asset is not None else None
                ),
                style_reference_sha256=(
                    context.style_reference.file_sha256
                    if context.style_reference is not None
                    else None
                ),
            )
            previous_evaluation = get_beat_visual_qa_evaluation_by_revision(
                self.session,
                qa_revision,
            )
            if previous_evaluation is None:
                try:
                    qa_decision = await self.qa_service.evaluate(output_path, qa_context)
                except Exception as exc:
                    raise BeatVisualExecutionError(
                        f"Visual QA failed for beat {beat.id}: {_safe_error(exc)}"
                    ) from exc
            else:
                qa_decision = VisualQADecision.model_validate(
                    previous_evaluation.decision_snapshot
                )
            create_beat_visual_qa_evaluation(
                self.session,
                result,
                qa_revision=qa_revision,
                prompt_version=self.qa_service.prompt_version,
                provider=self.qa_service.provider,
                model=self.qa_service.model,
                qa_attempt=qa_attempt,
                decision=qa_decision,
            )
            result = apply_automated_visual_qa_decision(
                self.session,
                result,
                qa_decision,
                qa_revision=qa_revision,
                prompt_version=self.qa_service.prompt_version,
                provider=self.qa_service.provider,
                model=self.qa_service.model,
                qa_attempt=qa_attempt,
            )
            if result.is_accepted:
                return result
            rejected_candidates.append((result, qa_decision))
            correction = qa_decision.correction_instruction

        usable_candidates = [
            item
            for item in rejected_candidates
            if not is_hard_qa_failure(item[1])
            and (
                item[1].severity is None
                or item[1].severity.value == "minor"
            )
        ]
        if usable_candidates:
            best_result, best_decision = min(
                usable_candidates,
                key=lambda item: qa_candidate_penalty(item[1]),
            )
            warning = (
                f"Visual QA did not pass after {total_attempts} attempts; "
                "accepted the best usable candidate with warning"
            )
            accepted_decision = VisualQADecision(
                result=VisualQAResult.PASS_WITH_WARNING,
                scores=best_decision.scores,
                problem_categories=best_decision.problem_categories,
                reasons=best_decision.reasons,
                correction_instruction=best_decision.correction_instruction,
                severity=best_decision.severity,
            )
            return apply_automated_visual_qa_decision(
                self.session,
                best_result,
                accepted_decision,
                qa_revision=best_result.qa_revision or "",
                prompt_version=self.qa_service.prompt_version,
                provider=self.qa_service.provider,
                model=self.qa_service.model,
                qa_attempt=best_result.qa_attempt or total_attempts,
                warning=warning,
            )
        raise BeatVisualExecutionError(
            f"Visual QA rejected all {total_attempts} candidates for beat {beat.id}"
        )

    @staticmethod
    def _build_prompt(
        project: Project,
        plan: VisualPlan,
        beat: VisualBeat,
        operation: VisualOperation,
        references: tuple[Any, ...],
    ) -> str | None:
        if operation in {
            VisualOperation.REUSE,
            VisualOperation.TRANSFORM,
            VisualOperation.OVERLAY,
        }:
            return None
        builder = ImagePromptBuilder()
        if operation is VisualOperation.EDIT_EXISTING:
            return builder.build_edit(
                plan,
                beat,
                references=references,
                project_style_prompt=project.global_image_style_prompt,
            )
        return builder.build(
            plan,
            beat,
            operation,
            references=references,
            project_style_prompt=project.global_image_style_prompt,
        )


def _build_qa_context(
    context: _ExecutionContext,
    beat: VisualBeat,
    operation: VisualOperation,
    generation_prompt: str,
    *,
    source_path: str | None,
    master_asset: MasterSceneAsset | None,
    style_id: str,
) -> VisualQAContext:
    character_ids = set(beat.characters_visible)
    object_ids = set(beat.important_objects)
    character_definitions = tuple(
        f"{item.id}: {item.description}"
        for item in context.plan.characters
        if item.id in character_ids
    )
    object_definitions = tuple(
        f"{item.id}: {item.description}; story function: {item.story_function}"
        for item in context.plan.important_objects
        if item.id in object_ids
    )
    return VisualQAContext(
        visual_purpose=beat.visual_purpose,
        what_viewer_should_understand=beat.what_viewer_should_understand,
        required_objects=tuple(beat.important_objects),
        important_physical_action=beat.change_from_previous_beat,
        location_id=beat.location_id,
        expected_physical_state=beat.physical_state,
        resolved_operation=operation.value,
        characters_visible=tuple(beat.characters_visible),
        character_definitions=character_definitions,
        object_definitions=object_definitions,
        camera_view=f"{beat.camera_framing.value}: {beat.camera_view}",
        change_from_previous=beat.change_from_previous_beat,
        generation_prompt=generation_prompt,
        style_id=style_id,
        style_reference_path=(
            context.style_reference.file_path
            if context.style_reference is not None
            else None
        ),
        master_reference_path=(
            master_asset.file_path if master_asset is not None else None
        ),
        previous_frame_path=source_path,
        source_reference_path=(
            source_path if operation is VisualOperation.EDIT_EXISTING else None
        ),
        information_added_beyond_narration=(
            beat.information_added_beyond_narration
        ),
    )


def _build_qa_revision(
    result: BeatVisualResult,
    qa_context: VisualQAContext,
    qa_service: VisualQAService,
    *,
    source_sha256: str | None,
    master_sha256: str | None,
    style_reference_sha256: str | None,
) -> str:
    return _stable_hash(
        {
            "candidate_sha256": result.file_sha256,
            "visual_plan_revision": result.visual_plan_revision,
            "semantic_state": result.semantic_state_snapshot,
            "source_sha256": source_sha256,
            "master_sha256": master_sha256,
            "style_reference_sha256": style_reference_sha256,
            "qa_prompt": build_visual_qa_request(qa_context),
            "qa_prompt_version": qa_service.prompt_version,
            "qa_provider": qa_service.provider,
            "qa_model": qa_service.model,
        }
    )


def build_visual_asset_execution_summary(
    results: list[BeatVisualResult],
) -> VisualAssetExecutionSummary:
    accepted = [item for item in results if item.is_accepted]
    counts = Counter(item.resolved_operation for item in accepted)
    api_operations = {
        VisualOperation.NEW_IMAGE.value,
        VisualOperation.REFERENCE_GENERATION.value,
        VisualOperation.EDIT_EXISTING.value,
    }
    api_calls = sum(counts[operation] for operation in api_operations)
    unique_paths = {item.output_path for item in accepted if item.output_path}
    beat_by_result_id = {item.id: item.beat_id for item in accepted}
    by_parent: dict[str | None, list[BeatVisualResult]] = {}
    for item in accepted:
        parent = item.source_result_id or item.source_master_asset_id
        by_parent.setdefault(parent, []).append(item)
    lineage = tuple(
        f"{_lineage_parent_label(parent, child, beat_by_result_id)} -> "
        f"{child.beat_id} [{child.resolved_operation}]"
        for parent, children in by_parent.items()
        for child in children
    )
    return VisualAssetExecutionSummary(
        beats=len(accepted),
        operation_counts=dict(sorted(counts.items())),
        api_image_calls=api_calls,
        unique_visual_assets=len(unique_paths),
        lineage=lineage,
    )


def build_visual_qa_execution_summary(
    results: list[BeatVisualResult],
) -> VisualQAExecutionSummary:
    generated_operations = {
        VisualOperation.NEW_IMAGE.value,
        VisualOperation.REFERENCE_GENERATION.value,
        VisualOperation.EDIT_EXISTING.value,
    }
    candidates = [
        item
        for item in results
        if item.resolved_operation in generated_operations
        and item.generation_status == BeatVisualGenerationStatus.SUCCEEDED.value
    ]
    accepted = [item for item in candidates if item.is_accepted]
    category_values = [
        category
        for item in candidates
        for category in (item.qa_problem_categories or [])
    ]
    continuity_categories = {
        "CONTINUITY",
        "LOCATION_DRIFT",
        "CHARACTER_DRIFT",
        "OBJECT_DRIFT",
        "EDIT_CHANGED_TOO_MUCH",
        "REFERENCE_NOT_RESPECTED",
    }
    style_categories = {
        "STYLE_DRIFT",
        "STYLE_DRIFT_REALISM",
        "STYLE_DRIFT_DETAIL",
        "STYLE_DRIFT_CHILDISH",
    }
    composition_categories = {
        "COMPOSITION",
        "COMPOSITION_UNCLEAR",
        "IMPORTANT_ACTION_TOO_SMALL",
        "EXCESSIVE_CLUTTER",
        "VIDEO_READABILITY",
    }
    return VisualQAExecutionSummary(
        beats=len({item.beat_id for item in accepted}),
        generated_candidates=len(candidates),
        pass_first_attempt=sum(
            item.qa_result == VisualQAResult.PASS.value and item.qa_attempt == 1
            for item in accepted
        ),
        pass_with_warning=sum(
            item.qa_result == VisualQAResult.PASS_WITH_WARNING.value
            for item in accepted
        ),
        regenerated=sum(max(item.attempt - 1, 0) for item in accepted),
        continuity_failures=sum(
            category in continuity_categories for category in category_values
        ),
        style_drift_failures=sum(
            category in style_categories for category in category_values
        ),
        composition_failures=sum(
            category in composition_categories for category in category_values
        ),
        average_attempts_per_accepted_generated_beat=(
            round(sum(item.attempt for item in accepted) / len(accepted), 2)
            if accepted
            else 0.0
        ),
    )


def _lineage_parent_label(
    parent_id: str | None,
    child: BeatVisualResult,
    beat_by_result_id: Mapping[str, str],
) -> str:
    if parent_id is None:
        return "root"
    return beat_by_result_id.get(
        parent_id,
        f"master:{child.master_scene_id or parent_id}",
    )


def _get_beat(plan: VisualPlan, beat_id: str) -> VisualBeat:
    beat = next((item for item in plan.visual_beats if item.id == beat_id), None)
    if beat is None:
        raise ValueError(f"Visual beat not found: {beat_id}")
    return beat


def _semantic_state_snapshot(beat: VisualBeat, position: int) -> dict[str, Any]:
    return {
        "beat_position": position,
        "location_id": beat.location_id,
        "master_scene_id": beat.master_scene_id,
        "characters_visible": list(beat.characters_visible),
        "important_objects": list(beat.important_objects),
        "camera_framing": beat.camera_framing.value,
        "camera_view": beat.camera_view,
        "physical_state": beat.physical_state,
        "progressive_change": (
            beat.progressive_change.model_dump(mode="json")
            if beat.progressive_change is not None
            else None
        ),
    }


def _reference_snapshot(
    selected: tuple[SelectedVisualReference, ...],
) -> list[dict[str, str]]:
    return [
        {
            "asset_id": item.reference.reference_id,
            "sha256": item.reference.sha256,
            "role": item.semantic_role,
            "reason": item.reason,
        }
        for item in selected
    ]


def _build_transform(beat: VisualBeat) -> TransformSpecification:
    movement_types = {
        CameraMovement.SUBTLE_ZOOM: TransformType.ZOOM_IN,
        CameraMovement.ZOOM: TransformType.ZOOM_IN,
        CameraMovement.PAN: TransformType.PAN,
        CameraMovement.CROP: TransformType.CROP,
        CameraMovement.HIGHLIGHT: TransformType.FOCUS,
        CameraMovement.STATIC: TransformType.FOCUS,
    }
    transform_type = movement_types[beat.camera_movement]
    return TransformSpecification(
        type=transform_type,
        target=beat.visual_focus or beat.camera_view,
        start_scale=1.0 if transform_type is TransformType.ZOOM_IN else None,
        end_scale=1.15 if transform_type is TransformType.ZOOM_IN else None,
    )


def _build_overlay(beat: VisualBeat) -> OverlaySpecification:
    description = beat.overlay_description
    if description is None:
        raise ValueError("OVERLAY beat does not contain overlay instructions")
    lowered = description.lower()
    overlay_type = OverlayType.HIGHLIGHT
    keywords = (
        ("distance", OverlayType.DISTANCE),
        ("depth", OverlayType.DEPTH),
        ("measurement", OverlayType.MEASUREMENT),
        ("route", OverlayType.ROUTE),
        ("arrow", OverlayType.ARROW),
        ("circle", OverlayType.CIRCLE),
        ("x mark", OverlayType.X_MARK),
        ("label", OverlayType.TEXT_LABEL),
    )
    for keyword, candidate in keywords:
        if keyword in lowered:
            overlay_type = candidate
            break
    return OverlaySpecification(
        type=overlay_type,
        instruction=description,
        semantic_anchor=beat.visual_focus or beat.what_viewer_should_understand,
    )


async def _verify_result_file(result: BeatVisualResult) -> bool:
    if result.output_path is None or result.file_sha256 is None:
        return False
    path = Path(result.output_path)
    if not path.is_file():
        return False
    actual_hash = await asyncio.to_thread(_sha256_file, path)
    return actual_hash == result.file_sha256


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=lambda item: item.model_dump(mode="json"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_error(exc: Exception) -> str:
    name = type(exc).__name__
    message = str(exc).strip()
    if not message:
        return name
    redacted = re.sub(
        r"(?i)(bearer\s+)[^\s,;]+",
        r"\1[REDACTED]",
        message,
    )
    redacted = re.sub(r"\bsk-[A-Za-z0-9_-]{6,}\b", "[REDACTED]", redacted)
    redacted = re.sub(
        r"(?i)((?:api[_ -]?key|token)\s*[=:]\s*)[^\s,;]+",
        r"\1[REDACTED]",
        redacted,
    )
    return redacted[:500]
