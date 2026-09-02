"""Resolve semantic beats into durable provider-specific visual operations."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.models.visual_plan import VisualOperation
from app.persistence import ProjectVisualExecutionPlan
from app.pipeline.visual_operation_engine import (
    VisualDecisionEvidence,
    VisualOperationDecisionEngine,
)
from app.providers import (
    ImageProvider,
    get_image_provider,
    get_image_provider_capabilities,
)
from app.repositories import (
    get_project,
    get_project_visual_plan_record,
    get_visual_execution_plan_by_revision,
    list_master_scene_assets,
    save_visual_execution_plan,
)
from app.services.visual_planning import require_current_project_visual_plan

logger = logging.getLogger(__name__)

ProviderResolver = Callable[[str, Mapping[str, Any] | None], ImageProvider]


def resolve_project_visual_operations(
    session: Session,
    project_id: str,
    *,
    provider_resolver: ProviderResolver = get_image_provider,
    available_visuals: Mapping[str, str] | None = None,
    evidence_by_beat: Mapping[str, VisualDecisionEvidence] | None = None,
) -> ProjectVisualExecutionPlan:
    """Resolve all beats without generating, editing, or downloading images."""
    state = require_current_project_visual_plan(session, project_id)
    project = get_project(session, project_id)
    plan_record = get_project_visual_plan_record(session, project_id)
    assert project is not None and plan_record is not None

    provider_options: dict[str, Any] = {}
    if project.image_model is not None:
        provider_options["model"] = project.image_model
    provider = provider_resolver(project.image_provider, provider_options)
    capabilities = get_image_provider_capabilities(provider)
    model = getattr(provider, "model", project.image_model)

    initial_visuals, reference_snapshot = _build_reference_inventory(
        session,
        project_id,
        available_visuals or {},
        {
            visual_id
            for beat in state.plan.visual_beats
            for visual_id in (beat.source_visual_id, beat.geography_established_by)
            if visual_id is not None
        }
        | {master.id for master in state.plan.possible_master_scenes},
    )
    evidence = dict(evidence_by_beat or {})
    decision_input_snapshot = {
        "references": reference_snapshot,
        "evidence": {
            beat_id: asdict(value) for beat_id, value in sorted(evidence.items())
        },
    }
    plan_revision = _stable_hash(state.plan.model_dump(mode="json"))
    capability_snapshot = capabilities.snapshot()
    resolution_revision = _stable_hash(
        {
            "project_id": project_id,
            "visual_plan_id": plan_record.id,
            "visual_plan_schema_version": plan_record.schema_version,
            "visual_plan_revision": plan_revision,
            "story_text_hash": plan_record.story_text_hash,
            "provider": project.image_provider,
            "model": model,
            "capabilities": capability_snapshot,
            "decision_inputs": decision_input_snapshot,
        }
    )
    existing = get_visual_execution_plan_by_revision(
        session,
        resolution_revision,
    )
    if existing is not None:
        _log_resolution(
            "resolution_reused",
            project_id,
            project.image_provider,
            model,
            resolution_revision,
            len(existing.decisions),
        )
        return existing

    engine = VisualOperationDecisionEngine()
    planned_visuals = dict(initial_visuals)
    records: list[dict[str, Any]] = []
    for position, beat in enumerate(state.plan.visual_beats):
        decision = engine.decide(
            state.plan,
            position,
            capabilities=capabilities,
            available_visuals=planned_visuals,
            evidence=evidence.get(beat.id),
        )
        records.append(
            {
                "position": position,
                "beat_id": decision.beat_id,
                "preferred_operation": decision.requested_operation.value,
                "resolved_operation": decision.operation.value,
                "fallback_used": decision.fallback_from is not None,
                "fallback_from": (
                    decision.fallback_from.value
                    if decision.fallback_from is not None
                    else None
                ),
                "reason": list(decision.reasons),
                "source_visual_ids": list(decision.source_visual_ids),
                "source_image_paths": [
                    path
                    for path in decision.source_image_paths
                    if not path.startswith("planned://")
                ],
            }
        )
        logger.info(
            "Visual beat operation resolved",
            extra={
                "visual_operation_event": "beat_resolved",
                "project_id": project_id,
                "provider": project.image_provider,
                "model": model,
                "beat_id": decision.beat_id,
                "preferred_operation": decision.requested_operation.value,
                "resolved_operation": decision.operation.value,
                "fallback_used": decision.fallback_from is not None,
                "decision_reason": "; ".join(decision.reasons),
            },
        )
        _record_planned_output(planned_visuals, beat.id, decision.operation, decision)

    result = save_visual_execution_plan(
        session,
        project_id=project_id,
        visual_plan=plan_record,
        visual_plan_revision=plan_revision,
        provider=project.image_provider,
        model=model,
        capability_snapshot=capability_snapshot,
        decision_input_snapshot=decision_input_snapshot,
        resolution_revision=resolution_revision,
        decisions=records,
    )
    _log_resolution(
        "resolution_persisted",
        project_id,
        project.image_provider,
        model,
        resolution_revision,
        len(records),
    )
    return result


def _build_reference_inventory(
    session: Session,
    project_id: str,
    available_visuals: Mapping[str, str],
    relevant_visual_ids: set[str],
) -> tuple[dict[str, str], dict[str, Any]]:
    usable: dict[str, str] = {}
    snapshot: dict[str, Any] = {"masters": {}, "visuals": {}}
    for asset in list_master_scene_assets(session, project_id):
        if asset.master_scene_id not in relevant_visual_ids:
            continue
        exists = Path(asset.file_path).is_file()
        snapshot["masters"][asset.master_scene_id] = {
            "file_path": asset.file_path,
            "file_sha256": asset.file_sha256,
            "available": exists,
        }
        if exists:
            usable[asset.master_scene_id] = asset.file_path
    for visual_id, path in sorted(available_visuals.items()):
        if visual_id not in relevant_visual_ids:
            continue
        exists = Path(path).is_file()
        snapshot["visuals"][visual_id] = {
            "file_path": path,
            "available": exists,
        }
        if exists:
            usable[visual_id] = path
    return usable, snapshot


def _record_planned_output(
    available_visuals: dict[str, str],
    beat_id: str,
    operation: VisualOperation,
    decision: Any,
) -> None:
    if operation in {
        VisualOperation.NEW_IMAGE,
        VisualOperation.REFERENCE_GENERATION,
        VisualOperation.EDIT_EXISTING,
    }:
        available_visuals[beat_id] = f"planned://{beat_id}"
    elif decision.source_image_paths:
        available_visuals[beat_id] = decision.source_image_paths[0]


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _log_resolution(
    event: str,
    project_id: str,
    provider: str,
    model: str | None,
    revision: str,
    beat_count: int,
) -> None:
    logger.info(
        "Project visual operations resolved",
        extra={
            "visual_operation_event": event,
            "project_id": project_id,
            "provider": provider,
            "model": model,
            "resolution_revision": revision,
            "beat_count": beat_count,
        },
    )
