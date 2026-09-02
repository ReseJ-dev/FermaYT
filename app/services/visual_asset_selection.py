"""Continuity-aware source and reference selection for visual beat assets."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from app.generators.style_reference import to_style_image_reference
from app.models.visual_plan import VisualBeat, VisualOperation, VisualPlan
from app.persistence import (
    BeatVisualResult,
    MasterSceneAsset,
    StyleReferenceAsset,
    VisualOperationDecisionRecord,
)
from app.provider_capabilities import ImageProviderCapabilities
from app.providers import ImageReference, ImageReferenceRole

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VisualAssetSource:
    asset_id: str
    path: str
    sha256: str
    reason: str
    result_id: str | None = None
    master_asset_id: str | None = None
    visual_id: str | None = None


@dataclass(frozen=True, slots=True)
class SelectedVisualReference:
    reference: ImageReference
    semantic_role: str
    reason: str


def select_source_asset(
    plan: VisualPlan,
    beat: VisualBeat,
    decision: VisualOperationDecisionRecord,
    accepted_results: list[BeatVisualResult],
    master_assets: dict[str, MasterSceneAsset],
) -> VisualAssetSource:
    """Choose the most relevant state/composition, not merely the latest frame."""
    candidates: list[tuple[float, int, VisualAssetSource]] = []
    current_position = decision.position
    explicit_sources = set(decision.source_visual_ids)

    for result in accepted_results:
        state = result.semantic_state_snapshot
        position = int(state.get("beat_position", -1))
        if position >= current_position or not result.output_path or not result.file_sha256:
            continue
        if (
            not Path(result.output_path).is_file()
            or _sha256_file(result.output_path) != result.file_sha256
        ):
            continue
        score, reasons = _score_result_source(
            beat,
            result,
            explicit_sources,
            decision.resolved_operation,
        )
        candidates.append(
            (
                score,
                position,
                VisualAssetSource(
                    asset_id=result.id,
                    result_id=result.id,
                    path=result.output_path,
                    sha256=result.file_sha256,
                    visual_id=result.beat_id,
                    reason="; ".join(reasons),
                ),
            )
        )

    master_definitions = {item.id: item for item in plan.possible_master_scenes}
    for master_id, asset in master_assets.items():
        definition = master_definitions.get(master_id)
        if (
            definition is None
            or not Path(asset.file_path).is_file()
            or _sha256_file(asset.file_path) != asset.file_sha256
        ):
            continue
        if (
            decision.resolved_operation
            in {
                VisualOperation.REUSE.value,
                VisualOperation.TRANSFORM.value,
                VisualOperation.OVERLAY.value,
            }
            and master_id not in explicit_sources
        ):
            continue
        score = 0.0
        reasons: list[str] = []
        if master_id == beat.master_scene_id:
            score += 18
            reasons.append("same immutable master scene")
        if definition.location_id == beat.location_id:
            score += 8
            reasons.append("same location")
        if master_id in explicit_sources:
            score += 20
            reasons.append("explicit semantic source")
        candidates.append(
            (
                score,
                -1,
                VisualAssetSource(
                    asset_id=asset.id,
                    master_asset_id=asset.id,
                    path=asset.file_path,
                    sha256=asset.file_sha256,
                    visual_id=master_id,
                    reason="; ".join(reasons) or "available master scene",
                ),
            )
        )

    if not candidates:
        raise ValueError(f"No usable source asset for beat {beat.id}")
    _, _, selected = max(candidates, key=lambda item: (item[0], item[1]))
    logger.info(
        "Visual source asset selected",
        extra={
            "beat_id": beat.id,
            "source_asset_id": selected.asset_id,
            "source_visual_id": selected.visual_id,
            "selection_reason": selected.reason,
        },
    )
    return selected


def select_visual_references(
    beat: VisualBeat,
    operation: VisualOperation,
    capabilities: ImageProviderCapabilities,
    *,
    style_reference: StyleReferenceAsset | None,
    master_asset: MasterSceneAsset | None,
    source: VisualAssetSource | None,
) -> tuple[SelectedVisualReference, ...]:
    """Select the minimum useful references in deterministic semantic order."""
    if not capabilities.reference_generation:
        return ()
    candidates: list[SelectedVisualReference] = []
    if style_reference is not None:
        candidates.append(
            SelectedVisualReference(
                reference=to_style_image_reference(style_reference),
                semantic_role="STYLE",
                reason="permanent project style and detail ceiling",
            )
        )
    if operation is not VisualOperation.NEW_IMAGE and master_asset is not None:
        candidates.append(
            SelectedVisualReference(
                reference=ImageReference(
                    reference_id=master_asset.master_scene_id,
                    file_path=master_asset.file_path,
                    sha256=master_asset.file_sha256,
                    role=ImageReferenceRole.CONTENT_CONTINUITY,
                ),
                semantic_role="MASTER_LOCATION",
                reason="preserve recurring environment geometry",
            )
        )
    if operation is not VisualOperation.NEW_IMAGE and source is not None:
        candidates.append(
            SelectedVisualReference(
                reference=ImageReference(
                    reference_id=source.asset_id,
                    file_path=source.path,
                    sha256=source.sha256,
                    role=ImageReferenceRole.CONTENT_CONTINUITY,
                ),
                semantic_role="PREVIOUS_STATE",
                reason="closest compatible accepted physical state",
            )
        )

    unique: list[SelectedVisualReference] = []
    hashes: set[str] = set()
    for candidate in candidates:
        if candidate.reference.sha256 in hashes:
            continue
        hashes.add(candidate.reference.sha256)
        unique.append(candidate)

    maximum = capabilities.max_reference_images
    if len(unique) > maximum:
        required_hash = source.sha256 if source is not None else None
        selected = unique[:maximum]
        if (
            operation is VisualOperation.EDIT_EXISTING
            and required_hash is not None
            and all(item.reference.sha256 != required_hash for item in selected)
        ):
            source_reference = next(
                item for item in unique if item.reference.sha256 == required_hash
            )
            selected[-1] = source_reference
        unique = selected

    for selected in unique:
        logger.info(
            "Visual reference selected",
            extra={
                "beat_id": beat.id,
                "reference_role": selected.semantic_role,
                "reference_asset_id": selected.reference.reference_id,
                "selection_reason": selected.reason,
            },
        )
    return tuple(unique)


def _score_result_source(
    beat: VisualBeat,
    result: BeatVisualResult,
    explicit_sources: set[str],
    resolved_operation: str,
) -> tuple[float, list[str]]:
    state = result.semantic_state_snapshot
    score = 0.0
    reasons: list[str] = []
    if result.beat_id in explicit_sources:
        score += 24
        reasons.append("explicit semantic source")
    if state.get("location_id") == beat.location_id:
        score += 10
        reasons.append("same location")
    if state.get("master_scene_id") == beat.master_scene_id and beat.master_scene_id:
        score += 12
        reasons.append("same master scene")
    score += 4 * _jaccard(
        state.get("characters_visible", ()),
        beat.characters_visible,
    )
    score += 4 * _jaccard(
        state.get("important_objects", ()),
        beat.important_objects,
    )
    if state.get("camera_framing") == beat.camera_framing.value:
        score += 4
        reasons.append("compatible framing")
    previous_state = (
        beat.progressive_change.previous_state
        if beat.progressive_change is not None
        else None
    )
    candidate_physical_state = str(state.get("physical_state", ""))
    if previous_state and previous_state.lower() in candidate_physical_state.lower():
        score += 8
        reasons.append("closest prior progressive state")
    score += 5 * SequenceMatcher(
        None,
        candidate_physical_state.lower(),
        beat.physical_state.lower(),
    ).ratio()
    if resolved_operation == VisualOperation.EDIT_EXISTING.value:
        score += 2
    return score, reasons or ["highest semantic continuity score"]


def _jaccard(left: object, right: object) -> float:
    left_set = set(left) if isinstance(left, (list, tuple, set)) else set()
    right_set = set(right) if isinstance(right, (list, tuple, set)) else set()
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
