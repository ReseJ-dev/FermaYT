"""Score and validate how each semantic visual beat should be produced."""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

from app.models.visual_plan import (
    CameraMovement,
    ShotFraming,
    VisualBeat,
    VisualOperation,
    VisualPlan,
)
from app.provider_capabilities import ImageProviderCapabilities

logger = logging.getLogger(__name__)


VisualProviderCapabilities = ImageProviderCapabilities


@dataclass(frozen=True, slots=True)
class VisualDecisionEvidence:
    """Optional evidence from later LLM/vision analysis, bounded to safe values."""

    physical_state_changed: bool | None = None
    visual_similarity: float | None = None
    substantially_different_composition: bool | None = None
    overlay_sufficient: bool | None = None
    new_image_improves_understanding: bool | None = None

    def __post_init__(self) -> None:
        if self.visual_similarity is not None and not 0 <= self.visual_similarity <= 1:
            raise ValueError("visual_similarity must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class VisualOperationDecision:
    beat_id: str
    operation: VisualOperation
    requested_operation: VisualOperation
    scores: Mapping[VisualOperation, float]
    reasons: tuple[str, ...]
    source_visual_ids: tuple[str, ...] = ()
    source_image_paths: tuple[str, ...] = ()
    fallback_from: VisualOperation | None = None

    @property
    def requires_image_api(self) -> bool:
        return self.operation in {
            VisualOperation.NEW_IMAGE,
            VisualOperation.REFERENCE_GENERATION,
            VisualOperation.EDIT_EXISTING,
        }


@dataclass(slots=True)
class _DecisionContext:
    beat: VisualBeat
    previous: VisualBeat | None
    references: tuple[tuple[str, str], ...]
    capabilities: VisualProviderCapabilities
    evidence: VisualDecisionEvidence
    same_location: bool
    character_continuity: float
    object_continuity: float
    visual_similarity: float
    physical_state_changed: bool
    camera_changed: bool
    substantially_different_composition: bool
    overlay_sufficient: bool
    new_image_improves_understanding: bool
    reasons: dict[VisualOperation, list[str]] = field(default_factory=dict)


class VisualOperationDecisionEngine:
    """Choose the clearest continuity-aware operation for one visual beat."""

    def decide(
        self,
        plan: VisualPlan,
        beat_index: int,
        *,
        capabilities: VisualProviderCapabilities,
        available_visuals: Mapping[str, str],
        evidence: VisualDecisionEvidence | None = None,
    ) -> VisualOperationDecision:
        if beat_index < 0 or beat_index >= len(plan.visual_beats):
            raise IndexError("beat_index is outside the visual plan")

        beat = plan.visual_beats[beat_index]
        previous = plan.visual_beats[beat_index - 1] if beat_index > 0 else None
        decision_evidence = evidence or VisualDecisionEvidence()
        references = self._find_references(plan, beat_index, available_visuals)
        context = self._build_context(
            beat,
            previous,
            references,
            capabilities,
            decision_evidence,
        )
        scores = self._score(context)
        selected = max(scores, key=scores.__getitem__)
        fallback_from: VisualOperation | None = None

        if not self._is_valid(selected, context):
            fallback_from = selected
            selected = self._fallback(selected, context)

        # Explicit fallback contracts take precedence over score-based guesses.
        if (
            beat.preferred_visual_operation is VisualOperation.EDIT_EXISTING
            and (not capabilities.image_editing or not references)
        ):
            fallback_from = VisualOperation.EDIT_EXISTING
            selected = (
                VisualOperation.REFERENCE_GENERATION
                if capabilities.reference_generation and references
                else VisualOperation.NEW_IMAGE
            )
        elif (
            beat.preferred_visual_operation
            is VisualOperation.REFERENCE_GENERATION
            and (not capabilities.reference_generation or not references)
        ):
            fallback_from = VisualOperation.REFERENCE_GENERATION
            selected = VisualOperation.NEW_IMAGE

        reasons = self._decision_reasons(context, selected, fallback_from)
        source_visual_ids, source_image_paths = self._decision_references(
            selected,
            references,
            capabilities,
        )
        decision = VisualOperationDecision(
            beat_id=beat.id,
            operation=selected,
            requested_operation=beat.preferred_visual_operation,
            scores=dict(scores),
            reasons=tuple(reasons),
            source_visual_ids=source_visual_ids,
            source_image_paths=source_image_paths,
            fallback_from=fallback_from,
        )
        self._log(decision)
        return decision

    def _build_context(
        self,
        beat: VisualBeat,
        previous: VisualBeat | None,
        references: tuple[tuple[str, str], ...],
        capabilities: VisualProviderCapabilities,
        evidence: VisualDecisionEvidence,
    ) -> _DecisionContext:
        same_location = previous is not None and beat.location_id == previous.location_id
        character_continuity = _jaccard(
            beat.characters_visible,
            previous.characters_visible if previous else (),
        )
        object_continuity = _jaccard(
            beat.important_objects,
            previous.important_objects if previous else (),
        )
        inferred_similarity = (
            (0.45 if same_location else 0)
            + 0.3 * character_continuity
            + 0.25 * object_continuity
        )
        visual_similarity = (
            evidence.visual_similarity
            if evidence.visual_similarity is not None
            else inferred_similarity
        )
        physical_state_changed = (
            evidence.physical_state_changed
            if evidence.physical_state_changed is not None
            else beat.progressive_change is not None
            or beat.preferred_visual_operation is VisualOperation.EDIT_EXISTING
        )
        camera_changed = previous is not None and (
            beat.camera_framing != previous.camera_framing
            or beat.camera_movement != CameraMovement.STATIC
            or beat.camera_view != previous.camera_view
        )
        substantially_different = (
            evidence.substantially_different_composition
            if evidence.substantially_different_composition is not None
            else not same_location
            or (camera_changed and visual_similarity < 0.45)
        )
        overlay_sufficient = (
            evidence.overlay_sufficient
            if evidence.overlay_sufficient is not None
            else beat.preferred_visual_operation is VisualOperation.OVERLAY
            or beat.overlay_description is not None
        )
        improves_understanding = (
            evidence.new_image_improves_understanding
            if evidence.new_image_improves_understanding is not None
            else previous is None
            or not same_location
            or substantially_different
        )
        return _DecisionContext(
            beat=beat,
            previous=previous,
            references=references,
            capabilities=capabilities,
            evidence=evidence,
            same_location=same_location,
            character_continuity=character_continuity,
            object_continuity=object_continuity,
            visual_similarity=visual_similarity,
            physical_state_changed=physical_state_changed,
            camera_changed=camera_changed,
            substantially_different_composition=substantially_different,
            overlay_sufficient=overlay_sufficient,
            new_image_improves_understanding=improves_understanding,
        )

    def _score(self, context: _DecisionContext) -> dict[VisualOperation, float]:
        scores = {operation: 0.0 for operation in VisualOperation}
        beat = context.beat

        # LLM guidance is useful evidence, but it cannot bypass capability validation.
        scores[beat.preferred_visual_operation] += 1.25
        self._reason(context, beat.preferred_visual_operation, "Visual Director guidance")

        if context.previous is None or not context.same_location:
            scores[VisualOperation.NEW_IMAGE] += 5
            self._reason(context, VisualOperation.NEW_IMAGE, "new location or first establishment")
        if context.substantially_different_composition:
            scores[VisualOperation.NEW_IMAGE] += 2
            scores[VisualOperation.REFERENCE_GENERATION] += 2
            scores[VisualOperation.REUSE] -= 3
            scores[VisualOperation.TRANSFORM] -= 3
            scores[VisualOperation.OVERLAY] -= 2
            self._reason(context, VisualOperation.NEW_IMAGE, "substantially different composition")
            self._reason(
                context,
                VisualOperation.REFERENCE_GENERATION,
                "new composition with continuity requirements",
            )
        if context.new_image_improves_understanding:
            # Clarity outranks call count. This is intentionally much larger than
            # the small generation-cost tie-breakers below.
            scores[VisualOperation.NEW_IMAGE] += 6
            scores[VisualOperation.REFERENCE_GENERATION] += 6
            scores[VisualOperation.REUSE] -= 2
            scores[VisualOperation.TRANSFORM] -= 2
            scores[VisualOperation.OVERLAY] -= 2
            clarity_reason = "materially clearer visual outweighs generation cost"
            self._reason(context, VisualOperation.NEW_IMAGE, clarity_reason)
            self._reason(
                context,
                VisualOperation.REFERENCE_GENERATION,
                clarity_reason,
            )

        if context.references:
            scores[VisualOperation.REFERENCE_GENERATION] += 2.5
            self._reason(context, VisualOperation.REFERENCE_GENERATION, "reference image available")
        if context.same_location:
            scores[VisualOperation.EDIT_EXISTING] += 2
            scores[VisualOperation.REUSE] += 1.5
            scores[VisualOperation.TRANSFORM] += 1.5
            scores[VisualOperation.OVERLAY] += 1
        if context.visual_similarity >= 0.65:
            scores[VisualOperation.EDIT_EXISTING] += 1.5
            scores[VisualOperation.REUSE] += 2
            scores[VisualOperation.TRANSFORM] += 2
            self._reason(context, VisualOperation.REUSE, "high visual continuity")

        if context.physical_state_changed:
            scores[VisualOperation.EDIT_EXISTING] += 5
            scores[VisualOperation.REUSE] -= 5
            scores[VisualOperation.TRANSFORM] -= 3
            self._reason(context, VisualOperation.EDIT_EXISTING, "physical state changed")
        else:
            scores[VisualOperation.REUSE] += 3
            self._reason(context, VisualOperation.REUSE, "physical state did not materially change")

        if context.camera_changed and not context.physical_state_changed:
            scores[VisualOperation.TRANSFORM] += 4
            self._reason(context, VisualOperation.TRANSFORM, "camera changes but scene state does not")
        if context.overlay_sufficient and not context.physical_state_changed:
            scores[VisualOperation.OVERLAY] += 6
            self._reason(context, VisualOperation.OVERLAY, "new information fits a programmatic overlay")

        if beat.camera_framing is ShotFraming.WIDE and context.previous is None:
            scores[VisualOperation.NEW_IMAGE] += 1
        if context.character_continuity >= 0.5 or context.object_continuity >= 0.5:
            scores[VisualOperation.REFERENCE_GENERATION] += 1
            self._reason(
                context,
                VisualOperation.REFERENCE_GENERATION,
                "character or object identity should remain consistent",
            )

        # Cost is deliberately a small tie-breaker, never the primary objective.
        scores[VisualOperation.NEW_IMAGE] -= 0.8
        scores[VisualOperation.REFERENCE_GENERATION] -= 0.6
        scores[VisualOperation.EDIT_EXISTING] -= 0.5

        if not context.capabilities.reference_generation or not context.references:
            scores[VisualOperation.REFERENCE_GENERATION] = float("-inf")
        if not context.capabilities.image_editing or not context.references:
            scores[VisualOperation.EDIT_EXISTING] = float("-inf")
        for operation in {
            VisualOperation.REUSE,
            VisualOperation.TRANSFORM,
            VisualOperation.OVERLAY,
        }:
            if not context.references:
                scores[operation] = float("-inf")
        return scores

    @staticmethod
    def _find_references(
        plan: VisualPlan,
        beat_index: int,
        available_visuals: Mapping[str, str],
    ) -> tuple[tuple[str, str], ...]:
        beat = plan.visual_beats[beat_index]
        candidate_ids: list[str] = []
        if beat.source_visual_id is not None:
            candidate_ids.append(beat.source_visual_id)
        if beat.geography_established_by is not None:
            candidate_ids.append(beat.geography_established_by)
        if beat_index > 0:
            candidate_ids.append(plan.visual_beats[beat_index - 1].id)
        candidate_ids.extend(
            master.id
            for master in plan.possible_master_scenes
            if master.location_id == beat.location_id
        )

        references: list[tuple[str, str]] = []
        seen: set[str] = set()
        for visual_id in candidate_ids:
            path = available_visuals.get(visual_id)
            if path and visual_id not in seen:
                references.append((visual_id, path))
                seen.add(visual_id)
        return tuple(references)

    @staticmethod
    def _is_valid(operation: VisualOperation, context: _DecisionContext) -> bool:
        if operation is VisualOperation.NEW_IMAGE:
            return True
        if not context.references:
            return False
        if operation is VisualOperation.REFERENCE_GENERATION:
            return context.capabilities.reference_generation
        if operation is VisualOperation.EDIT_EXISTING:
            return context.capabilities.image_editing
        return True

    @staticmethod
    def _fallback(
        operation: VisualOperation,
        context: _DecisionContext,
    ) -> VisualOperation:
        if context.references and context.capabilities.reference_generation:
            return VisualOperation.REFERENCE_GENERATION
        return VisualOperation.NEW_IMAGE

    @staticmethod
    def _decision_references(
        operation: VisualOperation,
        references: tuple[tuple[str, str], ...],
        capabilities: VisualProviderCapabilities,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if operation is VisualOperation.NEW_IMAGE:
            return (), ()
        selected_references = references
        if operation in {
            VisualOperation.REFERENCE_GENERATION,
            VisualOperation.EDIT_EXISTING,
        }:
            selected_references = references[: capabilities.max_reference_images]
        return (
            tuple(visual_id for visual_id, _ in selected_references),
            tuple(path for _, path in selected_references),
        )

    @staticmethod
    def _reason(
        context: _DecisionContext,
        operation: VisualOperation,
        reason: str,
    ) -> None:
        context.reasons.setdefault(operation, []).append(reason)

    @staticmethod
    def _decision_reasons(
        context: _DecisionContext,
        selected: VisualOperation,
        fallback_from: VisualOperation | None,
    ) -> list[str]:
        reasons = list(context.reasons.get(selected, ()))
        if context.same_location:
            reasons.append("same location")
        if context.character_continuity >= 0.5:
            reasons.append("recurring characters remain visible")
        if context.object_continuity >= 0.5:
            reasons.append("important objects remain continuous")
        if fallback_from is not None:
            if (
                fallback_from is VisualOperation.EDIT_EXISTING
                and not context.capabilities.image_editing
            ):
                reasons.append("provider does not support editing")
            elif (
                fallback_from is VisualOperation.REFERENCE_GENERATION
                and not context.capabilities.reference_generation
            ):
                reasons.append("provider does not support reference generation")
            elif not context.references:
                reasons.append(
                    f"no usable source visual for {fallback_from.value}"
                )
            else:
                reasons.append(f"fallback from {fallback_from.value}")
        return list(dict.fromkeys(reasons)) or ["highest validated continuity score"]

    @staticmethod
    def _log(decision: VisualOperationDecision) -> None:
        logger.info(
            "Beat %s:\n%s\nreason:\n%s",
            decision.beat_id,
            decision.operation.value,
            "\n".join(decision.reasons),
        )


def _jaccard(left: Collection[str], right: Collection[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union)
