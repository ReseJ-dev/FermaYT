"""Whole-story semantic visual planning before asset generation."""

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import ValidationError

from app.budgets import GenerationBudgetError
from app.errors import (
    PlanningBillingUncertainError,
    PlanningTooLargeError,
    StructuredAIProviderError,
    VisualDirectorError,
)
from app.models.visual_plan import (
    VisualPlan,
    VisualPlanDuplicateIdError,
    VisualPlanMasterSceneAssignmentError,
    VisualPlanReferenceError,
)
from app.provider_diagnostics import (
    StructuredAIProviderDiagnostic,
    find_structured_ai_provider_diagnostic,
)

MAX_VISUAL_PLAN_REPAIR_ATTEMPTS = 2


class VisualPlanPacingError(ValueError):
    """The plan has too few meaningful visible states for its narration."""

    category = "INSUFFICIENT_VISUAL_PACING"

    def __init__(self, actual: int, minimum: int) -> None:
        self.actual = actual
        self.minimum = minimum
        super().__init__(
            f"visual plan has {actual} beats; narration requires at least {minimum} "
            "meaningful visible states"
        )


@dataclass(frozen=True, slots=True)
class VisualPlanValidationDiagnostic:
    category: str
    summary: str
    issue: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class VisualPlanRepairEvent:
    repair_attempt: int
    category: str
    succeeded: bool
    issue: dict[str, object] | None = None


class VisualPlanningClient(Protocol):
    """Minimal boundary for a future structured-output LLM provider."""

    async def generate(self, prompt: str) -> str: ...


class PlanningAttemptExecutor(Protocol):
    @property
    def can_dispatch(self) -> bool: ...

    async def execute(self, prompt: str, kind: str) -> str: ...

    def mark_validation_started(self) -> None: ...

    def mark_validation(
        self,
        status: str,
        safe_error: str | None = None,
        validation_category: str | None = None,
    ) -> None: ...


class VisualDirector:
    """Turn complete narration into a validated semantic visual plan."""

    def __init__(
        self,
        client: VisualPlanningClient,
        *,
        max_repair_attempts: int = MAX_VISUAL_PLAN_REPAIR_ATTEMPTS,
        on_repair_event: Callable[[VisualPlanRepairEvent], None] | None = None,
        before_provider_call: Callable[[int, bool], None] | None = None,
        attempt_controller: PlanningAttemptExecutor | None = None,
    ) -> None:
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must not be negative")
        self._client = client
        self.max_repair_attempts = max_repair_attempts
        self._on_repair_event = on_repair_event
        self._before_provider_call = before_provider_call
        self._attempt_controller = attempt_controller
        self.provider_requests = 0

    async def create_plan(
        self,
        narration: str,
        *,
        planning_scope_metadata: dict[str, object] | None = None,
        project_title: str | None = None,
    ) -> VisualPlan:
        normalized_narration = narration.strip()
        if not normalized_narration:
            raise ValueError("Narration must not be empty")

        raw_plan = await self._generate(
            build_visual_director_request(
                normalized_narration,
                planning_scope_metadata=planning_scope_metadata,
                project_title=project_title,
            ),
            is_repair=False,
        )
        last_diagnostic: VisualPlanValidationDiagnostic | None = None
        for repair_attempt in range(self.max_repair_attempts + 1):
            if self._attempt_controller is not None:
                self._attempt_controller.mark_validation_started()
            try:
                payload = json.loads(raw_plan)
                if isinstance(payload, dict) and planning_scope_metadata is not None:
                    payload = dict(payload)
                    payload["planning_scope"] = planning_scope_metadata
                payload = hydrate_resolvable_master_scene_ids(payload)
                plan = VisualPlan.model_validate(payload)
                _validate_visual_plan_pacing(plan, normalized_narration)
            except (
                json.JSONDecodeError,
                TypeError,
                ValidationError,
                VisualPlanPacingError,
            ) as exc:
                last_diagnostic = _validation_diagnostic(exc)
                can_repair = repair_attempt < self.max_repair_attempts and (
                    self._attempt_controller is None
                    or self._attempt_controller.can_dispatch
                )
                if self._attempt_controller is not None:
                    self._attempt_controller.mark_validation(
                        "REPAIR_REQUIRED" if can_repair else "SCHEMA_INVALID",
                        last_diagnostic.summary,
                        last_diagnostic.category,
                    )
                self._emit_repair_event(
                    VisualPlanRepairEvent(
                        repair_attempt=repair_attempt,
                        category=last_diagnostic.category,
                        succeeded=False,
                        issue=last_diagnostic.issue,
                    )
                )
                if not can_repair:
                    stable_category = _stable_validation_category(
                        last_diagnostic.category
                    )
                    provider_diagnostic = StructuredAIProviderDiagnostic(
                        provider=str(getattr(self._client, "provider", "unknown")),
                        model=getattr(self._client, "model", None),
                        operation="visual_planning",
                        category="PLANNING_REPAIR_EXHAUSTED",
                        attempt=self.provider_requests,
                        max_attempts=self.provider_requests,
                        repair_attempt=repair_attempt,
                        provider_error=last_diagnostic.summary,
                        response_length=len(raw_plan),
                        validation_category=stable_category,
                    )
                    raise VisualDirectorError(
                        "Visual planning provider returned an invalid structured "
                        f"visual plan after {repair_attempt} repair attempt(s): "
                        f"{last_diagnostic.summary}",
                        validation_category=last_diagnostic.category,
                        diagnostic=last_diagnostic.issue,
                        provider_requests=self.provider_requests,
                        safe_diagnostic=provider_diagnostic,
                    ) from exc
                repair_prompt = build_visual_plan_repair_request(
                    raw_plan,
                    exc,
                    last_diagnostic,
                    planning_scope_metadata=planning_scope_metadata,
                )
                raw_plan = await self._generate(repair_prompt, is_repair=True)
                continue

            if repair_attempt > 0 and last_diagnostic is not None:
                self._emit_repair_event(
                    VisualPlanRepairEvent(
                        repair_attempt=repair_attempt,
                        category=last_diagnostic.category,
                        succeeded=True,
                        issue=last_diagnostic.issue,
                    )
                )
            if self._attempt_controller is not None:
                self._attempt_controller.mark_validation("SUCCEEDED")
            return plan

        raise AssertionError("visual plan validation loop did not terminate")

    async def _generate(self, prompt: str, *, is_repair: bool) -> str:
        next_request = self.provider_requests + 1
        if self._before_provider_call is not None:
            self._before_provider_call(next_request, is_repair)
        self.provider_requests = next_request
        try:
            if self._attempt_controller is not None:
                return await self._attempt_controller.execute(
                    prompt,
                    "REPAIR" if is_repair else "INITIAL",
                )
            return await self._client.generate(prompt)
        except (
            GenerationBudgetError,
            PlanningBillingUncertainError,
            PlanningTooLargeError,
        ):
            raise
        except StructuredAIProviderError as exc:
            message = (
                f"Visual plan repair provider failed on request {next_request}"
                if is_repair
                else "Visual planning provider failed"
            )
            diagnostic = find_structured_ai_provider_diagnostic(exc)
            if diagnostic is not None:
                diagnostic = diagnostic.with_context(
                    repair_attempt=max(next_request - 1, 0)
                )
            raise VisualDirectorError(
                message,
                validation_category=(
                    diagnostic.category
                    if diagnostic is not None
                    else "PLANNING_UNKNOWN_ERROR"
                ),
                provider_requests=self.provider_requests,
                safe_diagnostic=diagnostic,
            ) from exc
        except Exception as exc:
            message = (
                f"Visual plan repair provider failed on request {next_request}"
                if is_repair
                else "Visual planning provider failed"
            )
            diagnostic = StructuredAIProviderDiagnostic(
                provider=str(getattr(self._client, "provider", "unknown")),
                model=getattr(self._client, "model", None),
                operation="visual_planning",
                category="PLANNING_UNKNOWN_ERROR",
                attempt=next_request,
                max_attempts=next_request,
                provider_error=type(exc).__name__,
                repair_attempt=max(next_request - 1, 0),
            )
            raise VisualDirectorError(
                message,
                validation_category="PLANNING_UNKNOWN_ERROR",
                provider_requests=self.provider_requests,
                safe_diagnostic=diagnostic,
            ) from exc

    def _emit_repair_event(self, event: VisualPlanRepairEvent) -> None:
        if self._on_repair_event is not None:
            self._on_repair_event(event)


def _validation_summary(error: Exception) -> str:
    """Return useful schema diagnostics without including provider-generated values."""
    return _validation_diagnostic(error).summary


def _validation_diagnostic(error: Exception) -> VisualPlanValidationDiagnostic:
    """Classify a parse/schema/reference error using the canonical validator context."""
    if isinstance(error, VisualPlanPacingError):
        return VisualPlanValidationDiagnostic(
            category=error.category,
            summary=str(error),
            issue={
                "category": error.category,
                "actual_visible_states": error.actual,
                "minimum_visible_states": error.minimum,
            },
        )
    if isinstance(error, ValidationError):
        details: list[str] = []
        errors = error.errors(
            include_url=False,
            include_context=True,
            include_input=False,
        )
        for item in errors[:3]:
            location = ".".join(str(part) for part in item["loc"]) or "visual_plan"
            details.append(f"{location}: {item['msg']}")
        remaining = error.error_count() - len(details)
        if remaining > 0:
            details.append(f"and {remaining} more validation error(s)")
        for item in errors:
            context_error = item.get("ctx", {}).get("error")
            if isinstance(context_error, VisualPlanReferenceError):
                return VisualPlanValidationDiagnostic(
                    category=context_error.category,
                    summary="; ".join(details),
                    issue=context_error.as_dict(),
                )
            if isinstance(context_error, VisualPlanMasterSceneAssignmentError):
                return VisualPlanValidationDiagnostic(
                    category=context_error.category,
                    summary="; ".join(details),
                    issue=context_error.as_dict(),
                )
            if isinstance(context_error, VisualPlanDuplicateIdError):
                return VisualPlanValidationDiagnostic(
                    category="DUPLICATE_ID",
                    summary="; ".join(details),
                    issue={
                        "category": "DUPLICATE_ID",
                        "registry": context_error.label,
                        "duplicate_id": context_error.duplicate_id,
                    },
                )
        return VisualPlanValidationDiagnostic(
            category="SCHEMA_VALIDATION_ERROR",
            summary="; ".join(details),
        )
    if isinstance(error, json.JSONDecodeError):
        return VisualPlanValidationDiagnostic(
            category="JSON_PARSE_ERROR",
            summary=f"invalid JSON at line {error.lineno}, column {error.colno}",
        )
    return VisualPlanValidationDiagnostic(
        category="OTHER_VISUAL_PLAN_ERROR",
        summary="response has an unsupported JSON shape",
    )


def _validate_visual_plan_pacing(plan: VisualPlan, narration: str) -> None:
    word_count = len(re.findall(r"\b[\w'-]+\b", narration, flags=re.UNICODE))
    # At a typical explanatory narration pace this targets roughly one visible
    # state per 10-11 seconds, yielding about 45-70 states for an eight-minute video.
    minimum_states = max(1, math.ceil(word_count / 21))
    if len(plan.visual_beats) < minimum_states:
        raise VisualPlanPacingError(len(plan.visual_beats), minimum_states)


def hydrate_resolvable_master_scene_ids(payload: object) -> object:
    """Fill beat master references only when the location has one clear master."""
    if not isinstance(payload, dict):
        return payload
    environments = payload.get("recurring_environments")
    masters = payload.get("possible_master_scenes")
    beats = payload.get("visual_beats")
    if not all(isinstance(items, list) for items in (environments, masters, beats)):
        return payload

    recurring_locations = {
        item.get("location_id")
        for item in environments
        if isinstance(item, dict)
        and isinstance(item.get("id"), str)
        and isinstance(item.get("location_id"), str)
    }
    masters_by_location: dict[str, list[str]] = {}
    for item in masters:
        if not isinstance(item, dict):
            continue
        master_id = item.get("id")
        location_id = item.get("location_id")
        if isinstance(master_id, str) and master_id and isinstance(location_id, str):
            masters_by_location.setdefault(location_id, []).append(master_id)

    for beat in beats:
        if not isinstance(beat, dict):
            continue
        location_id = beat.get("location_id")
        candidates = masters_by_location.get(location_id, [])
        if location_id in recurring_locations and len(candidates) == 1:
            beat["master_scene_id"] = candidates[0]
    return payload


def _stable_validation_category(category: str) -> str:
    if category in {
        "UNKNOWN_REFERENCE",
        "FORWARD_REFERENCE",
        "REFERENCE_TYPE_MISMATCH",
        VisualPlanMasterSceneAssignmentError.category,
        "DUPLICATE_ID",
    }:
        return "PLANNING_REFERENCE_ERROR"
    if category == "JSON_PARSE_ERROR":
        return "PLANNING_INVALID_JSON"
    return "PLANNING_SCHEMA_ERROR"


def build_visual_plan_repair_request(
    raw_plan: str,
    error: Exception,
    diagnostic: VisualPlanValidationDiagnostic,
    *,
    planning_scope_metadata: dict[str, object] | None = None,
) -> str:
    """Build a targeted request that preserves content and repairs structure only."""
    validation_errors = _repair_error_payload(error, diagnostic)
    master_scene_context = _master_scene_repair_context(diagnostic)
    schema = json.dumps(VisualPlan.model_json_schema(), ensure_ascii=False)
    scope_context = ""
    if planning_scope_metadata and planning_scope_metadata.get("is_partial"):
        scope_context = f"""PARTIAL PLANNING SCOPE:
Repair only the plan for source characters
{planning_scope_metadata.get("source_start_char")} through
{planning_scope_metadata.get("source_end_char")}.
Do not add beats for narration outside that excerpt.
"""
    return f"""The previous VisualPlan is structurally invalid.

{scope_context}

Repair ONLY the structural consistency of the same JSON. Do not rewrite the story.
Preserve story meaning, beat count and order, narration mapping, locations,
characters, objects, physical progression, operations, and visual strategy unless a
specific validation error requires a minimal change.

VALIDATION CATEGORY: {diagnostic.category}
VALIDATION ERRORS:
{json.dumps(validation_errors, ensure_ascii=False)}

{master_scene_context}

ID REFERENCE RULES:
- Every referenced ID must exactly match an item defined in the same VisualPlan.
- Never invent a reference ID inside a VisualBeat and never put descriptive prose in
  an ID field. Reuse canonical IDs instead of making near-duplicates.
- geography_established_by may reference only a possible_master_scenes ID or an
  EARLIER visual_beats ID in the same location. A current, future, or undeclared beat
  ID is invalid. If geography is optional and no valid visual applies, use null.
- location_id, master_scene_id, character IDs, important-object IDs,
  source_visual_id, and progressive-change subject_id must use their corresponding
  declared registries and obey the supplied schema.
- Do not create empty placeholder definitions. Add a definition only when it is
  genuinely required and can be defined meaningfully from the existing plan.
- Before returning JSON, verify every *_id reference against definitions in this
  output and verify that every source/geography beat reference points backward.

Return the complete corrected JSON object only. Return no markdown or commentary.

JSON SCHEMA:
{schema}

INVALID JSON TO REPAIR:
<visual_plan>
{raw_plan}
</visual_plan>
"""


def _master_scene_repair_context(
    diagnostic: VisualPlanValidationDiagnostic,
) -> str:
    issue = diagnostic.issue
    if (
        diagnostic.category != VisualPlanMasterSceneAssignmentError.category
        or not isinstance(issue, dict)
    ):
        return ""
    environments = issue.get("recurring_environment_ids", [])
    allowed = issue.get("allowed_master_scene_ids", [])
    environment_lines = "\n".join(f"- {value}" for value in environments)
    allowed_lines = "\n".join(f"- {value}" for value in allowed)
    provided = json.dumps(issue.get("provided_master_scene_id"), ensure_ascii=False)
    return f"""MASTER SCENE REFERENCE VIOLATION:
Affected beat id: {issue.get("beat_id")}
Recurring environment id(s):
{environment_lines}
Currently supplied master_scene_id: {provided}
Allowed master_scene_id values:
{allowed_lines}
Exact invariant: {issue.get("invariant")}
The beat must reference one of the allowed master scenes above."""


def _repair_error_payload(
    error: Exception,
    diagnostic: VisualPlanValidationDiagnostic,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "category": diagnostic.category,
        "summary": diagnostic.summary,
    }
    if diagnostic.issue is not None:
        result["issue"] = diagnostic.issue
    if isinstance(error, ValidationError):
        result["errors"] = [
            {
                "location": [str(part) for part in item["loc"]],
                "type": item["type"],
                "message": item["msg"],
            }
            for item in error.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            )[:10]
        ]
    return result


def build_visual_director_request(
    narration: str,
    *,
    planning_scope_metadata: dict[str, object] | None = None,
    project_title: str | None = None,
) -> str:
    """Build a provider-neutral request for a full or bounded source range."""
    schema = json.dumps(VisualPlan.model_json_schema(), ensure_ascii=False)
    is_partial = bool(
        planning_scope_metadata and planning_scope_metadata.get("is_partial")
    )
    if is_partial:
        scope_directive = f"""PLANNING SCOPE.
You are planning ONLY this narration excerpt.
Do not invent beats for narration outside this excerpt.
Return a VisualPlan only for the supplied source range.
This is a partial plan and must not imply that it covers the complete story.
Source range: characters {planning_scope_metadata.get("source_start_char")}
through {planning_scope_metadata.get("source_end_char")}.
Requested scope: {planning_scope_metadata.get("scope_type")}.
Expected beat count is approximately
{planning_scope_metadata.get("expected_beat_count")}; use story-driven pacing.
The end of the excerpt contains a small deterministic lookahead for continuity.
Project title: {project_title or "Untitled project"}.
"""
        reading_instruction = (
            "Read ONLY the supplied narration excerpt before planning any beat."
        )
        pacing_instruction = (
            "12. PACING GUARD. Plan only this bounded excerpt. No completely "
            "unchanged static frame may last more than 12 seconds. Do not use "
            "whole-story or eight-minute beat targets."
        )
        narration_label = "SCOPED NARRATION EXCERPT"
    else:
        scope_directive = ""
        reading_instruction = (
            "Read the COMPLETE narration before planning any individual beat."
        )
        pacing_instruction = (
            "12. PACING GUARD. No completely unchanged static frame may last more "
            "than 12 seconds. For narration around eight minutes, create roughly "
            "45-70 meaningful visible states, while keeping paid NEW_IMAGE, "
            "REFERENCE_GENERATION, and EDIT_EXISTING operations near 25-40. Use "
            "REUSE with a new crop, zoom, pan, focus, or deterministic OVERLAY for "
            "the remaining states; do not buy one image per sentence."
        )
        narration_label = "COMPLETE NARRATION"
    return f"""You are the Visual Director for a coherent narrated video.

{scope_directive}

VISUAL PRODUCT PRINCIPLE. This is an automated visual storytelling system, not an
AI image generator with video export. Optimize for the experience of watching the
finished sequence. Apply this priority order to every planning trade-off:
1. storytelling clarity
2. spatial continuity
3. visual progression
4. readability
5. consistent style
6. generation efficiency
7. image beauty

A simple image that clearly advances the story is better than a beautiful image
that only decorates the narration. Do not optimize only for fewer API calls or only
for visual variety.

VISUAL SIMPLIFICATION. Your job is NOT to visualize everything mentioned in the
narration. Understand the complete semantic state, then choose the minimum visible
representation required for immediate understanding. Before every beat ask, "What can
be removed while preserving the meaning?" Never ask what else could be included.
One frame has exactly one dominant main_visual_idea. Prefer one main subject, zero to
two supporting objects, one simple environment concept, one main action/state, and one
representative character. Raise those soft limits only when the story-critical meaning
genuinely requires it. Fewer objects and fewer characters are better. Keep backgrounds
sparse, avoid decorative realism, and do not expand a scene for beauty.

Keep FULL STORY STATE separate from VISIBLE ELEMENTS FOR THIS BEAT. physical_state,
location definitions, continuity and master scenes may retain the complete truth.
main_visual_idea, visible_physical_state, characters_visible, important_objects and
essential_environment_cues are only the subset that must be drawn now. Use one minimal
environment cue when possible. Master-scene inventories preserve continuity; they are
not a checklist of objects to copy into every frame. Put known but unnecessary entities
in optional_entities_to_omit. If narration says several people but their exact number
is irrelevant, use one representative character and character_count_target=1. Preserve
story-critical identity and state.

Choose the simplest explanatory framing. Prefer WIDE, MEDIUM or CLOSE for direct facts,
and CUTAWAY_DIAGRAM for mechanisms or spatial relationships. A side view may be stated
in camera_view. Avoid complex perspective unless meaning requires it. When two
independent important visual facts compete for one frame, split them into adjacent
beats and explain the separation in split_reason. Splitting semantic beats does not
require another paid image: prefer REUSE, TRANSFORM, OVERLAY, a different crop/focus,
or a simple EDIT_EXISTING when the established asset can communicate the next fact.

{reading_instruction} Narration and
visuals have different jobs: do not paraphrase narration and do not write image
generation prompts. First identify all characters, locations, important objects,
recurring environments, and possible master scenes. Then create meaningful visual
beats that make spatial layout, movement, distance, routes, obstructions, danger,
breakage, and progressive physical changes easy to understand.

MASTER SCENES. Select only a small number of important recurring environments. Define
all possible_master_scenes first, with stable unique IDs, before creating visual_beats.
For each master, define stable environment geometry, recurring object positions,
overall color palette, and basic composition. Treat masters as immutable continuity
anchors, not frames that later beats can silently replace. Whenever a VisualBeat
occurs inside a recurring environment that has a master scene, the beat MUST include
the corresponding master_scene_id. If the same environment is reused across multiple
beats, preserve the correct master_scene_id unless the plan explicitly transitions to
another mastered scene. A recurring environment without a master scene does not force
a master_scene_id. Descendants may change story state such as water, damage, people,
objects or lighting while preserving the recognizable environment.
Reuse one stable location_id for every beat in the same physical place. Create a new
location_id only when the narration genuinely moves to a different sub-location, and
make that transition explicit in the beat's physical state and change description.
For every recurring character, make its canonical description visually specific and
stable: include helmet color, clothing colors, safety vest or gear, simplified body
proportions, face design, and persistent equipment when relevant. Reuse that exact
character ID and identity across beats.

DIRECTING RULES:
1. SHOT PROGRESSION. Use WIDE, MEDIUM, CLOSE, DETAIL and CUTAWAY_DIAGRAM in a
meaningful progression. Every change of framing needs a storytelling reason; never
change camera angle merely for variety.
2. ESTABLISH GEOGRAPHY BEFORE DETAIL. Before CLOSE or DETAIL, identify an earlier
WIDE/master visual in geography_established_by that shows where the subject exists.
3. VISUAL ANTICIPATION. When useful, establish the physical setup and trajectory
before the consequence. Record what is being anticipated, without inventing events.
4. PROGRESSIVE CHANGE. Show worsening states as versions of the same environment:
ankles to knees to waist, clear to opaque, open to partially then fully blocked.
5. SCALE. Use WIDE framing when a small person versus a huge environment or machine
creates tension.
6. DISTANCE FROM SAFETY. When relevant, explicitly show current position, safe exit,
distance/scale, and the obstacle between them.
7. RESTRAIN INFOGRAPHICS. Use diagrams only when spatial understanding requires them.
Prefer the recurring illustrated environment with a simple arrow, route or highlight;
do not make the video resemble presentation slides.
8. REUSE STRONG IMAGES. Prefer REUSE, TRANSFORM, EDIT_EXISTING or OVERLAY when a
strong established frame can be zoomed, cropped, panned, edited or highlighted.
Do not request an unrelated NEW_IMAGE solely to create variety.
9. VISUAL RHYTHM. Assign duration by purpose: a brief beat may be about 2 seconds;
a strong WIDE may hold 5-7 seconds with subtle movement. Do not use one identical
duration for every beat.
10. ADD INFORMATION. Each visual must add spatial, causal, scale, route or state
information beyond the narration instead of merely illustrating its wording.
11. BEAT GRANULARITY. Start a new semantic visual beat when narration introduces a
new object, location, physical state, obstacle, route, important resource, causal
mechanism, comparison, or decision. Do not create beats for filler wording.
Also start a semantic beat when viewer focus changes, important new physical
information appears, geography needs clarification, or cause and effect need separate
visual treatment. A semantic beat does not require a newly generated image: prefer
REUSE, TRANSFORM, OVERLAY, crop/reframe, or detail focus when an established asset can
communicate the change. If the story genuinely requires a long completely motionless
frame, include the exact phrase "intentional static hold" in framing_reason.
{pacing_instruction}

Prefer an established scene that evolves over unrelated replacement images. Target
approximately one meaningful visual change every 3-5 seconds, but do not create a new
image mechanically on that interval. Every referenced character, object, location,
and source visual must use an ID declared in this plan. A source beat must occur
earlier. Keep all descriptions semantic: this plan decides what viewers understand,
not the wording of future image prompts.

ID REFERENCE RULES AND FINAL VALIDATION CHECKLIST. Before returning the JSON,
silently verify every item:
- Every ID is non-empty and unique within its collection.
- Every referenced ID exactly matches an object defined in this same VisualPlan.
  Never invent an ID inside a beat and never put descriptive prose in an ID field.
  Reuse existing canonical IDs instead of creating near-duplicates.
- Every location_id, character ID, important-object ID and progressive-change
  subject_id refers to a declared item of the correct kind.
- Every recurring environment refers to a declared location. If that location has a
  master scene, every beat there sets that exact master_scene_id.
- Every master_scene_id belongs to the beat's location.
- REFERENCE_GENERATION, EDIT_EXISTING, REUSE, TRANSFORM and OVERLAY always provide
  source_visual_id. The source is a declared master scene or an EARLIER beat, never a
  later beat, and a master source belongs to the beat's location. Use NEW_IMAGE when
  no valid earlier source exists.
- CLOSE and DETAIL always provide geography_established_by pointing to a declared
  master scene or EARLIER beat in the SAME location. Never use the current beat, a
  future beat, or an undeclared beat-like ID. If this field is optional and no valid
  geography visual applies, use null.
- OVERLAY always provides overlay_description. All other operations set
  overlay_description to null or omit it.
- Use only keys defined by the JSON Schema. Include every required field, use exact
  enum spellings, keep visual_beats non-empty, and keep every duration above zero.
- Finally, verify every *_id reference against its corresponding definitions in this
  output before returning the JSON.

For each beat, use visual_focus for the one action/object viewers should notice first,
and must_not_show for story mistakes or distracting elements that must be excluded.
Also verify that main_visual_idea contains one dominant fact, visible_physical_state is
only the state that must be drawn, characters_visible and important_objects contain
only essential visible IDs, essential_environment_cues contains only the minimum cues
needed to recognize the place, and optional_entities_to_omit names deliberate
omissions. The complexity_budget is a strong simplicity target, not permission to
discard a story-critical fact.

Return exactly one JSON object matching the supplied JSON Schema. Return no markdown,
code fences, commentary, or image prompts.

JSON Schema:
{schema}

{narration_label}:
<narration>
{narration}
</narration>
"""
