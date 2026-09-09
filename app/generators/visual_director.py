"""Whole-story semantic visual planning before asset generation."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import ValidationError

from app.errors import VisualDirectorError
from app.models.visual_plan import (
    VisualPlan,
    VisualPlanDuplicateIdError,
    VisualPlanReferenceError,
)

MAX_VISUAL_PLAN_REPAIR_ATTEMPTS = 2


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


class VisualDirector:
    """Turn complete narration into a validated semantic visual plan."""

    def __init__(
        self,
        client: VisualPlanningClient,
        *,
        max_repair_attempts: int = MAX_VISUAL_PLAN_REPAIR_ATTEMPTS,
        on_repair_event: Callable[[VisualPlanRepairEvent], None] | None = None,
        before_provider_call: Callable[[int, bool], None] | None = None,
    ) -> None:
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must not be negative")
        self._client = client
        self.max_repair_attempts = max_repair_attempts
        self._on_repair_event = on_repair_event
        self._before_provider_call = before_provider_call
        self.provider_requests = 0

    async def create_plan(self, narration: str) -> VisualPlan:
        normalized_narration = narration.strip()
        if not normalized_narration:
            raise ValueError("Narration must not be empty")

        raw_plan = await self._generate(
            build_visual_director_request(normalized_narration),
            is_repair=False,
        )
        last_diagnostic: VisualPlanValidationDiagnostic | None = None
        for repair_attempt in range(self.max_repair_attempts + 1):
            try:
                payload = json.loads(raw_plan)
                plan = VisualPlan.model_validate(payload)
            except (json.JSONDecodeError, TypeError, ValidationError) as exc:
                last_diagnostic = _validation_diagnostic(exc)
                self._emit_repair_event(
                    VisualPlanRepairEvent(
                        repair_attempt=repair_attempt,
                        category=last_diagnostic.category,
                        succeeded=False,
                        issue=last_diagnostic.issue,
                    )
                )
                if repair_attempt >= self.max_repair_attempts:
                    raise VisualDirectorError(
                        "Visual planning provider returned an invalid structured "
                        f"visual plan after {repair_attempt} repair attempt(s): "
                        f"{last_diagnostic.summary}",
                        validation_category=last_diagnostic.category,
                        diagnostic=last_diagnostic.issue,
                        provider_requests=self.provider_requests,
                    ) from exc
                repair_prompt = build_visual_plan_repair_request(
                    raw_plan,
                    exc,
                    last_diagnostic,
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
            return plan

        raise AssertionError("visual plan validation loop did not terminate")

    async def _generate(self, prompt: str, *, is_repair: bool) -> str:
        next_request = self.provider_requests + 1
        if self._before_provider_call is not None:
            self._before_provider_call(next_request, is_repair)
        self.provider_requests = next_request
        try:
            return await self._client.generate(prompt)
        except Exception as exc:
            message = (
                f"Visual plan repair provider failed on request {next_request}"
                if is_repair
                else "Visual planning provider failed"
            )
            raise VisualDirectorError(
                message,
                validation_category="OTHER_VISUAL_PLAN_ERROR",
                provider_requests=self.provider_requests,
            ) from exc

    def _emit_repair_event(self, event: VisualPlanRepairEvent) -> None:
        if self._on_repair_event is not None:
            self._on_repair_event(event)


def _validation_summary(error: Exception) -> str:
    """Return useful schema diagnostics without including provider-generated values."""
    return _validation_diagnostic(error).summary


def _validation_diagnostic(error: Exception) -> VisualPlanValidationDiagnostic:
    """Classify a parse/schema/reference error using the canonical validator context."""
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


def build_visual_plan_repair_request(
    raw_plan: str,
    error: Exception,
    diagnostic: VisualPlanValidationDiagnostic,
) -> str:
    """Build a targeted request that preserves content and repairs structure only."""
    validation_errors = _repair_error_payload(error, diagnostic)
    schema = json.dumps(VisualPlan.model_json_schema(), ensure_ascii=False)
    return f"""The previous VisualPlan is structurally invalid.

Repair ONLY the structural consistency of the same JSON. Do not rewrite the story.
Preserve story meaning, beat count and order, narration mapping, locations,
characters, objects, physical progression, operations, and visual strategy unless a
specific validation error requires a minimal change.

VALIDATION CATEGORY: {diagnostic.category}
VALIDATION ERRORS:
{json.dumps(validation_errors, ensure_ascii=False)}

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


def build_visual_director_request(narration: str) -> str:
    """Build a provider-neutral request containing the complete narration."""
    schema = json.dumps(VisualPlan.model_json_schema(), ensure_ascii=False)
    return f"""You are the Visual Director for a coherent narrated video.

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

Read the COMPLETE narration before planning any individual beat. Narration and
visuals have different jobs: do not paraphrase narration and do not write image
generation prompts. First identify all characters, locations, important objects,
recurring environments, and possible master scenes. Then create meaningful visual
beats that make spatial layout, movement, distance, routes, obstructions, danger,
breakage, and progressive physical changes easy to understand.

MASTER SCENES. Select only a small number of important recurring environments. For
each master, define stable environment geometry, recurring object positions, overall
color palette, and basic composition. Treat masters as immutable continuity anchors,
not frames that later beats can silently replace. Every beat occurring in a recurring
environment must explicitly set master_scene_id. Descendants may change story state
such as water, damage, people, objects or lighting while preserving the recognizable
environment.

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

Return exactly one JSON object matching the supplied JSON Schema. Return no markdown,
code fences, commentary, or image prompts.

JSON Schema:
{schema}

COMPLETE NARRATION:
<narration>
{narration}
</narration>
"""
