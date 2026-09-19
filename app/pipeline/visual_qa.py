"""Vision-based frame QA with bounded correction and best-candidate fallback."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from app.budgets import GenerationBudgetError
from app.errors import VisualQAError
from app.generators.image_prompt import sanitize_provider_visual_text
from app.models.visual_plan import VisualBeat
from app.models.visual_qa import (
    VisualQADecision,
    VisualQAProblemCategory,
    VisualQAResult,
)
from app.provider_diagnostics import (
    StructuredAIProviderDiagnostic,
    find_structured_ai_provider_diagnostic,
    sanitize_provider_message,
)
from app.style_contracts import (
    DEFAULT_IMAGE_STYLE_ID,
    apply_image_style_contract,
    get_image_style_contract,
)

logger = logging.getLogger(__name__)

VISUAL_QA_PROMPT_VERSION = "visual_qa_v6"


class VisualQAClient(Protocol):
    """Provider-agnostic boundary for a vision-capable structured-output model."""

    async def evaluate(self, prompt: str, image_paths: tuple[str, ...]) -> str: ...


@dataclass(frozen=True, slots=True)
class VisualQAContext:
    visual_purpose: str
    what_viewer_should_understand: str
    required_objects: tuple[str, ...]
    important_physical_action: str
    location_id: str
    expected_physical_state: str
    resolved_operation: str = "NEW_IMAGE"
    characters_visible: tuple[str, ...] = ()
    character_definitions: tuple[str, ...] = ()
    object_definitions: tuple[str, ...] = ()
    camera_view: str = "unspecified"
    change_from_previous: str = "No intentional change specified"
    generation_prompt: str | None = None
    style_id: str = DEFAULT_IMAGE_STYLE_ID
    style_reference_path: str | None = None
    master_reference_path: str | None = None
    previous_frame_path: str | None = None
    source_reference_path: str | None = None
    information_added_beyond_narration: str | None = None
    required_entities: tuple[str, ...] = ()
    required_attributes: tuple[str, ...] = ()
    required_environment: str | None = None
    required_state: str | None = None
    forbidden_major_mismatches: tuple[str, ...] = ()
    generated_text_explicitly_required: bool = False
    main_visual_idea: str | None = None
    visible_physical_state: str | None = None
    essential_visible_entities: tuple[str, ...] = ()
    essential_environment_cues: tuple[str, ...] = ()
    optional_entities_to_omit: tuple[str, ...] = ()
    character_count_target: int | None = None
    background_complexity: str | None = None
    max_main_subjects: int | None = None
    max_supporting_objects: int | None = None
    max_environment_concepts: int | None = None
    max_main_actions: int | None = None

    @classmethod
    def from_beat(
        cls,
        beat: VisualBeat,
        *,
        style_id: str = DEFAULT_IMAGE_STYLE_ID,
        style_reference_path: str | None = None,
        master_reference_path: str | None = None,
        previous_frame_path: str | None = None,
    ) -> VisualQAContext:
        return cls(
            visual_purpose=beat.visual_purpose,
            what_viewer_should_understand=beat.what_viewer_should_understand,
            required_objects=tuple(beat.important_objects),
            important_physical_action=beat.change_from_previous_beat,
            location_id=beat.location_id,
            expected_physical_state=beat.visible_physical_state,
            resolved_operation=beat.preferred_visual_operation.value,
            characters_visible=tuple(beat.characters_visible),
            camera_view=beat.camera_view,
            change_from_previous=beat.change_from_previous_beat,
            style_id=style_id,
            style_reference_path=style_reference_path,
            master_reference_path=master_reference_path,
            previous_frame_path=previous_frame_path,
            information_added_beyond_narration=(
                beat.information_added_beyond_narration
            ),
            main_visual_idea=beat.main_visual_idea,
            visible_physical_state=beat.visible_physical_state,
            essential_visible_entities=tuple(
                dict.fromkeys((*beat.characters_visible, *beat.important_objects))
            ),
            essential_environment_cues=tuple(beat.essential_environment_cues),
            optional_entities_to_omit=tuple(beat.optional_entities_to_omit),
            character_count_target=beat.character_count_target,
            background_complexity=beat.background_complexity.value,
            max_main_subjects=beat.complexity_budget.max_main_subjects,
            max_supporting_objects=beat.complexity_budget.max_supporting_objects,
            max_environment_concepts=(
                beat.complexity_budget.max_environment_concepts
            ),
            max_main_actions=beat.complexity_budget.max_main_actions,
        )


@dataclass(frozen=True, slots=True)
class VisualQAOutcome:
    image_path: str
    decision: VisualQADecision | None
    attempts: int
    warning: str | None = None
    selected_candidate_path: str | None = None


@dataclass(frozen=True, slots=True)
class _Candidate:
    path: str
    decision: VisualQADecision


class VisualQAService:
    def __init__(
        self,
        client: VisualQAClient,
        *,
        provider: str | None = None,
        model: str | None = None,
        prompt_version: str = VISUAL_QA_PROMPT_VERSION,
    ) -> None:
        self._client = client
        self.provider = provider or str(getattr(client, "provider", "unknown"))
        self.model = model or getattr(client, "model", None)
        self.prompt_version = prompt_version

    async def evaluate(
        self,
        image_path: str,
        context: VisualQAContext,
    ) -> VisualQADecision:
        paths = [image_path]
        if context.style_reference_path is not None:
            paths.append(context.style_reference_path)
        if context.master_reference_path is not None:
            paths.append(context.master_reference_path)
        if context.previous_frame_path is not None:
            paths.append(context.previous_frame_path)
        if (
            context.source_reference_path is not None
            and context.source_reference_path not in paths
        ):
            paths.append(context.source_reference_path)
        prompt = build_visual_qa_request(context)
        try:
            raw_result = await self._client.evaluate(prompt, tuple(paths))
        except Exception as exc:
            raise VisualQAError(
                "Visual QA provider failed",
                diagnostic=find_structured_ai_provider_diagnostic(exc),
            ) from exc
        try:
            payload = json.loads(raw_result)
            if isinstance(payload, dict):
                payload = _without_qwen_schema_echo(payload)
                payload = _normalize_contradictory_qwen_pass(payload)
            decision = VisualQADecision.model_validate(payload)
            return _normalize_simplicity_decision(decision, context)
        except (json.JSONDecodeError, TypeError, ValidationError) as exc:
            diagnostic = StructuredAIProviderDiagnostic(
                provider=self.provider,
                model=self.model,
                operation="visual_qa",
                category="VISUAL_QA_INVALID_STRUCTURED_RESULT",
                attempt=1,
                max_attempts=1,
                provider_error=sanitize_provider_message(str(exc)),
                response_length=len(raw_result),
            )
            raise VisualQAError(
                "Visual QA provider returned an invalid structured result",
                diagnostic=diagnostic,
            ) from exc


_QWEN_SCHEMA_ECHO_KEYS = frozenset(
    {
        "$defs",
        "$schema",
        "title",
        "description",
        "type",
        "properties",
        "required",
        "additionalProperties",
    }
)


def _without_qwen_schema_echo(payload: dict[str, object]) -> dict[str, object]:
    """Remove JSON-Schema metadata while retaining strict decision fields."""
    return {
        key: value
        for key, value in payload.items()
        if key not in _QWEN_SCHEMA_ECHO_KEYS
    }


def _normalize_contradictory_qwen_pass(
    payload: dict[str, object],
) -> dict[str, object]:
    """Fail safely when Qwen says PASS while simultaneously reporting defects."""
    if payload.get("result") != VisualQAResult.PASS.value:
        return payload
    categories = payload.get("problem_categories")
    reasons = payload.get("reasons")
    correction = payload.get("correction_instruction")
    has_categories = isinstance(categories, list) and bool(categories)
    has_reasons = isinstance(reasons, list) and bool(reasons)
    has_correction = isinstance(correction, str) and bool(correction.strip())
    if not (has_categories or has_reasons or has_correction):
        return payload

    if not has_categories and not has_correction and payload.get("severity") is None:
        # Qwen sometimes explains a clean PASS with positive observations.
        # Explanatory prose is not a warning when no defect category, severity,
        # or requested correction accompanies it.
        normalized_pass = dict(payload)
        normalized_pass["reasons"] = []
        return normalized_pass

    normalized = dict(payload)
    category_values = (
        [item for item in categories if isinstance(item, str)]
        if isinstance(categories, list)
        else []
    )
    reason_values = (
        [item.strip() for item in reasons if isinstance(item, str) and item.strip()]
        if isinstance(reasons, list)
        else []
    )
    if not category_values:
        category_values = [VisualQAProblemCategory.OTHER.value]
    if not reason_values:
        reason_values = ["The provider reported that the frame needs correction"]
    normalized["problem_categories"] = category_values
    normalized["reasons"] = reason_values

    hard_values = {item.value for item in _HARD_FAILURE_CATEGORIES}
    is_hard = normalized.get("severity") == "critical" or bool(
        set(category_values) & hard_values
    )
    if is_hard or has_correction:
        normalized["result"] = VisualQAResult.REGENERATE.value
        normalized["severity"] = normalized.get("severity") or "major"
        if not has_correction:
            normalized["correction_instruction"] = (
                "Correct the reported visual problems: " + "; ".join(reason_values)
            )
    else:
        normalized["result"] = VisualQAResult.PASS_WITH_WARNING.value
        normalized["severity"] = normalized.get("severity") or "minor"
        normalized["correction_instruction"] = None
    return normalized


async def generate_with_visual_qa(
    generate_candidate: Callable[[str | None, str], Awaitable[str]],
    output_path: str,
    context: VisualQAContext,
    qa_service: VisualQAService | None,
    *,
    max_retries: int = 2,
    before_qa_request: Callable[[int], object] | None = None,
    on_qa_request: Callable[[int, bool], object] | None = None,
    on_qa_decision: Callable[[str, VisualQADecision], object] | None = None,
) -> VisualQAOutcome:
    """Generate, inspect and correct a frame without an unbounded retry loop."""
    if not 0 <= max_retries <= 5:
        raise ValueError("max_retries must be between 0 and 5")
    if qa_service is None:
        generated_path = await generate_candidate(None, output_path)
        warning = "Visual QA skipped: no vision-capable model configured"
        logger.warning(warning)
        return VisualQAOutcome(
            generated_path,
            None,
            1,
            warning,
            selected_candidate_path=generated_path,
        )

    candidates: list[_Candidate] = []
    correction: str | None = None
    total_attempts = max_retries + 1
    for attempt in range(1, total_attempts + 1):
        candidate_path = _candidate_path(output_path, attempt)
        try:
            generated_path = await generate_candidate(correction, candidate_path)
        except GenerationBudgetError:
            raise
        except Exception:
            if not candidates:
                raise
            warning = "Regeneration failed; kept the best earlier QA candidate"
            logger.warning(warning)
            return await _finish_best(candidates, output_path, attempt, warning)
        if not Path(generated_path).is_file():
            raise VisualQAError("Generated QA candidate file is missing")
        try:
            if before_qa_request is not None:
                before_qa_request(attempt)
            decision = await qa_service.evaluate(generated_path, context)
            if on_qa_decision is not None:
                on_qa_decision(generated_path, decision)
            if on_qa_request is not None:
                on_qa_request(attempt, True)
        except VisualQAError:
            if on_qa_request is not None:
                on_qa_request(attempt, False)
            if not candidates:
                warning = "Visual QA unavailable; kept the generated candidate"
                logger.warning(warning)
                await _copy_candidate(generated_path, output_path)
                return VisualQAOutcome(
                    output_path,
                    None,
                    attempt,
                    warning,
                    selected_candidate_path=generated_path,
                )
            warning = "Visual QA became unavailable; kept the best earlier candidate"
            logger.warning(warning)
            return await _finish_best(candidates, output_path, attempt, warning)

        logger.info(
            "Visual QA attempt %s: %s%s",
            attempt,
            decision.result.value,
            f" — {'; '.join(decision.reasons)}" if decision.reasons else "",
        )
        candidate = _Candidate(generated_path, decision)
        candidates.append(candidate)
        if decision.result in {
            VisualQAResult.PASS,
            VisualQAResult.PASS_WITH_WARNING,
        }:
            await _copy_candidate(generated_path, output_path)
            return VisualQAOutcome(
                output_path,
                decision,
                attempt,
                selected_candidate_path=generated_path,
            )
        correction = decision.correction_instruction

    warning = (
        f"Visual QA did not pass after {total_attempts} attempts; "
        "kept the best candidate"
    )
    logger.warning(warning)
    return await _finish_best(candidates, output_path, total_attempts, warning)


def build_visual_qa_request(context: VisualQAContext) -> str:
    schema = json.dumps(VisualQADecision.model_json_schema(), ensure_ascii=False)
    style_contract = get_image_style_contract(context.style_id).render()
    image_roles = ["IMAGE 1 is the candidate video frame."]
    next_image = 2
    if context.style_reference_path is not None:
        image_roles.append(
            f"IMAGE {next_image} is STYLE_REFERENCE; compare drawing complexity and style only."
        )
        next_image += 1
    if context.master_reference_path is not None:
        image_roles.append(
            f"IMAGE {next_image} is the immutable MASTER SCENE; compare layout and continuity."
        )
        next_image += 1
    if context.previous_frame_path is not None:
        previous_role = (
            "PREVIOUS VIDEO FRAME AND EDIT SOURCE"
            if context.resolved_operation == "EDIT_EXISTING"
            else "PREVIOUS VIDEO FRAME"
        )
        image_roles.append(
            f"IMAGE {next_image} is the {previous_role}; compare meaningful "
            "progression and reject arbitrary visual switching or redundant decoration."
        )
        next_image += 1
    if (
        context.source_reference_path is not None
        and context.source_reference_path
        not in {context.master_reference_path, context.previous_frame_path}
    ):
        image_roles.append(
            f"IMAGE {next_image} is the EDIT SOURCE; verify that KEEP elements "
            "remain unchanged and only the intentional state change was applied."
        )
    information_added = (
        context.information_added_beyond_narration
        or "The frame must add useful spatial, causal, scale, route, or state information."
    )
    return f"""Judge the candidate as a VIDEO FRAME, not standalone artwork.

Main question: will this frame communicate the required story information clearly
within a few seconds?

Apply this priority order: storytelling clarity, spatial continuity, visual
progression, readability, consistent style, generation efficiency, image beauty.
A beautiful decorative image must not PASS when it is less clear, breaks continuity,
or fails to advance the finished sequence.

IMAGE ROLES:
{chr(10).join(image_roles)}

REQUIRED STORY INFORMATION:
- visual purpose: {context.visual_purpose}
- viewer should understand: {context.what_viewer_should_understand}
- required objects: {", ".join(context.required_objects) or "none"}
- important physical action: {context.important_physical_action}
- location: {context.location_id}
- expected physical state: {context.expected_physical_state}
- intentional change from previous beat: {context.change_from_previous}
- resolved visual operation: {context.resolved_operation}
- visible characters: {", ".join(context.characters_visible) or "none"}
- character continuity definitions: {", ".join(context.character_definitions) or "none"}
- object continuity definitions: {", ".join(context.object_definitions) or "none"}
- camera / composition: {context.camera_view}
- provider-ready generation prompt: {context.generation_prompt or "not supplied"}
- information added beyond narration: {information_added}
- required entities: {", ".join(context.required_entities) or "none"}
- required identity attributes: {", ".join(context.required_attributes) or "none"}
- required environment: {context.required_environment or context.location_id}
- required story state: {context.required_state or context.expected_physical_state}
- forbidden major mismatches: {", ".join(context.forbidden_major_mismatches) or "none"}

PLANNED VISUAL SIMPLICITY:
- one-second visual idea: {context.main_visual_idea or context.what_viewer_should_understand}
- essential visible entities only: {", ".join(context.essential_visible_entities) or ", ".join((*context.characters_visible, *context.required_objects)) or "none"}
- required visible state: {context.visible_physical_state or context.required_state or context.expected_physical_state}
- minimal environment cues: {", ".join(context.essential_environment_cues) or context.location_id}
- intended visible character count: {context.character_count_target if context.character_count_target is not None else "not explicitly limited"}
- background complexity: {context.background_complexity or "simple enough for immediate reading"}
- complexity budget: main subjects <= {context.max_main_subjects if context.max_main_subjects is not None else "planned essentials"}; supporting objects <= {context.max_supporting_objects if context.max_supporting_objects is not None else "planned essentials"}; environment concepts <= {context.max_environment_concepts if context.max_environment_concepts is not None else "minimum needed"}; main actions <= {context.max_main_actions if context.max_main_actions is not None else 1}
- optional entities explicitly omitted: {", ".join(context.optional_entities_to_omit) or "none"}

CHECK STORY ACCURACY: required objects, visible physical action, and intended purpose.
Explicitly verify every required entity, its story-critical role/identity attributes,
the required environment, and required physical state. Generic people cannot replace
named roles such as miners: required work clothing and mining helmets must remain
recognizable. Use MISSING_REQUIRED_ENTITY, WRONG_ENTITY_IDENTITY, WRONG_ENVIRONMENT,
or WRONG_PHYSICAL_STATE for these hard failures.
CHECK CONTINUITY: master location, recurring characters and objects; reject environment
redesign. Compare recurring character helmet color, clothing colors, safety vest,
body proportions, simple face design, and carried equipment. Use
CHARACTER_IDENTITY_DRIFT when those identifiers change. Compare the master location's
identity, proportions, palette and geometry only for anchors actually required by the
current simplified visual core. The master is continuity authority, not mandatory
inventory: do not require its pipes, supports, rails, lights, tools, machinery, or
other objects when they are absent from the planned essential entities and environment
cues. Use LOCATION_IDENTITY_DRIFT or ENVIRONMENT_MISMATCH when the required environment
is replaced or redesigned. A previous frame from another location does not override
the explicitly required current location or its master; do not reject a legitimate
location transition described by the current requirements.
CHECK STYLE: compare line thickness, character simplification, color flatness,
shading level, detail ceiling, and handmade appearance against STYLE_REFERENCE.
Use STYLE_DETAIL_DRIFT for excess or inconsistent detail and STYLE_SHADING_DRIFT for
soft, dimensional, cinematic, or otherwise inconsistent shading. Generated content
or previous frames never override STYLE_REFERENCE. CHECK COMPOSITION: action
prominence, clutter, scale of important objects, and overcrowding. The illustrated
environment must fill the full 16:9 image edge-to-edge. Use UNWANTED_FRAME_OR_MARGIN
and REGENERATE for a large white external margin, drawn rectangular image frame,
paper-like border, poster/panel boundary, or fake canvas edge unless the story
explicitly requires it. CHECK VIDEO READABILITY: rapid understanding, needed
simplification, and whether crop/framing should change.
CHECK VISUAL SIMPLICITY AS A SEPARATE REQUIREMENT: decide whether the main visual
idea is obvious in about one second and whether there is one dominant subject/action.
Compare the candidate against PLANNED VISUAL SIMPLICITY, not against everything known
about the master location. Extra master inventory is not automatically required.
Flag unnecessary objects, extra characters, detailed background machinery or rock
texture, and unrelated competing actions. A complex detailed scene rendered with
thick outlines is not a simple explainer drawing. Use EXCESSIVE_VISUAL_COMPLEXITY
when clutter clearly obscures the planned core, TOO_MANY_CHARACTERS when the planned
count is materially exceeded, UNCLEAR_VISUAL_FOCUS when the one-second idea is not
obvious, MULTIPLE_COMPETING_ACTIONS for unrelated actions, and
UNNECESSARY_BACKGROUND_DETAIL for a minor usable excess. EXCESSIVE_CLUTTER remains
valid for object clutter. Do not invent missing items merely to fill the scene.
CHECK UNINTENDED TEXT: reject any visible prompt wording, pseudo-text, caption, section
heading, operation name,
technical label, watermark, or interface text unless readable story-world text is
explicitly required by the scene ({context.generated_text_explicitly_required}). Any
visible phrase such as VISUAL OPERATION, PURPOSE, STATE, CHANGE, VISUAL
REFERENCE, REFERENCE INSTRUCTIONS, IMAGE REFERENCE, or STYLE CONTRACT is always a hard
REGENERATE with problem category UNINTENDED_TEXT, never PASS or PASS_WITH_WARNING.
CHECK VISUAL PROGRESSION: compared with the previous frame when supplied, confirm
that the beat advances state, understanding, framing, or route information without
an arbitrary location switch or needless repetition.
CHECK OPERATION CORRECTNESS: for EDIT_EXISTING preserve the source camera, location
geometry, fixed objects and all KEEP constraints; apply only the explicitly intended
change. Do not label an intentional story change as continuity drift.

VIDEO READABILITY TEST: if this image appears for approximately 3-5 seconds, can the
viewer understand the intended visual information in about 2 seconds?

PERMANENT STYLE CONTRACT:
{style_contract}

The intended style is deliberately simple, rough, flat and slightly imperfect. Do not
reject crude geometry, uneven lines, simplified anatomy or mildly imperfect perspective.
Sparse backgrounds, naive drawing, simplified proportions and visibly sketch-like
construction are desirable, not defects. Reject meaningful realism, polish, detail
drift, clutter, or loss of readability. Do not require an aesthetically polished or
more detailed drawing, but still require a coherent, intact and readable result; do
not reward broken rendering merely because it is simple.

SIMPLICITY SEVERITY:
- harmless slight extra background detail with a clear focal idea: PASS_WITH_WARNING;
- clearly overcomplicated frame, materially excessive character count, obscured focal
  idea, or competing actions: REGENERATE;
- do not reject harmless minor variation.
For a complexity REGENERATE, correction_instruction must say what to REMOVE and what
small planned core to KEEP. Do not respond only with generic wording such as "improve
clarity" or add new objects.

Return PASS only when no correction is needed. PASS_WITH_WARNING is allowed only for
a usable frame with a minor non-blocking defect. Otherwise return REGENERATE with
six dimension scores, stable problem_categories, concrete reasons, severity, and one
actionable correction_instruction that preserves all correct semantic requirements.
Return exactly one JSON object matching
this schema, with no markdown or commentary:
{schema}
"""


_HARD_FAILURE_CATEGORIES = {
    VisualQAProblemCategory.MISSING_REQUIRED_ENTITY,
    VisualQAProblemCategory.WRONG_ENTITY_IDENTITY,
    VisualQAProblemCategory.WRONG_ENVIRONMENT,
    VisualQAProblemCategory.MISSING_REQUIRED_OBJECT,
    VisualQAProblemCategory.WRONG_PHYSICAL_STATE,
    VisualQAProblemCategory.WRONG_CHARACTER,
    VisualQAProblemCategory.LOCATION_DRIFT,
    VisualQAProblemCategory.CHARACTER_IDENTITY_DRIFT,
    VisualQAProblemCategory.LOCATION_IDENTITY_DRIFT,
    VisualQAProblemCategory.ENVIRONMENT_MISMATCH,
    VisualQAProblemCategory.STORY_ACCURACY,
    VisualQAProblemCategory.CONTINUITY,
    VisualQAProblemCategory.STYLE_DRIFT_REALISM,
    VisualQAProblemCategory.STYLE_DETAIL_DRIFT,
    VisualQAProblemCategory.STYLE_SHADING_DRIFT,
    VisualQAProblemCategory.STYLE_DRIFT,
    VisualQAProblemCategory.EDIT_CHANGED_TOO_MUCH,
    VisualQAProblemCategory.COMPOSITION_UNCLEAR,
    VisualQAProblemCategory.UNWANTED_TEXT,
    VisualQAProblemCategory.UNINTENDED_TEXT,
    VisualQAProblemCategory.UNWANTED_FRAME_OR_MARGIN,
    VisualQAProblemCategory.VIDEO_READABILITY,
    VisualQAProblemCategory.EXCESSIVE_VISUAL_COMPLEXITY,
    VisualQAProblemCategory.TOO_MANY_CHARACTERS,
    VisualQAProblemCategory.UNCLEAR_VISUAL_FOCUS,
    VisualQAProblemCategory.MULTIPLE_COMPETING_ACTIONS,
}

_SIMPLICITY_PROBLEM_CATEGORIES = {
    VisualQAProblemCategory.EXCESSIVE_VISUAL_COMPLEXITY,
    VisualQAProblemCategory.UNNECESSARY_BACKGROUND_DETAIL,
    VisualQAProblemCategory.TOO_MANY_CHARACTERS,
    VisualQAProblemCategory.UNCLEAR_VISUAL_FOCUS,
    VisualQAProblemCategory.MULTIPLE_COMPETING_ACTIONS,
    VisualQAProblemCategory.EXCESSIVE_CLUTTER,
}


def _normalize_simplicity_decision(
    decision: VisualQADecision,
    context: VisualQAContext,
) -> VisualQADecision:
    """Turn complexity failures into bounded removal instructions, never additions."""
    if decision.result is not VisualQAResult.REGENERATE:
        return decision
    if not set(decision.problem_categories) & _SIMPLICITY_PROBLEM_CATEGORIES:
        return decision
    keep = list(context.essential_visible_entities)
    if not keep:
        keep.extend(context.characters_visible)
        keep.extend(context.required_objects)
    state = context.visible_physical_state or context.required_state
    cues = list(context.essential_environment_cues)
    keep_text = ", ".join(dict.fromkeys(item for item in keep if item)) or "the main subject"
    if state:
        keep_text += f", {state}"
    if cues:
        keep_text += f", and minimal environment cues ({', '.join(cues)})"
    count = (
        f" Show exactly {context.character_count_target} visible character(s)."
        if context.character_count_target is not None
        else ""
    )
    focus = context.main_visual_idea or context.what_viewer_should_understand
    correction = (
        f"Keep only {keep_text}.{count} Remove extra characters and every unrelated "
        "object, including machinery, pipes, tools, signs, debris, decorative "
        f"texture, and background clutter. Keep one dominant visual idea: {focus}"
    )
    return decision.model_copy(update={"correction_instruction": correction})


def is_hard_qa_failure(decision: VisualQADecision) -> bool:
    """Conservatively prevent a seriously wrong frame from becoming lineage."""
    if decision.severity is not None and decision.severity.value == "critical":
        return True
    return bool(set(decision.problem_categories) & _HARD_FAILURE_CATEGORIES)


def qa_candidate_penalty(decision: VisualQADecision) -> float:
    """Rank imperfect candidates by video usefulness, not image beauty."""
    scores = decision.scores
    weighted_loss = (
        (1 - scores.story_clarity) * 5
        + (1 - scores.continuity) * 5
        + (1 - scores.composition) * 4
        + (1 - scores.operation_correctness) * 4
        + (1 - scores.visual_simplicity) * 4
        + (1 - scores.style) * 2
    )
    return weighted_loss + len(decision.problem_categories) * 0.1


def apply_visual_qa_correction(
    prompt: str,
    correction_instruction: str | None,
    style_id: str = DEFAULT_IMAGE_STYLE_ID,
) -> str:
    """Insert a trusted QA correction while keeping the style contract last."""
    if correction_instruction is None:
        return prompt
    correction = correction_instruction.strip()
    if not correction:
        raise ValueError("correction_instruction must not be empty")
    if "STYLE CONTRACT [" in correction:
        raise VisualQAError("QA correction cannot replace the image style contract")
    style = get_image_style_contract(style_id)
    contract = style.render_for_image_provider()
    normalized_prompt = prompt.strip()
    if normalized_prompt.endswith(contract):
        normalized_prompt = normalized_prompt[: -len(contract)].rstrip()
    elif normalized_prompt.endswith(style.render()):
        normalized_prompt = normalized_prompt[: -len(style.render())].rstrip()
    normalized_prompt = _remove_previous_qa_correction(normalized_prompt)
    correction = sanitize_provider_visual_text(correction)
    corrected_dynamic_prompt = (
        f"{normalized_prompt}\n\nRegenerate the illustration so that {correction}. "
        "Preserve every correct visual element."
    )
    return apply_image_style_contract(corrected_dynamic_prompt, style_id)


def _remove_previous_qa_correction(prompt: str) -> str:
    """Replace the prior bounded retry delta instead of accumulating corrections."""
    marker = "\n\nRegenerate the illustration so that "
    start = prompt.rfind(marker)
    if start < 0:
        return prompt
    tail = prompt[start + len(marker) :]
    if not tail.endswith(". Preserve every correct visual element."):
        return prompt
    return prompt[:start].rstrip()


async def _finish_best(
    candidates: list[_Candidate],
    output_path: str,
    attempts: int,
    warning: str,
) -> VisualQAOutcome:
    best = min(candidates, key=lambda candidate: _candidate_penalty(candidate.decision))
    if is_hard_qa_failure(best.decision):
        raise VisualQAError("Visual QA retries ended without a usable candidate")
    await _copy_candidate(best.path, output_path)
    return VisualQAOutcome(
        output_path,
        best.decision,
        attempts,
        warning,
        selected_candidate_path=best.path,
    )


def _candidate_penalty(decision: VisualQADecision) -> int:
    category_weights = {
        VisualQAProblemCategory.STORY_ACCURACY: 8,
        VisualQAProblemCategory.CONTINUITY: 7,
        VisualQAProblemCategory.VISUAL_PROGRESSION: 6,
        VisualQAProblemCategory.VIDEO_READABILITY: 5,
        VisualQAProblemCategory.COMPOSITION: 4,
        VisualQAProblemCategory.STYLE_DRIFT: 3,
    }
    return sum(
        category_weights.get(category, 2) for category in decision.problem_categories
    )


def _candidate_path(output_path: str, attempt: int) -> str:
    destination = Path(output_path)
    suffix = destination.suffix or ".png"
    return str(destination.with_name(f"{destination.stem}.qa-{attempt}{suffix}"))


async def _copy_candidate(source_path: str, output_path: str) -> None:
    destination = Path(output_path)
    await asyncio.to_thread(destination.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(shutil.copyfile, source_path, output_path)
