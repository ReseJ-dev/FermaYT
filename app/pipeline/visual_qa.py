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
from app.models.visual_plan import VisualBeat
from app.models.visual_qa import (
    VisualQADecision,
    VisualQAProblemCategory,
    VisualQAResult,
)
from app.style_contracts import DEFAULT_IMAGE_STYLE_ID, get_image_style_contract

logger = logging.getLogger(__name__)

VISUAL_QA_PROMPT_VERSION = "visual_qa_v1"


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
            expected_physical_state=beat.physical_state,
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
            raise VisualQAError("Visual QA provider failed") from exc
        try:
            return VisualQADecision.model_validate(json.loads(raw_result))
        except (json.JSONDecodeError, TypeError, ValidationError) as exc:
            raise VisualQAError(
                "Visual QA provider returned an invalid structured result"
            ) from exc


async def generate_with_visual_qa(
    generate_candidate: Callable[[str | None, str], Awaitable[str]],
    output_path: str,
    context: VisualQAContext,
    qa_service: VisualQAService | None,
    *,
    max_retries: int = 2,
    before_qa_request: Callable[[int], object] | None = None,
    on_qa_request: Callable[[int, bool], object] | None = None,
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
- required objects: {', '.join(context.required_objects) or 'none'}
- important physical action: {context.important_physical_action}
- location: {context.location_id}
- expected physical state: {context.expected_physical_state}
- intentional change from previous beat: {context.change_from_previous}
- resolved visual operation: {context.resolved_operation}
- visible characters: {', '.join(context.characters_visible) or 'none'}
- character continuity definitions: {', '.join(context.character_definitions) or 'none'}
- object continuity definitions: {', '.join(context.object_definitions) or 'none'}
- camera / composition: {context.camera_view}
- provider-ready generation prompt: {context.generation_prompt or 'not supplied'}
- information added beyond narration: {information_added}

CHECK STORY ACCURACY: required objects, visible physical action, and intended purpose.
CHECK CONTINUITY: master location, recurring characters and objects; reject environment
redesign. CHECK STYLE: reject realism, excess detail, polish, cinematic treatment,
childishness, and unwanted textures. CHECK COMPOSITION: action prominence, clutter,
scale of important objects, and overcrowding. CHECK VIDEO READABILITY: rapid
understanding, needed simplification, and whether crop/framing should change.
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
Reject meaningful realism, polish, detail drift, clutter, or loss of readability.

Return PASS only when no correction is needed. PASS_WITH_WARNING is allowed only for
a usable frame with a minor non-blocking defect. Otherwise return REGENERATE with
five dimension scores, stable problem_categories, concrete reasons, severity, and one
actionable correction_instruction that preserves all correct semantic requirements.
Return exactly one JSON object matching
this schema, with no markdown or commentary:
{schema}
"""


_HARD_FAILURE_CATEGORIES = {
    VisualQAProblemCategory.MISSING_REQUIRED_OBJECT,
    VisualQAProblemCategory.WRONG_PHYSICAL_STATE,
    VisualQAProblemCategory.WRONG_CHARACTER,
    VisualQAProblemCategory.LOCATION_DRIFT,
    VisualQAProblemCategory.STORY_ACCURACY,
    VisualQAProblemCategory.CONTINUITY,
    VisualQAProblemCategory.STYLE_DRIFT_REALISM,
    VisualQAProblemCategory.EDIT_CHANGED_TOO_MUCH,
    VisualQAProblemCategory.COMPOSITION_UNCLEAR,
    VisualQAProblemCategory.VIDEO_READABILITY,
}


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
    contract = get_image_style_contract(style_id).render()
    normalized_prompt = prompt.strip()
    if normalized_prompt.endswith(contract):
        normalized_prompt = normalized_prompt[: -len(contract)].rstrip()
    return (
        f"{normalized_prompt}\n\nVISUAL QA CORRECTION FOR REGENERATION:\n"
        f"{correction}\nPreserve all elements that QA did not identify as problems.\n\n"
        f"{contract}"
    )


async def _finish_best(
    candidates: list[_Candidate],
    output_path: str,
    attempts: int,
    warning: str,
) -> VisualQAOutcome:
    best = min(candidates, key=lambda candidate: _candidate_penalty(candidate.decision))
    if is_hard_qa_failure(best.decision):
        raise VisualQAError(
            "Visual QA retries ended without a usable candidate"
        )
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
    return sum(category_weights[category] for category in decision.problem_categories)


def _candidate_path(output_path: str, attempt: int) -> str:
    destination = Path(output_path)
    suffix = destination.suffix or ".png"
    return str(destination.with_name(f"{destination.stem}.qa-{attempt}{suffix}"))


async def _copy_candidate(source_path: str, output_path: str) -> None:
    destination = Path(output_path)
    await asyncio.to_thread(destination.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(shutil.copyfile, source_path, output_path)
