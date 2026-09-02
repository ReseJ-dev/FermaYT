"""Monotonic mapping of semantic visual beats to narration audio intervals."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from sqlalchemy.orm import Session

from app.models.timeline import BeatAlignmentMethod, NarrationTimingTrack
from app.models.visual_plan import VisualBeat
from app.persistence import ProjectNarrationAlignment, ProjectNarrationAsset
from app.repositories import (
    create_narration_alignment,
    get_narration_alignment_by_revision,
    get_project_visual_plan_record,
)
from app.services.narration import narration_timing_from_asset
from app.services.visual_planning import (
    hash_story_text,
    require_current_project_visual_plan,
)

NARRATION_ALIGNMENT_VERSION = "narration_alignment_v1"


@dataclass(frozen=True, slots=True)
class BeatAlignmentDraft:
    position: int
    beat_id: str
    source_text: str
    source_start_char: int
    source_end_char: int
    audio_start: float
    audio_end: float
    alignment_method: BeatAlignmentMethod
    alignment_confidence: float
    warning: str | None = None


def align_project_visual_beats(
    session: Session,
    project_id: str,
    narration_asset: ProjectNarrationAsset,
) -> ProjectNarrationAlignment:
    state = require_current_project_visual_plan(session, project_id)
    plan_record = get_project_visual_plan_record(session, project_id)
    if plan_record is None:
        raise ValueError("Project visual plan is missing")
    if narration_asset.project_id != project_id:
        raise ValueError("Narration asset belongs to another project")
    if narration_asset.story_text_hash != hash_story_text(
        narration_asset.project.story_text
    ):
        raise ValueError("Narration asset is stale for the current project story")
    plan_revision = _stable_hash(state.plan.model_dump(mode="json"))
    timing = narration_timing_from_asset(narration_asset)
    alignment_revision = _stable_hash(
        {
            "project_id": project_id,
            "visual_plan_revision": plan_revision,
            "narration_revision": narration_asset.generation_revision,
            "timing": timing.model_dump(mode="json"),
            "alignment_version": NARRATION_ALIGNMENT_VERSION,
        }
    )
    existing = get_narration_alignment_by_revision(session, alignment_revision)
    if existing is not None:
        return existing
    drafts = align_visual_beats(
        narration_asset.project.story_text,
        state.plan.visual_beats,
        timing,
        narration_asset.duration,
    )
    warnings = [item.warning for item in drafts if item.warning is not None]
    return create_narration_alignment(
        session,
        project_id=project_id,
        visual_plan_id=plan_record.id,
        visual_plan_revision=plan_revision,
        narration_asset_id=narration_asset.id,
        alignment_version=NARRATION_ALIGNMENT_VERSION,
        alignment_revision=alignment_revision,
        warnings=warnings,
        average_confidence=sum(item.alignment_confidence for item in drafts)
        / len(drafts),
        beat_timings=[
            {
                "position": item.position,
                "beat_id": item.beat_id,
                "source_text": item.source_text,
                "source_start_char": item.source_start_char,
                "source_end_char": item.source_end_char,
                "audio_start": item.audio_start,
                "audio_end": item.audio_end,
                "alignment_method": item.alignment_method.value,
                "alignment_confidence": item.alignment_confidence,
                "warning": item.warning,
            }
            for item in drafts
        ],
    )


def align_visual_beats(
    story_text: str,
    beats: list[VisualBeat],
    timing: NarrationTimingTrack,
    duration: float,
) -> list[BeatAlignmentDraft]:
    """Align the complete ordered beat sequence while never moving backward."""
    if not beats:
        raise ValueError("visual plan must contain beats")
    cursor = 0
    drafts: list[BeatAlignmentDraft] = []
    for position, beat in enumerate(beats):
        match = _match_segment(story_text, beat.narration_segment, cursor)
        if match is None:
            remaining_weight = sum(
                max(len(item.narration_segment), 1) for item in beats[position:]
            )
            own_weight = max(len(beat.narration_segment), 1)
            available = max(len(story_text) - cursor, len(beats) - position)
            width = max(1, round(available * own_weight / remaining_weight))
            start_char = min(cursor, max(len(story_text) - 1, 0))
            end_char = min(len(story_text), max(start_char + 1, cursor + width))
            method = BeatAlignmentMethod.PROPORTIONAL_FALLBACK
            confidence = 0.35
            warning = (
                f"Ambiguous narration alignment for beat {beat.id}; "
                "used monotonic proportional fallback"
            )
        else:
            start_char, end_char, method, confidence, warning = match
        if drafts and start_char < drafts[-1].source_end_char:
            start_char = drafts[-1].source_end_char
            end_char = max(end_char, start_char + 1)
            end_char = min(end_char, len(story_text))
        cursor = end_char
        audio_start = _char_to_audio(
            start_char,
            timing,
            duration,
        )
        audio_end = _char_to_audio(
            end_char,
            timing,
            duration,
        )
        if drafts:
            audio_start = max(audio_start, drafts[-1].audio_start)
        if audio_end <= audio_start:
            audio_end = min(duration, audio_start + max(duration * 0.001, 0.001))
        if audio_end <= audio_start:
            raise ValueError(f"Could not create a positive audio interval for {beat.id}")
        drafts.append(
            BeatAlignmentDraft(
                position=position,
                beat_id=beat.id,
                source_text=beat.narration_segment,
                source_start_char=start_char,
                source_end_char=end_char,
                audio_start=audio_start,
                audio_end=audio_end,
                alignment_method=method,
                alignment_confidence=confidence,
                warning=warning,
            )
        )
    return drafts


def _match_segment(
    story: str,
    segment: str,
    cursor: int,
) -> tuple[int, int, BeatAlignmentMethod, float, str | None] | None:
    exact = story.find(segment, cursor)
    if exact >= 0:
        return exact, exact + len(segment), BeatAlignmentMethod.EXACT, 1.0, None
    story_tokens = list(re.finditer(r"\w+", story[cursor:], flags=re.UNICODE))
    segment_tokens = [
        item.group().casefold()
        for item in re.finditer(r"\w+", segment, flags=re.UNICODE)
    ]
    values = [item.group().casefold() for item in story_tokens]
    if not segment_tokens:
        return None
    for index in range(len(values) - len(segment_tokens) + 1):
        if values[index : index + len(segment_tokens)] == segment_tokens:
            first = story_tokens[index]
            last = story_tokens[index + len(segment_tokens) - 1]
            return (
                cursor + first.start(),
                cursor + last.end(),
                BeatAlignmentMethod.NORMALIZED_EXACT,
                0.96,
                None,
            )

    best: tuple[float, int, int] | None = None
    min_width = max(1, len(segment_tokens) - 2)
    max_width = len(segment_tokens) + 2
    for start in range(len(values)):
        for width in range(min_width, max_width + 1):
            end = min(len(values), start + width)
            if end <= start:
                continue
            score = SequenceMatcher(
                None,
                segment_tokens,
                values[start:end],
                autojunk=False,
            ).ratio()
            if best is None or score > best[0]:
                best = score, start, end
    if best is None or best[0] < 0.72:
        return None
    score, start, end = best
    first = story_tokens[start]
    last = story_tokens[end - 1]
    warning = (
        f"Conservative fuzzy narration match used (confidence {score:.2f})"
        if score < 0.9
        else None
    )
    return (
        cursor + first.start(),
        cursor + last.end(),
        BeatAlignmentMethod.FUZZY,
        round(score * 0.9, 3),
        warning,
    )


def _char_to_audio(
    character: int,
    timing: NarrationTimingTrack,
    duration: float,
) -> float:
    if character <= 0:
        return 0.0
    for item in timing.items:
        start_char = item.source_start_char
        end_char = item.source_end_char
        if start_char is None or end_char is None:
            continue
        if start_char <= character <= end_char:
            fraction = (character - start_char) / max(end_char - start_char, 1)
            return item.start + (item.end - item.start) * fraction
        if character < start_char:
            return item.start
    return duration


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
