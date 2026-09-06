"""Versioned production profiles and the deterministic Draft execution policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from app.models.visual_plan import (
    CameraMovement,
    VisualBeat,
    VisualOperation,
    VisualPlan,
)


class ProductionProfile(str, Enum):
    DRAFT = "DRAFT"
    FINAL = "FINAL"


FINAL_PROFILE_VERSION = "final_v1"
DRAFT_PROFILE_VERSION = "draft_visual_policy_v1"

_CRITICAL_TERMS = (
    "accident", "failure", "break", "broken", "collapse", "blocked",
    "danger", "consequence", "impact", "explosion", "fire", "flood",
    "rescue", "escape", "final state", "авари", "слом", "обруш",
    "заблок", "опас", "последств", "взрыв", "пожар", "затоп", "спас",
)


@dataclass(frozen=True, slots=True)
class DraftBeatSelection:
    key_beat_ids: frozenset[str]
    target_paid_beats: int
    actual_paid_beats: int
    ratio: float
    policy_version: str = DRAFT_PROFILE_VERSION

    def is_key(self, beat_id: str) -> bool:
        return beat_id in self.key_beat_ids

    def snapshot(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "ratio": self.ratio,
            "target_paid_beats": self.target_paid_beats,
            "actual_paid_beats": self.actual_paid_beats,
            "key_beat_ids": sorted(self.key_beat_ids),
        }


def select_draft_key_beats(
    plan: VisualPlan,
    paid_visual_ratio: float = 0.3,
) -> DraftBeatSelection:
    """Select visual story anchors without changing the semantic plan."""
    if not 0.05 <= paid_visual_ratio <= 1:
        raise ValueError("draft paid visual ratio must be between 0.05 and 1")
    beats = plan.visual_beats
    target = max(1, math.ceil(len(beats) * paid_visual_ratio))
    scores: dict[str, float] = {}
    mandatory: set[str] = set()
    seen_locations: set[str] = set()
    seen_characters: set[str] = set()
    seen_objects: set[str] = set()
    for position, beat in enumerate(beats):
        score = 0.0
        if position in {0, len(beats) - 1}:
            mandatory.add(beat.id)
            score += 100
        if beat.location_id not in seen_locations:
            mandatory.add(beat.id)
            score += 70
            seen_locations.add(beat.location_id)
        introduced_characters = set(beat.characters_visible) - seen_characters
        introduced_objects = set(beat.important_objects) - seen_objects
        if introduced_characters:
            mandatory.add(beat.id)
            score += 35 + 5 * len(introduced_characters)
        if introduced_objects:
            mandatory.add(beat.id)
            score += 30 + 4 * len(introduced_objects)
        seen_characters.update(beat.characters_visible)
        seen_objects.update(beat.important_objects)
        if beat.progressive_change is not None:
            mandatory.add(beat.id)
            score += 65
        if beat.safety_geography is not None:
            score += 25
        combined = (
            f"{beat.visual_purpose} {beat.what_viewer_should_understand} "
            f"{beat.change_from_previous_beat}"
        ).lower()
        if any(term in combined for term in _CRITICAL_TERMS):
            mandatory.add(beat.id)
            score += 80
        if beat.preferred_visual_operation in {
            VisualOperation.NEW_IMAGE,
            VisualOperation.EDIT_EXISTING,
        }:
            score += 15
        scores[beat.id] = score
    selected = set(mandatory)
    for beat_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        if len(selected) >= target:
            break
        selected.add(beat_id)
    return DraftBeatSelection(
        key_beat_ids=frozenset(selected),
        target_paid_beats=target,
        actual_paid_beats=len(selected),
        ratio=paid_visual_ratio,
    )


def draft_free_operation(beat: VisualBeat) -> VisualOperation:
    if beat.overlay_description is not None:
        return VisualOperation.OVERLAY
    if beat.camera_movement is CameraMovement.STATIC:
        return VisualOperation.REUSE
    return VisualOperation.TRANSFORM
