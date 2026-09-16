"""Runtime-only selection of a prefix of an immutable semantic VisualPlan."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256

from app.models.visual_plan import VisualPlan


class GenerationScopeType(str, Enum):
    FULL = "FULL"
    FIRST_SECONDS = "FIRST_SECONDS"
    FIRST_BEATS = "FIRST_BEATS"
    STYLE_PREVIEW = "STYLE_PREVIEW"


GENERATION_SCOPE_VERSION = "generation_scope_v1"
PLANNING_SCOPE_VERSION = "planning_scope_v1"
DEFAULT_NARRATION_WPM = 150.0
FIRST_SECONDS_LOOKAHEAD_SECONDS = 15.0
SECONDS_PER_EXPECTED_BEAT = 4.5


@dataclass(frozen=True, slots=True)
class PlanningScope:
    """Deterministic source excerpt selected before any planning provider call."""

    type: GenerationScopeType
    requested_seconds: float | None
    requested_beats: int | None
    source_start_char: int
    source_end_char: int
    continuation_boundary_char: int | None
    source_hash: str
    is_partial: bool
    expected_beat_count: int
    narration_excerpt: str
    version: str = PLANNING_SCOPE_VERSION

    @classmethod
    def derive(
        cls,
        story_text: str,
        generation_scope: GenerationScope,
        *,
        words_per_minute: float = DEFAULT_NARRATION_WPM,
    ) -> PlanningScope:
        text = story_text.strip()
        if not text:
            raise ValueError("story_text must not be empty")
        if words_per_minute <= 0:
            raise ValueError("words_per_minute must be positive")
        words = list(re.finditer(r"\b[\w'-]+\b", text, flags=re.UNICODE))
        if generation_scope.is_full:
            return cls._build(
                generation_scope.type,
                text,
                len(text),
                None,
                requested_seconds=None,
                requested_beats=None,
                is_partial=False,
                expected_beats=max(1, math.ceil(len(words) / 21)),
            )

        words_per_second = words_per_minute / 60.0
        if generation_scope.type is GenerationScopeType.FIRST_SECONDS:
            requested_seconds = float(generation_scope.value or 0)
            requested_beats = None
            target_seconds = requested_seconds
            lookahead_seconds = FIRST_SECONDS_LOOKAHEAD_SECONDS
        else:
            requested_beats = (
                3
                if generation_scope.type is GenerationScopeType.STYLE_PREVIEW
                else int(generation_scope.value or 1)
            )
            requested_seconds = None
            target_seconds = requested_beats * SECONDS_PER_EXPECTED_BEAT
            lookahead_seconds = 10.0

        target_words = max(1, math.ceil(target_seconds * words_per_second))
        excerpt_words = max(
            target_words + 1,
            math.ceil((target_seconds + lookahead_seconds) * words_per_second),
        )
        continuation = _end_after_words(text, words, target_words)
        end = _clean_source_boundary(
            text,
            _end_after_words(text, words, excerpt_words),
        )
        is_partial = end < len(text)
        if not is_partial:
            continuation = None
        expected = (
            requested_beats
            if requested_beats is not None
            else max(1, math.ceil((target_seconds + lookahead_seconds) / SECONDS_PER_EXPECTED_BEAT))
        )
        return cls._build(
            generation_scope.type,
            text,
            end,
            continuation,
            requested_seconds=requested_seconds,
            requested_beats=requested_beats,
            is_partial=is_partial,
            expected_beats=expected,
        )

    @classmethod
    def _build(
        cls,
        scope_type: GenerationScopeType,
        text: str,
        end: int,
        continuation: int | None,
        *,
        requested_seconds: float | None,
        requested_beats: int | None,
        is_partial: bool,
        expected_beats: int,
    ) -> PlanningScope:
        excerpt = text[:end].strip()
        return cls(
            type=scope_type,
            requested_seconds=requested_seconds,
            requested_beats=requested_beats,
            source_start_char=0,
            source_end_char=end,
            continuation_boundary_char=continuation,
            source_hash=sha256(excerpt.encode()).hexdigest(),
            is_partial=is_partial,
            expected_beat_count=expected_beats,
            narration_excerpt=excerpt,
        )

    @property
    def scope_key(self) -> str:
        if not self.is_partial and self.type is GenerationScopeType.FULL:
            return "FULL"
        identity = (
            f"{self.type.value}:{self.requested_seconds}:{self.requested_beats}:"
            f"{self.source_start_char}:{self.source_end_char}:{self.source_hash}"
        )
        return sha256(identity.encode()).hexdigest()

    def max_output_tokens(self, configured_maximum: int) -> int:
        return min(configured_maximum, self.required_output_tokens())

    def required_output_tokens(self) -> int:
        """Conservative bounded capacity for the expected structured beat count."""
        dense_beats = min(self.expected_beat_count, 20)
        remaining_beats = max(self.expected_beat_count - dense_beats, 0)
        scaled = 2500 + dense_beats * 650 + remaining_beats * 350
        return max(6000, scaled)

    def metadata(self) -> dict[str, str | float | int | bool | None]:
        return {
            "scope_type": self.type.value,
            "requested_seconds": self.requested_seconds,
            "requested_beats": self.requested_beats,
            "source_start_char": self.source_start_char,
            "source_end_char": self.source_end_char,
            "continuation_boundary_char": self.continuation_boundary_char,
            "source_hash": self.source_hash,
            "is_partial": self.is_partial,
            "expected_beat_count": self.expected_beat_count,
            "version": self.version,
        }


def _end_after_words(text: str, words: list[re.Match[str]], count: int) -> int:
    if not words or count >= len(words):
        return len(text)
    return words[max(0, count - 1)].end()


def _clean_source_boundary(text: str, minimum_end: int) -> int:
    if minimum_end >= len(text):
        return len(text)
    window_end = min(len(text), minimum_end + 600)
    tail = text[minimum_end:window_end]
    sentence = re.search(r"[.!?](?:[\"'”’)]*)\s", tail)
    if sentence is not None:
        return minimum_end + sentence.end()
    paragraph = tail.find("\n\n")
    if paragraph >= 0:
        return minimum_end + paragraph
    return minimum_end


@dataclass(frozen=True, slots=True)
class GenerationScope:
    type: GenerationScopeType = GenerationScopeType.FULL
    value: float | int | None = None
    version: str = GENERATION_SCOPE_VERSION

    def __post_init__(self) -> None:
        if self.type in {
            GenerationScopeType.FULL,
            GenerationScopeType.STYLE_PREVIEW,
        }:
            if self.value is not None:
                raise ValueError(f"{self.type.value} generation scope must not have a value")
            return
        if (
            self.value is None
            or not math.isfinite(float(self.value))
            or float(self.value) <= 0
        ):
            raise ValueError("Partial generation scope requires a positive value")
        if self.type is GenerationScopeType.FIRST_BEATS and float(self.value) % 1:
            raise ValueError("FIRST_BEATS scope requires a whole number")

    @classmethod
    def full(cls) -> GenerationScope:
        return cls()

    @property
    def is_full(self) -> bool:
        return self.type is GenerationScopeType.FULL

    @property
    def is_image_only(self) -> bool:
        return self.type is GenerationScopeType.STYLE_PREVIEW

    def select_beat_ids(self, plan: VisualPlan) -> tuple[str, ...]:
        """Select a prefix using semantic durations; persisted plan is untouched."""
        beats = plan.visual_beats
        if self.is_full:
            return tuple(beat.id for beat in beats)
        if self.type is GenerationScopeType.STYLE_PREVIEW:
            count = min(3, len(beats))
        elif self.type is GenerationScopeType.FIRST_BEATS:
            count = min(int(self.value or 0), len(beats))
        else:
            target = float(self.value or 0)
            elapsed = 0.0
            count = 0
            for beat in beats:
                elapsed += beat.estimated_duration_seconds
                count += 1
                if elapsed >= target:
                    break
        return tuple(beat.id for beat in beats[: max(1, count)])

    def select_aligned_beat_ids(
        self,
        plan: VisualPlan,
        audio_end_by_beat: Mapping[str, float],
    ) -> tuple[str, ...]:
        """Refine FIRST_SECONDS against canonical narration beat boundaries."""
        if self.type is not GenerationScopeType.FIRST_SECONDS:
            return self.select_beat_ids(plan)
        target = float(self.value or 0)
        selected: list[str] = []
        for beat in plan.visual_beats:
            selected.append(beat.id)
            if audio_end_by_beat.get(beat.id, 0.0) >= target:
                break
        return tuple(selected)

    def snapshot(self) -> dict[str, str | float | int | None]:
        value: float | int | None = self.value
        if self.type is GenerationScopeType.FIRST_BEATS and value is not None:
            value = int(value)
        elif value is not None and math.isfinite(float(value)):
            value = float(value)
        return {"type": self.type.value, "value": value, "version": self.version}
