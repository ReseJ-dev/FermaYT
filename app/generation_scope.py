"""Runtime-only selection of a prefix of an immutable semantic VisualPlan."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from app.models.visual_plan import VisualPlan


class GenerationScopeType(str, Enum):
    FULL = "FULL"
    FIRST_SECONDS = "FIRST_SECONDS"
    FIRST_BEATS = "FIRST_BEATS"


GENERATION_SCOPE_VERSION = "generation_scope_v1"


@dataclass(frozen=True, slots=True)
class GenerationScope:
    type: GenerationScopeType = GenerationScopeType.FULL
    value: float | int | None = None
    version: str = GENERATION_SCOPE_VERSION

    def __post_init__(self) -> None:
        if self.type is GenerationScopeType.FULL:
            if self.value is not None:
                raise ValueError("FULL generation scope must not have a value")
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

    def select_beat_ids(self, plan: VisualPlan) -> tuple[str, ...]:
        """Select a prefix using semantic durations; persisted plan is untouched."""
        beats = plan.visual_beats
        if self.is_full:
            return tuple(beat.id for beat in beats)
        if self.type is GenerationScopeType.FIRST_BEATS:
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
