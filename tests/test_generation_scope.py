"""Runtime pilot-scope selection does not mutate the semantic VisualPlan."""

import pytest
from test_visual_asset_execution import _plan_payload

from app.generation_scope import GenerationScope, GenerationScopeType
from app.models.visual_plan import VisualPlan


def test_first_beats_selects_ordered_prefix_without_mutating_plan() -> None:
    plan = VisualPlan.model_validate(_plan_payload())
    original = plan.model_dump(mode="json")

    selected = GenerationScope(GenerationScopeType.FIRST_BEATS, 3).select_beat_ids(plan)

    assert selected == ("beat_1", "beat_2", "beat_3")
    assert plan.model_dump(mode="json") == original


def test_first_seconds_ends_at_first_complete_estimated_beat_boundary() -> None:
    plan = VisualPlan.model_validate(_plan_payload())

    selected = GenerationScope(
        GenerationScopeType.FIRST_SECONDS,
        10,
    ).select_beat_ids(plan)

    assert selected == ("beat_1", "beat_2", "beat_3")


def test_first_seconds_can_be_refined_to_canonical_audio_boundary() -> None:
    plan = VisualPlan.model_validate(_plan_payload())

    selected = GenerationScope(
        GenerationScopeType.FIRST_SECONDS,
        30,
    ).select_aligned_beat_ids(
        plan,
        {"beat_1": 12.0, "beat_2": 24.0, "beat_3": 36.0},
    )

    assert selected == ("beat_1", "beat_2", "beat_3")


@pytest.mark.parametrize(
    ("scope_type", "value"),
    [
        (GenerationScopeType.FIRST_SECONDS, 0),
        (GenerationScopeType.FIRST_SECONDS, float("inf")),
        (GenerationScopeType.FIRST_BEATS, -1),
        (GenerationScopeType.FIRST_BEATS, 1.5),
    ],
)
def test_invalid_partial_scope_is_rejected(
    scope_type: GenerationScopeType,
    value: float,
) -> None:
    with pytest.raises(ValueError):
        GenerationScope(scope_type, value)
