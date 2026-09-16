"""Runtime pilot-scope selection does not mutate the semantic VisualPlan."""

import pytest
from test_visual_asset_execution import _plan_payload

from app.generation_scope import GenerationScope, GenerationScopeType, PlanningScope
from app.models.visual_plan import VisualPlan


def test_first_beats_selects_ordered_prefix_without_mutating_plan() -> None:
    plan = VisualPlan.model_validate(_plan_payload())
    original = plan.model_dump(mode="json")

    selected = GenerationScope(GenerationScopeType.FIRST_BEATS, 3).select_beat_ids(plan)

    assert selected == ("beat_1", "beat_2", "beat_3")
    assert plan.model_dump(mode="json") == original


def test_style_preview_has_fixed_three_beat_limit() -> None:
    payload = _plan_payload()
    payload["visual_beats"].append(
        {**payload["visual_beats"][-1], "id": "beat_extra"}
    )
    plan = VisualPlan.model_validate(payload)

    scope = GenerationScope(GenerationScopeType.STYLE_PREVIEW)

    assert scope.is_image_only is True
    assert scope.select_beat_ids(plan) == ("beat_1", "beat_2", "beat_3")
    assert scope.snapshot() == {
        "type": "STYLE_PREVIEW",
        "value": None,
        "version": "generation_scope_v1",
    }


def test_style_preview_rejects_client_supplied_limit() -> None:
    with pytest.raises(ValueError):
        GenerationScope(GenerationScopeType.STYLE_PREVIEW, 4)


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


def _long_story(word_count: int = 1441) -> str:
    return " ".join(
        f"word{index}{'.' if (index + 1) % 20 == 0 else ''}"
        for index in range(word_count)
    )


def test_first_seconds_planning_scope_is_bounded_before_provider_call() -> None:
    story = _long_story()
    thirty = PlanningScope.derive(
        story, GenerationScope(GenerationScopeType.FIRST_SECONDS, 30)
    )
    sixty = PlanningScope.derive(
        story, GenerationScope(GenerationScopeType.FIRST_SECONDS, 60)
    )
    full = PlanningScope.derive(story, GenerationScope.full())

    assert len(thirty.narration_excerpt.split()) in range(100, 122)
    assert len(thirty.narration_excerpt.split()) < 1441 // 10
    assert len(thirty.narration_excerpt) < len(sixty.narration_excerpt) < len(story)
    assert thirty.is_partial is sixty.is_partial is True
    assert full.narration_excerpt == story
    assert full.is_partial is False
    assert thirty.max_output_tokens(32768) < sixty.max_output_tokens(32768)
    assert sixty.max_output_tokens(32768) < full.max_output_tokens(32768)
