"""Regression tests for deterministic simple-visual generation guards."""

import pytest
from pydantic import ValidationError

from app.errors import ImagePromptBuildError
from app.generators.image_prompt import ImagePromptBuilder, build_simplified_visual_core
from app.models.visual_plan import (
    VisualOperation,
    VisualPlan,
    assess_visual_beat_complexity,
)
from tests.test_master_scene import _plan


def test_complexity_budget_violation_is_rejected() -> None:
    payload = _plan().model_dump(mode="json")
    payload["important_objects"].extend(
        {
            "id": f"object_{index}",
            "name": f"Object {index}",
            "description": "A story-critical object",
            "story_function": "Requires visual focus",
        }
        for index in range(3)
    )
    payload["visual_beats"][0]["important_objects"] = [
        "object_0",
        "object_1",
        "object_2",
    ]

    with pytest.raises(ValidationError, match="COMPLEXITY_BUDGET_EXCEEDED"):
        VisualPlan.model_validate(payload)


def test_duplicate_visible_entities_are_repaired_deterministically() -> None:
    payload = _plan().model_dump(mode="json")
    payload["visual_beats"][0]["important_objects"] = ["ladder", "ladder"]
    payload["visual_beats"][0]["essential_environment_cues"] = [
        "plain shaft wall",
        "plain shaft wall",
    ]

    plan = VisualPlan.model_validate(payload)
    beat = plan.visual_beats[0]

    assert beat.important_objects == ["ladder"]
    assert beat.essential_environment_cues == ["plain shaft wall"]
    assert not assess_visual_beat_complexity(beat).budget_exceeded


def test_multiple_independent_visual_facts_require_split() -> None:
    beat = _plan().visual_beats[0].model_copy(
        update={
            "main_visual_idea": (
                "Water reaches the cabinet, the lights fail, and the miner climbs."
            ),
            "visible_physical_state": (
                "Water reaches equipment; lights fail; miner climbs a ladder."
            ),
        }
    )

    assessment = assess_visual_beat_complexity(beat)

    assert assessment.split_required
    assert assessment.main_action_count >= 2


def test_several_phrases_for_one_visual_state_do_not_force_split() -> None:
    beat = _plan().visual_beats[0].model_copy(
        update={
            "main_visual_idea": "Water rises around the cabinet.",
            "visible_physical_state": (
                "Water rising around the cabinet, covering its base, reflecting light."
            ),
        }
    )

    assessment = assess_visual_beat_complexity(beat)

    assert not assessment.split_required
    assert assessment.main_action_count == 1


def test_omitted_inventory_cannot_reenter_through_environment_cue() -> None:
    plan = _plan()
    beat = plan.visual_beats[0].model_copy(
        update={
            "essential_environment_cues": [
                "plain shaft wall with pipes, rails, and simple stone"
            ],
            "optional_entities_to_omit": ["pipes", "rails"],
        }
    )
    plan = plan.model_copy(update={"visual_beats": [beat]})

    core = build_simplified_visual_core(plan, beat)

    assert "pipes" not in core.minimal_location.lower()
    assert "rails" not in core.minimal_location.lower()
    assert "plain shaft wall" in core.minimal_location.lower()


def test_framing_is_not_duplicated_by_environment_cues() -> None:
    plan = _plan()
    beat = plan.visual_beats[0].model_copy(
        update={
            "camera_view": "wide side perspective",
            "essential_environment_cues": ["vertical shaft in side view"],
        }
    )
    plan = plan.model_copy(update={"visual_beats": [beat]})

    core = build_simplified_visual_core(plan, beat)
    prompt = ImagePromptBuilder().build(plan, beat, VisualOperation.NEW_IMAGE)

    assert "side view" not in core.minimal_location.lower()
    assert prompt.lower().count("wide side perspective") == 1


def test_prompt_builder_rechecks_complexity_before_dispatch() -> None:
    plan = _plan()
    beat = plan.visual_beats[0].model_copy(
        update={"character_count_target": 3}
    )
    unsafe_plan = plan.model_copy(update={"visual_beats": [beat]})

    with pytest.raises(ImagePromptBuildError, match="COMPLEXITY_BUDGET_EXCEEDED"):
        ImagePromptBuilder().build(
            unsafe_plan,
            beat,
            VisualOperation.NEW_IMAGE,
        )
