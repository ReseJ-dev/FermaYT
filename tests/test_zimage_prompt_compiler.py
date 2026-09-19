"""Regression tests for semantic-preserving Z-Image prompt compilation."""

from copy import deepcopy

import pytest

from app.errors import ImagePromptBuildError
from app.models.visual_plan import VisualBeat, VisualOperation, VisualPlan
from app.services.zimage_prompt_compiler import compile_zimage_semantic_prompt
from tests.test_visual_asset_execution import _plan_payload


def _ground_moves_plan() -> tuple[VisualPlan, VisualBeat]:
    payload = deepcopy(_plan_payload())
    payload["locations"][0].update(
        {
            "name": "Side tunnel",
            "description": "Underground mine side tunnel",
            "spatial_layout": "support timbers line both walls around the miner",
        }
    )
    payload["characters"][0].update(
        {
            "name": "Miner",
            "description": "orange helmet and reflective dark work clothes",
        }
    )
    payload["important_objects"].extend(
        [
            {
                "id": "support_timbers",
                "name": "Support timbers",
                "description": "wooden structural supports visibly shifting",
                "story_function": "show structural movement",
            },
            {
                "id": "small_debris",
                "name": "Small debris",
                "description": "small rocks jumping near the miner boots",
                "story_function": "make ground vibration visible",
            },
        ]
    )
    beat = payload["visual_beats"][4]
    beat.update(
        {
            "id": "beat_05_ground_moves",
            "what_viewer_should_understand": (
                "ground vibration makes small debris jump near the miner boots"
            ),
            "characters_visible": ["miners"],
            "important_objects": ["support_timbers", "small_debris"],
            "camera_framing": "DETAIL",
            "camera_view": "low-angle DETAIL shot on boots and vibrating floor",
            "physical_state": (
                "ground vibration; small debris jumps; support timbers shift"
            ),
            "main_visual_idea": "Ground vibration makes small debris jump.",
            "visible_physical_state": (
                "ground vibration; small debris jumps; support timbers shift"
            ),
            "essential_environment_cues": ["narrow side tunnel with timber walls"],
            "optional_entities_to_omit": [],
            "character_count_target": 1,
            "background_complexity": "SPARSE",
            "complexity_budget": {
                "max_main_subjects": 1,
                "max_supporting_objects": 2,
                "max_environment_concepts": 1,
                "max_main_actions": 1,
            },
            "change_from_previous_beat": (
                "stable side tunnel becomes the first visible structural movement"
            ),
            "preferred_visual_operation": "NEW_IMAGE",
            "source_visual_id": None,
            "overlay_description": None,
        }
    )
    plan = VisualPlan.model_validate(payload)
    return plan, plan.visual_beats[4]


def _assert_ground_move_semantics(prompt: str) -> None:
    lowered = prompt.lower()
    for concept in (
        "miner",
        "side tunnel",
        "support timbers",
        "ground vibration",
        "small debris",
        "low-angle detail",
    ):
        assert concept in lowered


def test_real_ground_moves_beat_preserves_all_required_semantics() -> None:
    plan, beat = _ground_moves_plan()

    compiled = compile_zimage_semantic_prompt(
        plan,
        target=beat,
        operation=VisualOperation.NEW_IMAGE,
        manual_scene_override=None,
        qa_correction=None,
    )

    assert len(compiled.prompt) <= 800
    _assert_ground_move_semantics(compiled.prompt)
    assert compiled.report["final_length"] == len(compiled.prompt)
    assert set(compiled.report["preserved_fields"]) >= {
        "physical_state_action",
        "characters",
        "location",
        "critical_objects",
        "camera_composition",
        "change_from_previous",
        "style",
    }
    assert compiled.prompt.index("Subject:") < compiled.prompt.index("Action:")
    assert compiled.prompt.index("Action:") < compiled.prompt.index("Objects:")
    assert compiled.prompt.index("Objects:") < compiled.prompt.index("Location:")
    assert compiled.prompt.index("Location:") < compiled.prompt.index("Camera:")
    assert compiled.prompt.index("Camera:") < compiled.prompt.index("Style:")


def test_ground_moves_qa_retry_keeps_original_semantics_plus_delta() -> None:
    plan, beat = _ground_moves_plan()
    correction = "Make the floor vibration and timber shift more obvious"

    retry = compile_zimage_semantic_prompt(
        plan,
        target=beat,
        operation=VisualOperation.NEW_IMAGE,
        manual_scene_override=None,
        qa_correction=correction,
    )

    assert len(retry.prompt) <= 800
    _assert_ground_move_semantics(retry.prompt)
    assert correction.lower() in retry.prompt.lower()
    assert "qa_correction" in retry.report["preserved_fields"]


def test_zimage_compiles_visible_state_instead_of_full_story_inventory() -> None:
    plan, _ = _ground_moves_plan()
    payload = plan.model_dump(mode="json")
    beat_payload = payload["visual_beats"][4]
    beat_payload["physical_state"] = (
        "Ground vibration plus a distant ventilation fan and an unused rail cart."
    )
    beat_payload["main_visual_idea"] = "Small debris jumps as the ground vibrates."
    beat_payload["visible_physical_state"] = (
        "Small debris jumps beside the miner boots and support timbers shift."
    )
    beat_payload["optional_entities_to_omit"] = []
    plan = VisualPlan.model_validate(payload)

    compiled = compile_zimage_semantic_prompt(
        plan,
        target=plan.visual_beats[4],
        operation=VisualOperation.NEW_IMAGE,
        manual_scene_override=None,
        qa_correction=None,
    )

    lowered = compiled.prompt.lower()
    assert "ground vibrat" in lowered
    assert "small debris" in lowered
    assert "support timbers" in lowered
    assert "ventilation fan" not in lowered
    assert "rail cart" not in lowered


def test_zimage_rejects_semantics_that_cannot_represent_required_identity() -> None:
    plan, beat = _ground_moves_plan()
    payload = plan.model_dump(mode="json")
    impossible_name = "critical-object-" + "identity" * 80
    payload["important_objects"][2]["name"] = impossible_name
    plan = VisualPlan.model_validate(payload)
    beat = plan.visual_beats[4]

    with pytest.raises(
        ImagePromptBuildError, match="PROMPT_REQUIRED_STATE_TRUNCATED"
    ):
        compile_zimage_semantic_prompt(
            plan,
            target=beat,
            operation=VisualOperation.NEW_IMAGE,
            manual_scene_override=None,
            qa_correction=None,
        )
