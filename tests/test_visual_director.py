"""Tests for safe whole-story visual planning."""

import asyncio
import json

import pytest

from app.errors import VisualDirectorError
from app.generators.visual_director import VisualDirector
from app.models.visual_plan import VisualOperation


def valid_plan_payload() -> dict[str, object]:
    return {
        "story_summary": "A mine accident closes the main escape route.",
        "visual_strategy": "Keep the shaft layout stable as damage accumulates.",
        "characters": [
            {
                "id": "miners",
                "name": "Miners",
                "description": "A small group wearing helmets",
            }
        ],
        "locations": [
            {
                "id": "shaft",
                "name": "Main shaft",
                "description": "A deep vertical mine shaft",
                "spatial_layout": "Surface above, miners below, side tunnel at bottom",
            }
        ],
        "important_objects": [
            {
                "id": "ladder",
                "name": "Ladder",
                "description": "The main vertical escape ladder",
                "story_function": "Initially provides the route to the surface",
            }
        ],
        "recurring_environments": [
            {
                "id": "shaft_environment",
                "location_id": "shaft",
                "continuity_requirements": "Keep surface, ladder and tunnel aligned",
            }
        ],
        "possible_master_scenes": [
            {
                "id": "shaft_master",
                "location_id": "shaft",
                "description": "Wide cutaway showing the complete shaft",
                "environment_geometry": "Vertical shaft with a side tunnel at bottom",
                "recurring_object_positions": "Ladder runs along the right wall",
                "color_palette": "Dark stone, amber lamps, muted workwear",
                "basic_composition": "Surface at top and miners small at bottom",
                "characters_visible": ["miners"],
                "important_objects": ["ladder"],
            }
        ],
        "visual_beats": [
            {
                "id": "beat_1",
                "narration_segment": "The miners worked far below the surface.",
                "visual_purpose": "Establish the mine geography.",
                "what_viewer_should_understand": "The surface is far above the miners.",
                "location_id": "shaft",
                "characters_visible": ["miners"],
                "important_objects": ["ladder"],
                "camera_framing": "WIDE",
                "camera_view": "Wide vertical cutaway",
                "framing_reason": "Establish distance and the complete escape route.",
                "camera_movement": "SUBTLE_ZOOM",
                "geography_established_by": None,
                "master_scene_id": "shaft_master",
                "physical_state": "Ladder connects miners to the open surface route.",
                "progressive_change": None,
                "safety_geography": {
                    "current_position": "Miners at the bottom of the shaft",
                    "exit_or_safe_area": "Surface opening",
                    "distance_or_scale": "The miners appear small far below",
                    "obstacle_between": "No obstacle yet; ladder is continuous",
                },
                "anticipated_consequence": "The ladder is visibly the only main route.",
                "change_from_previous_beat": "Initial spatial establishment.",
                "information_added_beyond_narration": "Shows vertical scale and route.",
                "preferred_visual_operation": "NEW_IMAGE",
                "source_visual_id": None,
                "overlay_description": None,
                "estimated_duration_seconds": 4.5,
            },
            {
                "id": "beat_2",
                "narration_segment": "The ladder collapsed.",
                "visual_purpose": "Show the loss of the main route.",
                "what_viewer_should_understand": "The same route is now unusable.",
                "location_id": "shaft",
                "characters_visible": ["miners"],
                "important_objects": ["ladder"],
                "camera_framing": "CLOSE",
                "camera_view": "Same cutaway, closer on the broken middle section",
                "framing_reason": "Make the exact failed connection readable.",
                "camera_movement": "HIGHLIGHT",
                "geography_established_by": "beat_1",
                "master_scene_id": "shaft_master",
                "physical_state": "The ladder is broken and no longer spans the shaft.",
                "progressive_change": {
                    "subject_id": "ladder",
                    "previous_state": "Connected from miners to surface",
                    "current_state": "Broken in the middle",
                    "progression": "Open escape route becomes blocked",
                },
                "safety_geography": {
                    "current_position": "Miners below the break",
                    "exit_or_safe_area": "Surface opening above",
                    "distance_or_scale": "Most of the shaft remains above them",
                    "obstacle_between": "Missing ladder section",
                },
                "anticipated_consequence": None,
                "change_from_previous_beat": "The established ladder connection breaks.",
                "information_added_beyond_narration": "Locates the break between miners and safety.",
                "preferred_visual_operation": "EDIT_EXISTING",
                "source_visual_id": "beat_1",
                "overlay_description": None,
                "estimated_duration_seconds": 4.0,
            },
        ],
    }


class FakePlanningClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompt: str | None = None

    async def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.response


def test_director_sends_complete_narration_and_returns_validated_plan() -> None:
    narration = "First the miners descend. Later their ladder collapses."
    client = FakePlanningClient(json.dumps(valid_plan_payload()))

    plan = asyncio.run(VisualDirector(client).create_plan(narration))

    assert client.prompt is not None
    assert narration in client.prompt
    assert "do not write image\ngeneration prompts" in client.prompt
    assert "ESTABLISH GEOGRAPHY BEFORE DETAIL" in client.prompt
    assert "VISUAL ANTICIPATION" in client.prompt
    assert "DISTANCE FROM SAFETY" in client.prompt
    assert "REUSE STRONG IMAGES" in client.prompt
    assert "ADD INFORMATION" in client.prompt
    assert len(plan.visual_beats) == 2
    assert plan.visual_beats[1].preferred_visual_operation is VisualOperation.EDIT_EXISTING


def test_director_rejects_blank_narration_without_calling_client() -> None:
    client = FakePlanningClient(json.dumps(valid_plan_payload()))

    with pytest.raises(ValueError, match="Narration must not be empty"):
        asyncio.run(VisualDirector(client).create_plan("   "))

    assert client.prompt is None


def test_director_fails_safely_for_invalid_json() -> None:
    client = FakePlanningClient("```json\n{}\n```")

    with pytest.raises(VisualDirectorError, match="invalid structured visual plan"):
        asyncio.run(VisualDirector(client).create_plan("Complete narration"))


def test_director_fails_safely_for_invalid_schema() -> None:
    payload = valid_plan_payload()
    payload["visual_beats"] = []
    client = FakePlanningClient(json.dumps(payload))

    with pytest.raises(VisualDirectorError, match="invalid structured visual plan"):
        asyncio.run(VisualDirector(client).create_plan("Complete narration"))


def test_director_rejects_unknown_and_forward_source_references() -> None:
    payload = valid_plan_payload()
    beats = payload["visual_beats"]
    assert isinstance(beats, list)
    assert isinstance(beats[0], dict)
    beats[0]["preferred_visual_operation"] = "REUSE"
    beats[0]["source_visual_id"] = "beat_2"
    client = FakePlanningClient(json.dumps(payload))

    with pytest.raises(VisualDirectorError, match="invalid structured visual plan"):
        asyncio.run(VisualDirector(client).create_plan("Complete narration"))


def test_director_rejects_close_view_without_established_geography() -> None:
    payload = valid_plan_payload()
    beats = payload["visual_beats"]
    assert isinstance(beats, list)
    assert isinstance(beats[1], dict)
    beats[1]["geography_established_by"] = None
    client = FakePlanningClient(json.dumps(payload))

    with pytest.raises(VisualDirectorError, match="invalid structured visual plan"):
        asyncio.run(VisualDirector(client).create_plan("Complete narration"))


def test_director_requires_master_id_in_recurring_environment() -> None:
    payload = valid_plan_payload()
    beats = payload["visual_beats"]
    assert isinstance(beats, list)
    assert isinstance(beats[0], dict)
    beats[0]["master_scene_id"] = None
    client = FakePlanningClient(json.dumps(payload))

    with pytest.raises(VisualDirectorError, match="invalid structured visual plan"):
        asyncio.run(VisualDirector(client).create_plan("Complete narration"))


def test_director_hides_provider_error_details() -> None:
    class FailingClient:
        async def generate(self, prompt: str) -> str:
            raise RuntimeError("secret provider response")

    with pytest.raises(VisualDirectorError) as exc_info:
        asyncio.run(VisualDirector(FailingClient()).create_plan("Complete narration"))

    assert str(exc_info.value) == "Visual planning provider failed"
    assert "secret" not in str(exc_info.value)
