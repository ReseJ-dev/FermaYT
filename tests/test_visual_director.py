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
                "camera_view": "Wide vertical cutaway",
                "physical_state": "Ladder connects miners to the open surface route.",
                "change_from_previous_beat": "Initial spatial establishment.",
                "preferred_visual_operation": "NEW_IMAGE",
                "source_visual_id": None,
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
                "camera_view": "Same cutaway, closer on the broken middle section",
                "physical_state": "The ladder is broken and no longer spans the shaft.",
                "change_from_previous_beat": "The established ladder connection breaks.",
                "preferred_visual_operation": "EDIT_EXISTING",
                "source_visual_id": "beat_1",
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


def test_director_hides_provider_error_details() -> None:
    class FailingClient:
        async def generate(self, prompt: str) -> str:
            raise RuntimeError("secret provider response")

    with pytest.raises(VisualDirectorError) as exc_info:
        asyncio.run(VisualDirector(FailingClient()).create_plan("Complete narration"))

    assert str(exc_info.value) == "Visual planning provider failed"
    assert "secret" not in str(exc_info.value)
