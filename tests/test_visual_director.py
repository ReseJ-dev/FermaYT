"""Tests for safe whole-story visual planning."""

import asyncio
import json
from copy import deepcopy

import pytest

from app.errors import StructuredAIProviderError, VisualDirectorError
from app.generators.visual_director import (
    VisualDirector,
    VisualPlanPacingError,
    _validate_visual_plan_pacing,
    build_visual_director_request,
)
from app.models.visual_plan import VisualOperation, VisualPlan
from app.provider_diagnostics import (
    StructuredAIProviderDiagnostic,
    find_structured_ai_provider_diagnostic,
)


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


def test_visual_director_requires_semantic_granularity_and_cost_aware_states() -> None:
    request = build_visual_director_request("A long narration about a mine route.")

    assert "new object, location, physical state, obstacle, route" in request
    assert "45-70 meaningful visible states" in request
    assert "25-40" in request
    assert "REUSE with a new crop, zoom, pan, focus" in request


def test_long_narration_with_too_few_beats_fails_pacing_validation() -> None:
    plan = VisualPlan.model_validate(valid_plan_payload())
    narration = " ".join(f"word{index}" for index in range(240))

    with pytest.raises(VisualPlanPacingError, match="at least 12"):
        _validate_visual_plan_pacing(plan, narration)


class FakePlanningClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompt: str | None = None

    async def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.response


class SequencedPlanningClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses[len(self.prompts) - 1]


def plan_with_unknown_geography(
    invalid_id: str = "beat_5_main_tunnel_wide",
) -> dict[str, object]:
    payload = valid_plan_payload()
    beats = payload["visual_beats"]
    assert isinstance(beats, list)
    assert isinstance(beats[1], dict)
    beats[1]["geography_established_by"] = invalid_id
    return payload


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
    assert "ID REFERENCE RULES AND FINAL VALIDATION CHECKLIST" in client.prompt
    assert "Every ID is non-empty and unique" in client.prompt
    assert "OVERLAY always provides overlay_description" in client.prompt
    assert "Use only keys defined by the JSON Schema" in client.prompt
    assert "EARLIER beat" in client.prompt
    assert "automated visual storytelling system" in client.prompt
    priorities = [
        "1. storytelling clarity",
        "2. spatial continuity",
        "3. visual progression",
        "4. readability",
        "5. consistent style",
        "6. generation efficiency",
        "7. image beauty",
    ]
    assert [client.prompt.index(priority) for priority in priorities] == sorted(
        client.prompt.index(priority) for priority in priorities
    )
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

    with pytest.raises(VisualDirectorError, match="visual_beats: List should have at least 1 item"):
        asyncio.run(VisualDirector(client).create_plan("Complete narration"))


def test_director_schema_diagnostic_does_not_include_generated_values() -> None:
    payload = valid_plan_payload()
    payload["unexpected_secret_field"] = "provider-generated-sensitive-value"
    client = FakePlanningClient(json.dumps(payload))

    with pytest.raises(VisualDirectorError) as exc_info:
        asyncio.run(VisualDirector(client).create_plan("Complete narration"))

    message = str(exc_info.value)
    assert "unexpected_secret_field: Extra inputs are not permitted" in message
    assert "provider-generated-sensitive-value" not in message


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


def test_director_hydrates_missing_master_id_without_repair() -> None:
    payload = valid_plan_payload()
    beats = payload["visual_beats"]
    assert isinstance(beats, list)
    assert isinstance(beats[0], dict)
    beats[0]["master_scene_id"] = None
    client = SequencedPlanningClient([json.dumps(payload)])

    plan = asyncio.run(VisualDirector(client).create_plan("Complete narration"))

    assert plan.visual_beats[0].master_scene_id == "shaft_master"
    assert len(client.prompts) == 1


def test_canonical_model_still_rejects_missing_master_id() -> None:
    payload = valid_plan_payload()
    payload["visual_beats"][0]["master_scene_id"] = None  # type: ignore[index]

    with pytest.raises(ValueError, match="must reference one of these master scenes"):
        VisualPlan.model_validate(payload)


def test_correct_master_id_is_preserved() -> None:
    client = SequencedPlanningClient([json.dumps(valid_plan_payload())])

    plan = asyncio.run(VisualDirector(client).create_plan("Complete narration"))

    assert {beat.master_scene_id for beat in plan.visual_beats} == {"shaft_master"}
    assert len(client.prompts) == 1


def test_non_mastered_recurring_environment_does_not_require_master_id() -> None:
    payload = valid_plan_payload()
    payload["possible_master_scenes"] = []
    for beat in payload["visual_beats"]:  # type: ignore[union-attr]
        beat["master_scene_id"] = None
    client = SequencedPlanningClient([json.dumps(payload)])

    plan = asyncio.run(VisualDirector(client).create_plan("Complete narration"))

    assert all(beat.master_scene_id is None for beat in plan.visual_beats)
    assert len(client.prompts) == 1


def test_wrong_master_id_is_corrected_when_location_is_unambiguous() -> None:
    payload = valid_plan_payload()
    payload["locations"].append(  # type: ignore[union-attr]
        {
            "id": "surface",
            "name": "Surface",
            "description": "Open ground above the shaft",
            "spatial_layout": "A flat clearing around the shaft entrance",
        }
    )
    wrong_master = deepcopy(payload["possible_master_scenes"][0])  # type: ignore[index]
    wrong_master.update(id="surface_master", location_id="surface")
    payload["possible_master_scenes"].append(wrong_master)  # type: ignore[union-attr]
    payload["visual_beats"][0]["master_scene_id"] = "surface_master"  # type: ignore[index]
    client = SequencedPlanningClient([json.dumps(payload)])

    plan = asyncio.run(VisualDirector(client).create_plan("Complete narration"))

    assert plan.visual_beats[0].master_scene_id == "shaft_master"
    assert len(client.prompts) == 1


def test_ambiguous_master_id_requires_targeted_repair() -> None:
    invalid = valid_plan_payload()
    alternative = deepcopy(invalid["possible_master_scenes"][0])  # type: ignore[index]
    alternative["id"] = "shaft_master_alternative"
    invalid["possible_master_scenes"].append(alternative)  # type: ignore[union-attr]
    invalid["visual_beats"][0]["master_scene_id"] = None  # type: ignore[index]
    repaired = deepcopy(invalid)
    repaired["visual_beats"][0]["master_scene_id"] = "shaft_master"  # type: ignore[index]
    client = SequencedPlanningClient([json.dumps(invalid), json.dumps(repaired)])
    events = []

    plan = asyncio.run(
        VisualDirector(client, on_repair_event=events.append).create_plan(
            "Complete narration"
        )
    )

    assert plan.visual_beats[0].master_scene_id == "shaft_master"
    assert len(client.prompts) == 2
    repair_prompt = client.prompts[1]
    assert "Affected beat id: beat_1" in repair_prompt
    assert "Recurring environment id(s):\n- shaft_environment" in repair_prompt
    assert "Currently supplied master_scene_id: null" in repair_prompt
    assert "- shaft_master\n- shaft_master_alternative" in repair_prompt
    assert "A beat in a mastered recurring environment" in repair_prompt
    assert events[0].issue == {
        "category": "MASTER_SCENE_ASSIGNMENT_REQUIRED",
        "beat_id": "beat_1",
        "recurring_environment_ids": ["shaft_environment"],
        "recurring_environment_id": "shaft_environment",
        "provided_master_scene_id": None,
        "allowed_master_scene_ids": [
            "shaft_master",
            "shaft_master_alternative",
        ],
        "invariant": (
            "A beat in a mastered recurring environment must reference a master "
            "scene belonging to that environment."
        ),
    }


def test_all_beats_in_one_mastered_environment_are_hydrated() -> None:
    payload = valid_plan_payload()
    for beat in payload["visual_beats"]:  # type: ignore[union-attr]
        beat["master_scene_id"] = None
    client = SequencedPlanningClient([json.dumps(payload)])

    plan = asyncio.run(VisualDirector(client).create_plan("Complete narration"))

    assert [beat.master_scene_id for beat in plan.visual_beats] == [
        "shaft_master",
        "shaft_master",
    ]
    assert len(client.prompts) == 1


def test_master_hydration_consumes_zero_repair_attempts() -> None:
    payload = valid_plan_payload()
    payload["visual_beats"][0]["master_scene_id"] = None  # type: ignore[index]
    events = []
    client = SequencedPlanningClient([json.dumps(payload)])
    director = VisualDirector(client, on_repair_event=events.append)

    asyncio.run(director.create_plan("Complete narration"))

    assert director.provider_requests == 1
    assert events == []


def test_director_hides_provider_error_details() -> None:
    class FailingClient:
        async def generate(self, prompt: str) -> str:
            raise RuntimeError("secret provider response")

    with pytest.raises(VisualDirectorError) as exc_info:
        asyncio.run(VisualDirector(FailingClient()).create_plan("Complete narration"))

    assert str(exc_info.value) == "Visual planning provider failed"
    assert "secret" not in str(exc_info.value)
    diagnostic = find_structured_ai_provider_diagnostic(exc_info.value)
    assert diagnostic is not None
    assert diagnostic.category == "PLANNING_UNKNOWN_ERROR"
    assert diagnostic.provider_error == "RuntimeError"


def test_director_preserves_safe_provider_diagnostic() -> None:
    expected = StructuredAIProviderDiagnostic(
        provider="dashscope",
        model="qwen-plus",
        operation="visual_planning",
        category="PLANNING_BAD_REQUEST",
        attempt=1,
        max_attempts=3,
        http_status=400,
        provider_error='{"message":"Invalid response_format"}',
        request_id="request-123",
    )

    class FailingClient:
        provider = "dashscope"
        model = "qwen-plus"

        async def generate(self, prompt: str) -> str:
            raise StructuredAIProviderError(
                "safe diagnostic",
                diagnostic=expected,
                user_summary="Visual planning provider failed",
            )

    with pytest.raises(VisualDirectorError) as error:
        asyncio.run(VisualDirector(FailingClient()).create_plan("Narration"))

    diagnostic = find_structured_ai_provider_diagnostic(error.value)
    assert diagnostic is not None
    assert diagnostic.category == "PLANNING_BAD_REQUEST"
    assert diagnostic.http_status == 400
    assert diagnostic.request_id == "request-123"
    assert diagnostic.repair_attempt == 0


def test_unknown_geography_is_repaired_to_existing_canonical_id() -> None:
    invalid = plan_with_unknown_geography()
    repaired = deepcopy(invalid)
    repaired["visual_beats"][1]["geography_established_by"] = "beat_1"  # type: ignore[index]
    client = SequencedPlanningClient(
        [json.dumps(invalid), json.dumps(repaired)]
    )

    plan = asyncio.run(VisualDirector(client).create_plan("Complete narration"))

    assert plan.visual_beats[1].geography_established_by == "beat_1"
    assert len(client.prompts) == 2
    assert "UNKNOWN_REFERENCE" in client.prompts[1]
    assert "beat_5_main_tunnel_wide" in client.prompts[1]
    assert "Repair ONLY the structural consistency" in client.prompts[1]


def test_unknown_optional_geography_is_repaired_to_null() -> None:
    invalid = plan_with_unknown_geography()
    invalid["visual_beats"][1]["camera_framing"] = "MEDIUM"  # type: ignore[index]
    repaired = deepcopy(invalid)
    repaired["visual_beats"][1]["geography_established_by"] = None  # type: ignore[index]
    client = SequencedPlanningClient(
        [json.dumps(invalid), json.dumps(repaired)]
    )

    plan = asyncio.run(VisualDirector(client).create_plan("Complete narration"))

    assert plan.visual_beats[1].geography_established_by is None
    assert len(client.prompts) == 2


def test_repair_can_add_a_genuine_missing_master_geography_definition() -> None:
    invalid = plan_with_unknown_geography("lower_shaft_master")
    repaired = deepcopy(invalid)
    new_master = deepcopy(repaired["possible_master_scenes"][0])  # type: ignore[index]
    new_master["id"] = "lower_shaft_master"
    repaired["possible_master_scenes"].append(new_master)  # type: ignore[union-attr]
    repaired["visual_beats"][1]["geography_established_by"] = "lower_shaft_master"  # type: ignore[index]
    repaired["visual_beats"][1]["master_scene_id"] = "lower_shaft_master"  # type: ignore[index]
    client = SequencedPlanningClient(
        [json.dumps(invalid), json.dumps(repaired)]
    )

    plan = asyncio.run(VisualDirector(client).create_plan("Complete narration"))

    assert {master.id for master in plan.possible_master_scenes} == {
        "shaft_master",
        "lower_shaft_master",
    }
    assert plan.visual_beats[1].geography_established_by == "lower_shaft_master"


@pytest.mark.parametrize(
    ("field", "invalid_value", "repaired_value"),
    [
        ("location_id", "unknown_location", "shaft"),
        ("characters_visible", ["unknown_character"], ["miners"]),
        ("important_objects", ["unknown_object"], ["ladder"]),
    ],
)
def test_other_unknown_registry_references_use_same_repair_flow(
    field: str,
    invalid_value: object,
    repaired_value: object,
) -> None:
    invalid = valid_plan_payload()
    invalid["visual_beats"][0][field] = invalid_value  # type: ignore[index]
    repaired = deepcopy(invalid)
    repaired["visual_beats"][0][field] = repaired_value  # type: ignore[index]
    client = SequencedPlanningClient(
        [json.dumps(invalid), json.dumps(repaired)]
    )

    plan = asyncio.run(VisualDirector(client).create_plan("Complete narration"))

    assert len(plan.visual_beats) == 2
    assert len(client.prompts) == 2


def test_invalid_repairs_stop_at_the_configured_limit() -> None:
    invalid = json.dumps(plan_with_unknown_geography())
    client = SequencedPlanningClient([invalid, invalid, invalid])

    with pytest.raises(VisualDirectorError) as exc_info:
        asyncio.run(VisualDirector(client).create_plan("Complete narration"))

    assert len(client.prompts) == 3
    assert "after 2 repair attempt(s)" in str(exc_info.value)
    assert exc_info.value.validation_category == "UNKNOWN_REFERENCE"
    assert exc_info.value.diagnostic is not None
    assert exc_info.value.diagnostic["owner_id"] == "beat_2"
    assert exc_info.value.diagnostic["field"] == "geography_established_by"
    provider_diagnostic = find_structured_ai_provider_diagnostic(exc_info.value)
    assert provider_diagnostic is not None
    assert provider_diagnostic.category == "PLANNING_REPAIR_EXHAUSTED"
    assert provider_diagnostic.validation_category == "PLANNING_REFERENCE_ERROR"
    assert provider_diagnostic.repair_attempt == 2


def test_valid_plan_does_not_call_repair() -> None:
    client = SequencedPlanningClient([json.dumps(valid_plan_payload())])

    asyncio.run(VisualDirector(client).create_plan("Complete narration"))

    assert len(client.prompts) == 1


def test_duplicate_ids_remain_invalid_after_bounded_repairs() -> None:
    invalid = valid_plan_payload()
    duplicate = deepcopy(invalid["visual_beats"][0])  # type: ignore[index]
    invalid["visual_beats"].append(duplicate)  # type: ignore[union-attr]
    raw = json.dumps(invalid)
    client = SequencedPlanningClient([raw, raw, raw])

    with pytest.raises(VisualDirectorError) as exc_info:
        asyncio.run(VisualDirector(client).create_plan("Complete narration"))

    assert exc_info.value.validation_category == "DUPLICATE_ID"
    assert len(client.prompts) == 3


def test_current_beat_id_cannot_masquerade_as_geography_id() -> None:
    invalid = plan_with_unknown_geography("beat_2")
    client = SequencedPlanningClient([json.dumps(invalid)])

    with pytest.raises(VisualDirectorError) as exc_info:
        asyncio.run(
            VisualDirector(client, max_repair_attempts=0).create_plan(
                "Complete narration"
            )
        )

    assert exc_info.value.validation_category == "UNKNOWN_REFERENCE"
    assert exc_info.value.diagnostic is not None
    assert exc_info.value.diagnostic["invalid_id"] == "beat_2"
