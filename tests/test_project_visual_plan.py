"""Integration tests for durable Project visual planning."""

import asyncio
import json
import logging
from collections.abc import Iterator, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from app.database import (
    create_session_factory,
    create_sqlite_engine,
    init_database,
)
from app.errors import StaleProjectVisualPlanError, VisualDirectorError
from app.pipeline.visual_operation_engine import VisualDecisionEvidence
from app.provider_capabilities import ImageProviderCapabilities
from app.repositories import (
    create_project,
    create_scene,
    delete_project,
    get_project_visual_plan_record,
    list_project_visual_execution_plans,
    list_scenes,
    update_project,
)
from app.services.visual_operations import resolve_project_visual_operations
from app.services.visual_planning import (
    VISUAL_DIRECTOR_VERSION,
    VISUAL_PLAN_SCHEMA_VERSION,
    VisualPlanStatus,
    create_project_visual_plan,
    hash_story_text,
    load_project_visual_plan,
    load_project_visual_plan_state,
    require_current_project_visual_plan,
)


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    return tmp_path / "app.db"


@pytest.fixture
def session(database_path: Path) -> Iterator[Session]:
    engine = create_sqlite_engine(database_path)
    init_database(engine)
    session_factory = create_session_factory(engine)
    with session_factory() as database_session:
        yield database_session
    engine.dispose()


class FakePlanningClient:
    def __init__(
        self,
        response: str,
        *,
        failure: Exception | None = None,
    ) -> None:
        self.response = response
        self.failure = failure
        self.calls = 0
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if self.failure is not None:
            raise self.failure
        return self.response


def _valid_plan_payload() -> dict[str, Any]:
    return {
        "story_summary": "A mine ladder fails and blocks the route.",
        "visual_strategy": "Keep one recognizable shaft while its state changes.",
        "characters": [
            {
                "id": "miners",
                "name": "Miners",
                "description": "Three workers in simple helmets",
            }
        ],
        "locations": [
            {
                "id": "main_shaft",
                "name": "Main shaft",
                "description": "A deep vertical mine shaft",
                "spatial_layout": "Surface above, miners below, ladder on right wall",
            }
        ],
        "important_objects": [
            {
                "id": "ladder",
                "name": "Ladder",
                "description": "The main vertical escape ladder",
                "story_function": "Connects the miners to the surface",
            }
        ],
        "recurring_environments": [
            {
                "id": "shaft_environment",
                "location_id": "main_shaft",
                "continuity_requirements": "Keep shaft and ladder positions stable",
            }
        ],
        "possible_master_scenes": [
            {
                "id": "shaft_master",
                "location_id": "main_shaft",
                "description": "Wide cutaway of the complete shaft",
                "environment_geometry": "One straight shaft from tunnel to surface",
                "recurring_object_positions": "Ladder follows the right wall",
                "color_palette": "Muted grey, ochre and blue",
                "basic_composition": "Surface at top and miners small at bottom",
                "characters_visible": ["miners"],
                "important_objects": ["ladder"],
            }
        ],
        "visual_beats": [
            {
                "id": "beat_1",
                "narration_segment": "The miners worked far below the surface.",
                "visual_purpose": "Establish geography and distance from safety.",
                "what_viewer_should_understand": "The exit is far above the miners.",
                "location_id": "main_shaft",
                "characters_visible": ["miners"],
                "important_objects": ["ladder"],
                "camera_framing": "WIDE",
                "camera_view": "Full vertical side cutaway",
                "framing_reason": "The complete route must be clear before detail.",
                "camera_movement": "SUBTLE_ZOOM",
                "geography_established_by": None,
                "master_scene_id": "shaft_master",
                "physical_state": "The ladder is intact and the route is open.",
                "progressive_change": None,
                "safety_geography": {
                    "current_position": "Bottom tunnel",
                    "exit_or_safe_area": "Surface opening",
                    "distance_or_scale": "Miners appear small below",
                    "obstacle_between": "No obstacle yet",
                },
                "anticipated_consequence": "The ladder is visibly the only route.",
                "visual_focus": "Distance between miners and surface",
                "must_not_show": ["damage", "debris"],
                "change_from_previous_beat": "Initial state.",
                "information_added_beyond_narration": "Shows scale and route geometry.",
                "preferred_visual_operation": "NEW_IMAGE",
                "source_visual_id": None,
                "overlay_description": None,
                "estimated_duration_seconds": 5.5,
            }
        ],
    }


def _create_project(session: Session, story_text: str = "Complete mine story"):
    return create_project(
        session,
        name="Mine story",
        story_text=story_text,
    )


def _two_beat_plan_payload() -> dict[str, Any]:
    payload = _valid_plan_payload()
    second = deepcopy(payload["visual_beats"][0])
    second.update(
        {
            "id": "beat_2",
            "narration_segment": "The camera moves closer to the same route.",
            "visual_purpose": "Show a closer composition with continuity.",
            "what_viewer_should_understand": "This is the same mine shaft.",
            "camera_framing": "MEDIUM",
            "camera_view": "Closer view of the lower shaft",
            "framing_reason": "The route was established in the prior wide shot.",
            "camera_movement": "STATIC",
            "geography_established_by": "beat_1",
            "change_from_previous_beat": "Only the composition changes.",
            "preferred_visual_operation": "REFERENCE_GENERATION",
            "source_visual_id": "beat_1",
            "estimated_duration_seconds": 4,
        }
    )
    payload["visual_beats"].append(second)
    return payload


def test_project_story_creates_and_persists_validated_visual_plan(
    session: Session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    project = _create_project(session)
    client = FakePlanningClient(json.dumps(_valid_plan_payload()))

    with caplog.at_level(logging.INFO, logger="app.services.visual_planning"):
        plan = asyncio.run(
            create_project_visual_plan(session, project.id, client)
        )

    record = get_project_visual_plan_record(session, project.id)
    assert record is not None
    assert record.project_id == project.id
    assert record.schema_version == VISUAL_PLAN_SCHEMA_VERSION
    assert record.visual_director_version == VISUAL_DIRECTOR_VERSION
    assert record.story_text_hash == hash_story_text(project.story_text)
    assert record.plan_json == plan.model_dump(mode="json")
    assert record.created_at.tzinfo is not None
    assert record.updated_at.tzinfo is not None
    assert {item.visual_planning_event for item in caplog.records} >= {
        "planning_start",
        "planning_success",
        "persistence_success",
    }
    assert all(item.project_id == project.id for item in caplog.records)


def test_story_change_marks_plan_stale_without_deleting_it(
    session: Session,
) -> None:
    project = _create_project(session, "Original complete story")
    original = asyncio.run(
        create_project_visual_plan(
            session,
            project.id,
            FakePlanningClient(json.dumps(_valid_plan_payload())),
        )
    )
    before = get_project_visual_plan_record(session, project.id)
    assert before is not None
    original_record_id = before.id
    original_plan_json = deepcopy(before.plan_json)

    current = load_project_visual_plan_state(session, project.id)
    assert current is not None
    assert current.status is VisualPlanStatus.CURRENT
    assert current.plan == original

    update_project(session, project.id, story_text="Changed complete story")

    stale = load_project_visual_plan_state(session, project.id)
    assert stale is not None
    assert stale.status is VisualPlanStatus.STALE
    assert stale.record_id == original_record_id
    assert stale.plan.model_dump(mode="json") == original_plan_json
    with pytest.raises(StaleProjectVisualPlanError, match="story changed"):
        require_current_project_visual_plan(session, project.id)


def test_database_init_adds_story_hash_to_stage_one_schema(
    database_path: Path,
) -> None:
    engine = create_sqlite_engine(database_path)
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE project_visual_plans ("
                "id VARCHAR(36) PRIMARY KEY, "
                "project_id VARCHAR(36) NOT NULL, "
                "schema_version VARCHAR(32) NOT NULL, "
                "visual_director_version VARCHAR(32) NOT NULL, "
                "plan_json JSON NOT NULL, "
                "created_at DATETIME NOT NULL, "
                "updated_at DATETIME NOT NULL)"
            )
        )

    init_database(engine)

    columns = {
        column["name"]
        for column in inspect(engine).get_columns("project_visual_plans")
    }
    assert "story_text_hash" in columns
    engine.dispose()


def test_visual_plan_reloads_unchanged_after_database_restart(
    database_path: Path,
) -> None:
    first_engine = create_sqlite_engine(database_path)
    init_database(first_engine)
    first_factory = create_session_factory(first_engine)
    with first_factory() as first_session:
        project = _create_project(first_session)
        expected = asyncio.run(
            create_project_visual_plan(
                first_session,
                project.id,
                FakePlanningClient(json.dumps(_valid_plan_payload())),
            )
        ).model_dump(mode="json")
        project_id = project.id
    first_engine.dispose()

    second_engine = create_sqlite_engine(database_path)
    init_database(second_engine)
    second_factory = create_session_factory(second_engine)
    with second_factory() as second_session:
        reloaded = load_project_visual_plan(second_session, project_id)
        assert reloaded is not None
        assert reloaded.model_dump(mode="json") == expected
    second_engine.dispose()


def test_provider_specific_resolution_persists_across_restart(
    database_path: Path,
) -> None:
    first_engine = create_sqlite_engine(database_path)
    init_database(first_engine)
    first_factory = create_session_factory(first_engine)
    with first_factory() as first_session:
        project = _create_project(first_session)
        asyncio.run(
            create_project_visual_plan(
                first_session,
                project.id,
                FakePlanningClient(json.dumps(_valid_plan_payload())),
            )
        )
        resolution = resolve_project_visual_operations(first_session, project.id)
        resolution_id = resolution.id
        project_id = project.id
        assert resolution.provider == "seedream"
        assert resolution.model == "seedream-5-0-260128"
        assert resolution.capability_snapshot["reference_generation"] is True
        assert resolution.capability_snapshot["max_reference_images"] == 14
        assert [item.resolved_operation for item in resolution.decisions] == [
            "NEW_IMAGE"
        ]
    first_engine.dispose()

    second_engine = create_sqlite_engine(database_path)
    init_database(second_engine)
    second_factory = create_session_factory(second_engine)
    with second_factory() as second_session:
        reloaded = resolve_project_visual_operations(second_session, project_id)
        assert reloaded.id == resolution_id
        assert reloaded.decisions[0].beat_id == "beat_1"
        assert reloaded.decisions[0].reason
    second_engine.dispose()


def test_provider_change_creates_a_new_resolution_without_overwriting_history(
    session: Session,
) -> None:
    project = _create_project(session)
    asyncio.run(
        create_project_visual_plan(
            session,
            project.id,
            FakePlanningClient(json.dumps(_valid_plan_payload())),
        )
    )
    seedream = resolve_project_visual_operations(session, project.id)

    update_project(session, project.id, image_provider="qwen")
    qwen = resolve_project_visual_operations(session, project.id)

    assert qwen.id != seedream.id
    assert qwen.provider == "qwen"
    assert qwen.model == "qwen-image-3.0"
    history = list_project_visual_execution_plans(session, project.id)
    assert [item.provider for item in history] == ["seedream", "qwen"]


def test_unsupported_reference_fallback_and_reasons_are_persisted(
    session: Session,
) -> None:
    project = _create_project(session)
    asyncio.run(
        create_project_visual_plan(
            session,
            project.id,
            FakePlanningClient(json.dumps(_two_beat_plan_payload())),
        )
    )

    class TextOnlyProvider:
        capabilities = ImageProviderCapabilities()
        model = "text-only-model"

        async def generate(self, prompt: str) -> str:
            raise AssertionError(f"image generation must not run: {prompt}")

    def resolver(
        name: str,
        config: Mapping[str, Any] | None,
    ) -> TextOnlyProvider:
        del name, config
        return TextOnlyProvider()

    resolution = resolve_project_visual_operations(
        session,
        project.id,
        provider_resolver=resolver,
    )
    second = resolution.decisions[1]

    assert second.preferred_operation == "REFERENCE_GENERATION"
    assert second.resolved_operation == "NEW_IMAGE"
    assert second.fallback_used is True
    assert second.fallback_from == "REFERENCE_GENERATION"
    assert "provider does not support reference generation" in second.reason


def test_reference_availability_change_creates_new_resolution(
    session: Session,
    tmp_path: Path,
) -> None:
    project = _create_project(session)
    asyncio.run(
        create_project_visual_plan(
            session,
            project.id,
            FakePlanningClient(json.dumps(_two_beat_plan_payload())),
        )
    )
    without_asset = resolve_project_visual_operations(session, project.id)
    beat_image = tmp_path / "beat-1.png"
    beat_image.write_bytes(b"image")

    with_asset = resolve_project_visual_operations(
        session,
        project.id,
        available_visuals={"beat_1": str(beat_image)},
    )

    assert with_asset.id != without_asset.id
    assert with_asset.resolution_revision != without_asset.resolution_revision
    assert with_asset.decision_input_snapshot["references"]["visuals"][
        "beat_1"
    ]["available"] is True


def test_capability_change_creates_new_resolution(
    session: Session,
) -> None:
    project = _create_project(session)
    asyncio.run(
        create_project_visual_plan(
            session,
            project.id,
            FakePlanningClient(json.dumps(_valid_plan_payload())),
        )
    )

    class FakeProvider:
        def __init__(self, capabilities: ImageProviderCapabilities) -> None:
            self.capabilities = capabilities
            self.model = "fake-model"

        async def generate(self, prompt: str) -> str:
            raise AssertionError(f"image generation must not run: {prompt}")

    capabilities = ImageProviderCapabilities()

    def resolver(
        name: str,
        config: Mapping[str, Any] | None,
    ) -> FakeProvider:
        del name, config
        return FakeProvider(capabilities)

    original = resolve_project_visual_operations(
        session,
        project.id,
        provider_resolver=resolver,
    )
    capabilities = ImageProviderCapabilities(reference_generation=True)
    changed = resolve_project_visual_operations(
        session,
        project.id,
        provider_resolver=resolver,
    )

    assert changed.id != original.id
    assert original.capability_snapshot["reference_generation"] is False
    assert changed.capability_snapshot["reference_generation"] is True


def test_same_semantic_plan_resolves_differently_for_provider_capabilities(
    session: Session,
) -> None:
    project = _create_project(session)
    asyncio.run(
        create_project_visual_plan(
            session,
            project.id,
            FakePlanningClient(json.dumps(_two_beat_plan_payload())),
        )
    )

    class FakeProvider:
        def __init__(self, provider_name: str) -> None:
            self.capabilities = ImageProviderCapabilities(
                reference_generation=provider_name == "qwen"
            )
            self.model = f"{provider_name}-model"

        async def generate(self, prompt: str) -> str:
            raise AssertionError(f"image generation must not run: {prompt}")

    def resolver(
        name: str,
        config: Mapping[str, Any] | None,
    ) -> FakeProvider:
        del config
        return FakeProvider(name)

    decision_evidence = {
        "beat_2": VisualDecisionEvidence(
            substantially_different_composition=True,
            new_image_improves_understanding=True,
        )
    }
    seedream = resolve_project_visual_operations(
        session,
        project.id,
        provider_resolver=resolver,
        evidence_by_beat=decision_evidence,
    )
    update_project(session, project.id, image_provider="qwen")
    qwen = resolve_project_visual_operations(
        session,
        project.id,
        provider_resolver=resolver,
        evidence_by_beat=decision_evidence,
    )

    assert seedream.decisions[1].resolved_operation == "NEW_IMAGE"
    assert qwen.decisions[1].resolved_operation == "REFERENCE_GENERATION"


def test_stale_plan_is_not_accepted_for_operation_resolution(
    session: Session,
) -> None:
    project = _create_project(session)
    asyncio.run(
        create_project_visual_plan(
            session,
            project.id,
            FakePlanningClient(json.dumps(_valid_plan_payload())),
        )
    )
    update_project(session, project.id, story_text="A different complete story")

    with pytest.raises(StaleProjectVisualPlanError, match="story changed"):
        resolve_project_visual_operations(session, project.id)

    assert list_project_visual_execution_plans(session, project.id) == []


@pytest.mark.parametrize(
    ("response", "failure"),
    [
        ("not-json", None),
        (json.dumps({"story_summary": "Incomplete"}), None),
        ("", TimeoutError("provider secret must not be logged")),
    ],
    ids=["malformed-json", "invalid-schema", "provider-timeout"],
)
def test_failed_regeneration_keeps_previous_valid_plan(
    session: Session,
    caplog: pytest.LogCaptureFixture,
    response: str,
    failure: Exception | None,
) -> None:
    project = _create_project(session)
    asyncio.run(
        create_project_visual_plan(
            session,
            project.id,
            FakePlanningClient(json.dumps(_valid_plan_payload())),
        )
    )
    before = get_project_visual_plan_record(session, project.id)
    assert before is not None
    original_id = before.id
    original_json = deepcopy(before.plan_json)
    original_updated_at = before.updated_at

    with (
        caplog.at_level(
            logging.WARNING,
            logger="app.services.visual_planning",
        ),
        pytest.raises(VisualDirectorError),
    ):
        asyncio.run(
            create_project_visual_plan(
                session,
                project.id,
                FakePlanningClient(response, failure=failure),
            )
        )

    session.expire_all()
    after = get_project_visual_plan_record(session, project.id)
    assert after is not None
    assert after.id == original_id
    assert after.plan_json == original_json
    assert after.updated_at == original_updated_at
    assert "provider secret must not be logged" not in caplog.text


def test_project_deletion_cascades_to_visual_plan(session: Session) -> None:
    project = _create_project(session)
    asyncio.run(
        create_project_visual_plan(
            session,
            project.id,
            FakePlanningClient(json.dumps(_valid_plan_payload())),
        )
    )
    assert get_project_visual_plan_record(session, project.id) is not None

    assert delete_project(session, project.id) is True
    assert get_project_visual_plan_record(session, project.id) is None


def test_empty_persisted_story_does_not_call_visual_director(
    session: Session,
) -> None:
    project = _create_project(session)
    session.execute(
        text("UPDATE projects SET story_text = '   ' WHERE id = :project_id"),
        {"project_id": project.id},
    )
    session.commit()
    session.expire_all()
    client = FakePlanningClient(json.dumps(_valid_plan_payload()))

    with pytest.raises(ValueError, match="story_text must not be empty"):
        asyncio.run(create_project_visual_plan(session, project.id, client))

    assert client.calls == 0
    assert get_project_visual_plan_record(session, project.id) is None


def test_complete_project_story_is_passed_to_planning_client(
    session: Session,
) -> None:
    story = "BEGIN complete story. Middle physical change. END complete story."
    project = _create_project(session, story)
    client = FakePlanningClient(json.dumps(_valid_plan_payload()))

    asyncio.run(create_project_visual_plan(session, project.id, client))

    assert client.calls == 1
    assert story in client.prompts[0]
    assert f"<narration>\n{story}\n</narration>" in client.prompts[0]


def test_visual_plan_is_additive_and_manual_scenes_remain_unchanged(
    session: Session,
) -> None:
    project = _create_project(session)
    manual_scene = create_scene(
        session,
        project_id=project.id,
        text="Manual narration",
        image_prompt="Manual image prompt",
    )

    asyncio.run(
        create_project_visual_plan(
            session,
            project.id,
            FakePlanningClient(json.dumps(_valid_plan_payload())),
        )
    )

    scenes = list_scenes(session, project.id)
    assert len(scenes) == 1
    assert scenes[0].id == manual_scene.id
    assert scenes[0].text == "Manual narration"
    assert scenes[0].image_prompt == "Manual image prompt"
