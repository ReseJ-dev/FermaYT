"""Integration tests for Stage 3 visual beat asset execution."""

import asyncio
import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session

import app.services.visual_asset_execution as execution_module
from app.database import create_session_factory, create_sqlite_engine, init_database
from app.errors import BeatVisualExecutionError
from app.models.visual_plan import VisualOperation, VisualPlan
from app.persistence import MasterSceneAsset, StyleReferenceAsset
from app.pipeline.visual_qa import VisualQAService
from app.provider_capabilities import ImageProviderCapabilities
from app.providers import ImageReference
from app.repositories import (
    create_beat_visual_result,
    create_project,
    delete_project,
    get_master_scene_asset,
    list_beat_visual_results,
    save_project_visual_plan_record,
    save_visual_execution_plan,
)
from app.services.visual_asset_execution import (
    VisualBeatAssetExecutor,
    build_visual_asset_execution_summary,
    build_visual_qa_execution_summary,
)
from app.services.visual_asset_selection import (
    VisualAssetSource,
    select_visual_references,
)
from app.services.visual_operations import resolve_project_visual_operations
from app.services.visual_planning import hash_story_text
from app.services.visual_qa import VisualQAManualOverrideService


@pytest.fixture
def session(tmp_path: Path) -> Iterator[Session]:
    engine = create_sqlite_engine(tmp_path / "app.db")
    init_database(engine)
    factory = create_session_factory(engine)
    with factory() as database_session:
        yield database_session
    engine.dispose()


class FakeImageProvider:
    capabilities = ImageProviderCapabilities(
        reference_generation=True,
        multiple_references=True,
        max_reference_images=3,
        image_editing=True,
    )
    model = "fake-image-model"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, tuple[ImageReference, ...]]] = []
        self.fail_next_edit = False

    async def generate(self, prompt: str) -> str:
        self.calls.append(("generate", prompt, ()))
        return f"fake://generate/{len(self.calls)}"

    async def generate_with_references(
        self,
        prompt: str,
        references: tuple[ImageReference, ...],
    ) -> str:
        self.calls.append(("reference", prompt, references))
        return f"fake://reference/{len(self.calls)}"

    async def edit(
        self,
        prompt: str,
        references: tuple[ImageReference, ...],
    ) -> str:
        self.calls.append(("edit", prompt, references))
        if self.fail_next_edit:
            self.fail_next_edit = False
            raise RuntimeError("temporary provider failure")
        return f"fake://edit/{len(self.calls)}"


async def fake_download(url: str, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(url.encode("utf-8"))
    return output_path


def _plan_payload() -> dict[str, Any]:
    beats = [
        _beat(
            "beat_1",
            "NEW_IMAGE",
            None,
            "WIDE",
            "Full vertical shaft cutaway",
            "STATIC",
            "Dry shaft, intact rod and open route",
            "Initial establishing state",
        ),
        _beat(
            "beat_2",
            "TRANSFORM",
            "beat_1",
            "MEDIUM",
            "Closer focus on miners in the same shaft",
            "ZOOM",
            "Dry shaft, intact rod and open route",
            "Only the focus moves toward the miners",
        ),
        _beat(
            "beat_3",
            "EDIT_EXISTING",
            "beat_2",
            "MEDIUM",
            "Same miners and rod framing",
            "STATIC",
            "Rod low position before lift; route open",
            "Rod moves upward",
            progressive=("rod", "low position", "raised position"),
        ),
        _beat(
            "beat_4",
            "EDIT_EXISTING",
            "beat_3",
            "CLOSE",
            "Connection and raised rod in the established shaft",
            "STATIC",
            "Raised rod; top connection intact before fracture",
            "Top connection breaks",
            progressive=("rod", "connection intact", "connection broken"),
        ),
        _beat(
            "beat_5",
            "OVERLAY",
            "beat_4",
            "CLOSE",
            "Same broken connection",
            "STATIC",
            "Raised rod with broken connection",
            "Add only direction and distance information",
            overlay="Draw a downward arrow and distance measurement beside the rod",
        ),
        _beat(
            "beat_6",
            "EDIT_EXISTING",
            "beat_4",
            "WIDE",
            "Return to the full established shaft",
            "STATIC",
            "Broken rod and debris now block the shaft route",
            "Debris blocks the shaft",
            progressive=("debris", "route open", "route blocked"),
        ),
        _beat(
            "beat_7",
            "REUSE",
            "beat_6",
            "WIDE",
            "Same full blocked shaft",
            "STATIC",
            "Broken rod and debris block the shaft route",
            "No visual state change",
        ),
        _beat(
            "beat_8",
            "REFERENCE_GENERATION",
            "beat_6",
            "MEDIUM",
            "Alternate side tunnel connected to the same mine",
            "STATIC",
            "Blocked main shaft and open alternate tunnel",
            "Reveal a different useful route while preserving mine identity",
        ),
    ]
    return {
        "story_summary": "A mine lifting rod breaks and blocks the main route.",
        "visual_strategy": "Preserve one shaft while its state progressively changes.",
        "characters": [
            {
                "id": "miners",
                "name": "Miners",
                "description": "Three workers with simple helmets",
            }
        ],
        "locations": [
            {
                "id": "shaft",
                "name": "Mine shaft",
                "description": "A vertical shaft with a side tunnel",
                "spatial_layout": "Surface above, miners below, side tunnel left",
            }
        ],
        "important_objects": [
            {
                "id": "rod",
                "name": "Lifting rod",
                "description": "Central wooden lifting rod",
                "story_function": "Moves material through the shaft",
            },
            {
                "id": "debris",
                "name": "Debris",
                "description": "Large simple broken timber pieces",
                "story_function": "Blocks the escape route",
            },
        ],
        "recurring_environments": [
            {
                "id": "shaft_environment",
                "location_id": "shaft",
                "continuity_requirements": "Keep shaft geometry and platforms stable",
            }
        ],
        "possible_master_scenes": [
            {
                "id": "shaft_master",
                "location_id": "shaft",
                "description": "Full mine shaft cutaway",
                "environment_geometry": "One vertical shaft with left side tunnel",
                "recurring_object_positions": "Rod center, platforms right",
                "color_palette": "Muted grey and ochre",
                "basic_composition": "Surface top, miners bottom",
                "characters_visible": ["miners"],
                "important_objects": ["rod", "debris"],
            }
        ],
        "visual_beats": beats,
    }


def _beat(
    beat_id: str,
    operation: str,
    source_id: str | None,
    framing: str,
    camera_view: str,
    movement: str,
    physical_state: str,
    change: str,
    *,
    progressive: tuple[str, str, str] | None = None,
    overlay: str | None = None,
) -> dict[str, Any]:
    return {
        "id": beat_id,
        "narration_segment": f"Narration for {beat_id}",
        "visual_purpose": f"Advance the physical story at {beat_id}",
        "what_viewer_should_understand": physical_state,
        "location_id": "shaft",
        "characters_visible": ["miners"],
        "important_objects": ["rod", "debris"],
        "camera_framing": framing,
        "camera_view": camera_view,
        "framing_reason": "Make the current state easy to understand",
        "camera_movement": movement,
        "geography_established_by": "beat_1" if beat_id != "beat_1" else None,
        "master_scene_id": "shaft_master",
        "physical_state": physical_state,
        "progressive_change": (
            {
                "subject_id": progressive[0],
                "previous_state": progressive[1],
                "current_state": progressive[2],
                "progression": f"{progressive[1]} to {progressive[2]}",
            }
            if progressive is not None
            else None
        ),
        "safety_geography": None,
        "anticipated_consequence": None,
        "visual_focus": camera_view,
        "must_not_show": ["unrelated machinery"],
        "change_from_previous_beat": change,
        "information_added_beyond_narration": "Shows spatial cause and consequence",
        "preferred_visual_operation": operation,
        "source_visual_id": source_id,
        "overlay_description": overlay,
        "estimated_duration_seconds": 4,
    }


def _setup_execution(
    session: Session,
) -> tuple[str, VisualPlan, Any]:
    project = create_project(
        session,
        name="Mine continuity",
        story_text="Complete mine rod failure narration",
        image_provider="seedream",
    )
    plan = VisualPlan.model_validate(_plan_payload())
    plan_record = save_project_visual_plan_record(
        session,
        project_id=project.id,
        schema_version="visual_plan_v1",
        visual_director_version="visual_director_v1",
        story_text_hash=hash_story_text(project.story_text),
        plan_json=plan.model_dump(mode="json"),
    )
    plan_revision = _hash(plan.model_dump(mode="json"))
    decisions = [
        {
            "position": position,
            "beat_id": beat.id,
            "preferred_operation": beat.preferred_visual_operation.value,
            "resolved_operation": beat.preferred_visual_operation.value,
            "fallback_used": False,
            "fallback_from": None,
            "reason": ["integration fixture resolved operation"],
            "source_visual_ids": (
                [beat.source_visual_id] if beat.source_visual_id is not None else []
            ),
            "source_image_paths": [],
        }
        for position, beat in enumerate(plan.visual_beats)
    ]
    execution = save_visual_execution_plan(
        session,
        project_id=project.id,
        visual_plan=plan_record,
        visual_plan_revision=plan_revision,
        provider="seedream",
        model="fake-image-model",
        capability_snapshot=FakeImageProvider.capabilities.snapshot(),
        decision_input_snapshot={"references": {}, "evidence": {}},
        resolution_revision=_hash({"fixture": project.id}),
        decisions=decisions,
    )
    return project.id, plan, execution


def _executor(
    session: Session,
    tmp_path: Path,
    provider: FakeImageProvider,
    *,
    qa_service: VisualQAService | None = None,
    max_visual_qa_attempts: int = 3,
) -> VisualBeatAssetExecutor:
    def resolver(
        name: str,
        config: Mapping[str, Any] | None,
    ) -> FakeImageProvider:
        del name, config
        return provider

    return VisualBeatAssetExecutor(
        session,
        provider_resolver=resolver,
        downloader=fake_download,
        projects_root=tmp_path / "projects",
        qa_service=qa_service,
        max_visual_qa_attempts=max_visual_qa_attempts,
    )


class FakeVisualQAClient:
    provider = "fake-vision"
    model = "fake-vision-model"

    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self.decisions = decisions
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def evaluate(self, prompt: str, image_paths: tuple[str, ...]) -> str:
        self.calls.append((prompt, image_paths))
        index = min(len(self.calls) - 1, len(self.decisions) - 1)
        return json.dumps(self.decisions[index])


def _qa_pass() -> dict[str, Any]:
    return {
        "result": "PASS",
        "scores": {
            "story_clarity": 0.95,
            "continuity": 0.95,
            "style": 0.9,
            "composition": 0.9,
            "operation_correctness": 0.95,
        },
        "problem_categories": [],
        "reasons": [],
        "correction_instruction": None,
        "severity": None,
    }


def _qa_regenerate(
    category: str,
    reason: str,
    correction: str,
    *,
    severity: str = "major",
) -> dict[str, Any]:
    return {
        "result": "REGENERATE",
        "scores": {
            "story_clarity": 0.7,
            "continuity": 0.5,
            "style": 0.8,
            "composition": 0.75,
            "operation_correctness": 0.5,
        },
        "problem_categories": [category],
        "reasons": [reason],
        "correction_instruction": correction,
        "severity": severity,
    }


def test_eight_beat_story_executes_asset_graph_without_unnecessary_calls(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id, plan, execution = _setup_execution(session)
    provider = FakeImageProvider()
    monkeypatch.setattr(
        execution_module,
        "resolve_project_visual_operations",
        lambda session, project_id, provider_resolver: execution,
    )
    executor = _executor(session, tmp_path, provider)

    results = asyncio.run(executor.execute_project(project_id, execution.id))

    assert len(results) == 8
    assert all(item.generation_status == "SUCCEEDED" for item in results)
    assert [call[0] for call in provider.calls] == [
        "generate",  # immutable master first
        "generate",  # beat 1 NEW_IMAGE
        "edit",      # beat 3
        "edit",      # beat 4
        "edit",      # beat 6
        "reference", # beat 8
    ]
    assert "Narration for beat_1" not in (results[0].prompt_used or "")
    assert "STYLE CONTRACT [rough_explainer_v1]" in (results[0].prompt_used or "")
    assert "KEEP UNCHANGED" in (results[2].prompt_used or "")
    assert "CHANGE ONLY" in (results[2].prompt_used or "")
    assert results[1].source_result_id == results[0].id
    assert results[1].transform_metadata["type"] == "ZOOM_IN"
    assert results[4].source_result_id == results[3].id
    assert results[4].overlay_metadata["type"] == "DISTANCE"
    assert results[5].source_result_id == results[3].id
    assert results[6].source_result_id == results[5].id
    assert results[6].output_path == results[5].output_path
    assert results[7].source_result_id == results[5].id
    assert [item["role"] for item in results[7].reference_snapshot] == [
        "MASTER_LOCATION",
        "PREVIOUS_STATE",
    ]
    assert len(provider.calls[-1][2]) <= provider.capabilities.max_reference_images
    master = get_master_scene_asset(session, project_id, "shaft_master")
    assert master is not None
    assert Path(master.file_path).is_file()
    assert plan.model_dump(mode="json") == _plan_payload()

    call_count = len(provider.calls)
    repeated = asyncio.run(executor.execute_project(project_id, execution.id))
    assert [item.id for item in repeated] == [item.id for item in results]
    assert len(provider.calls) == call_count

    summary = build_visual_asset_execution_summary(results)
    assert summary.beats == 8
    assert summary.api_image_calls == 5
    assert summary.unique_visual_assets == 5
    assert summary.operation_counts == {
        "EDIT_EXISTING": 3,
        "NEW_IMAGE": 1,
        "OVERLAY": 1,
        "REFERENCE_GENERATION": 1,
        "REUSE": 1,
        "TRANSFORM": 1,
    }
    assert delete_project(session, project_id) is True
    assert list_beat_visual_results(session, project_id) == []
    assert get_master_scene_asset(session, project_id, "shaft_master") is None


def test_executor_consumes_real_stage_two_resolution(
    session: Session,
    tmp_path: Path,
) -> None:
    project = create_project(
        session,
        name="Single beat",
        story_text="One complete establishing narration",
        image_provider="seedream",
    )
    payload = _plan_payload()
    payload["recurring_environments"] = []
    payload["possible_master_scenes"] = []
    payload["visual_beats"] = [payload["visual_beats"][0]]
    payload["visual_beats"][0]["master_scene_id"] = None
    plan = VisualPlan.model_validate(payload)
    save_project_visual_plan_record(
        session,
        project_id=project.id,
        schema_version="visual_plan_v1",
        visual_director_version="visual_director_v1",
        story_text_hash=hash_story_text(project.story_text),
        plan_json=plan.model_dump(mode="json"),
    )
    provider = FakeImageProvider()

    def resolver(
        name: str,
        config: Mapping[str, Any] | None,
    ) -> FakeImageProvider:
        del name, config
        return provider

    execution = resolve_project_visual_operations(
        session,
        project.id,
        provider_resolver=resolver,
    )
    executor = VisualBeatAssetExecutor(
        session,
        provider_resolver=resolver,
        downloader=fake_download,
        projects_root=tmp_path / "projects",
    )

    result = asyncio.run(
        executor.execute_beat(project.id, execution.id, "beat_1")
    )

    assert result.execution_plan_id == execution.id
    assert result.resolved_operation == "NEW_IMAGE"
    assert [call[0] for call in provider.calls] == ["generate"]


def test_failed_beat_is_persisted_and_retried_without_losing_successes(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id, _, execution = _setup_execution(session)
    provider = FakeImageProvider()
    monkeypatch.setattr(
        execution_module,
        "resolve_project_visual_operations",
        lambda session, project_id, provider_resolver: execution,
    )
    executor = _executor(session, tmp_path, provider)
    beat_1 = asyncio.run(executor.execute_beat(project_id, execution.id, "beat_1"))
    beat_2 = asyncio.run(executor.execute_beat(project_id, execution.id, "beat_2"))
    provider.fail_next_edit = True

    with pytest.raises(BeatVisualExecutionError, match="beat_3"):
        asyncio.run(executor.execute_beat(project_id, execution.id, "beat_3"))

    attempts = list_beat_visual_results(
        session,
        project_id,
        execution_plan_id=execution.id,
        beat_id="beat_3",
    )
    assert len(attempts) == 1
    assert attempts[0].generation_status == "FAILED"
    assert attempts[0].is_accepted is False
    assert "temporary provider failure" in (attempts[0].error or "")
    assert Path(beat_1.output_path or "").is_file()
    assert beat_2.output_path == beat_1.output_path

    retried = asyncio.run(executor.execute_beat(project_id, execution.id, "beat_3"))
    attempts = list_beat_visual_results(
        session,
        project_id,
        execution_plan_id=execution.id,
        beat_id="beat_3",
    )
    assert retried.generation_status == "SUCCEEDED"
    assert retried.attempt == 2
    assert [item.generation_status for item in attempts] == ["FAILED", "SUCCEEDED"]
    assert Path(beat_1.output_path or "").is_file()


def test_reference_selection_orders_roles_and_respects_provider_limit(
    tmp_path: Path,
) -> None:
    style_path = tmp_path / "style.png"
    master_path = tmp_path / "master.png"
    source_path = tmp_path / "source.png"
    style_path.write_bytes(b"\x89PNG\r\n\x1a\nstyle")
    master_path.write_bytes(b"master")
    source_path.write_bytes(b"source")
    style_hash = hashlib.sha256(style_path.read_bytes()).hexdigest()
    master_hash = hashlib.sha256(master_path.read_bytes()).hexdigest()
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    style = StyleReferenceAsset(
        project_id="project",
        style_id="rough_explainer_v1",
        file_path=str(style_path),
        file_sha256=style_hash,
    )
    master = MasterSceneAsset(
        id="master-asset",
        project_id="project",
        master_scene_id="shaft_master",
        file_path=str(master_path),
        file_sha256=master_hash,
        style_version="rough_explainer_v1",
        generation_prompt="master prompt",
        provider="fake",
        model="fake",
        reference_hashes=[],
    )
    source = VisualAssetSource(
        asset_id="beat-result",
        result_id="beat-result",
        path=str(source_path),
        sha256=source_hash,
        visual_id="beat_3",
        reason="closest state",
    )
    beat = VisualPlan.model_validate(_plan_payload()).visual_beats[3]

    selected = select_visual_references(
        beat,
        VisualOperation.EDIT_EXISTING,
        ImageProviderCapabilities(
            reference_generation=True,
            multiple_references=True,
            max_reference_images=3,
            image_editing=True,
        ),
        style_reference=style,
        master_asset=master,
        source=source,
    )
    assert [item.semantic_role for item in selected] == [
        "STYLE",
        "MASTER_LOCATION",
        "PREVIOUS_STATE",
    ]

    limited = select_visual_references(
        beat,
        VisualOperation.EDIT_EXISTING,
        ImageProviderCapabilities(
            reference_generation=True,
            multiple_references=True,
            max_reference_images=2,
            image_editing=True,
        ),
        style_reference=style,
        master_asset=master,
        source=source,
    )
    assert [item.semantic_role for item in limited] == [
        "STYLE",
        "PREVIOUS_STATE",
    ]


def test_pending_attempt_is_marked_failed_after_restart(tmp_path: Path) -> None:
    database_path = tmp_path / "restart.db"
    engine = create_sqlite_engine(database_path)
    init_database(engine)
    factory = create_session_factory(engine)
    with factory() as first_session:
        project_id, _, execution = _setup_execution(first_session)
        pending = create_beat_visual_result(
            first_session,
            project_id=project_id,
            visual_plan_id=execution.visual_plan_id,
            visual_plan_revision=execution.visual_plan_revision,
            execution_plan_id=execution.id,
            beat_id="beat_1",
            resolved_operation="NEW_IMAGE",
            source_result_id=None,
            source_master_asset_id=None,
            source_path=None,
            output_path=None,
            file_sha256=None,
            master_scene_id="shaft_master",
            prompt_used="prompt",
            provider="seedream",
            model="fake-image-model",
            style_version="rough_explainer_v1",
            reference_snapshot=[],
            generation_status="PENDING",
            qa_status="NOT_RUN",
            is_accepted=False,
            error=None,
            transform_metadata=None,
            overlay_metadata=None,
            semantic_state_snapshot={"beat_position": 0},
            generation_revision="d" * 64,
            attempt=1,
        )
        pending_id = pending.id

    init_database(engine)
    with factory() as second_session:
        recovered = second_session.get(type(pending), pending_id)
        assert recovered is not None
        assert recovered.generation_status == "FAILED"
        assert "interrupted" in (recovered.error or "").lower()
        assert recovered.is_accepted is False
    engine.dispose()


def test_generated_candidate_passes_qa_and_becomes_accepted(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id, _, execution = _setup_execution(session)
    provider = FakeImageProvider()
    client = FakeVisualQAClient([_qa_pass()])
    monkeypatch.setattr(
        execution_module,
        "resolve_project_visual_operations",
        lambda session, project_id, provider_resolver: execution,
    )
    executor = _executor(
        session,
        tmp_path,
        provider,
        qa_service=VisualQAService(client),
    )

    result = asyncio.run(executor.execute_beat(project_id, execution.id, "beat_1"))

    assert result.is_accepted is True
    assert result.qa_status == "PASS"
    assert result.qa_provider == "fake-vision"
    assert result.qa_model == "fake-vision-model"
    assert result.qa_prompt_version == "visual_qa_v1"
    assert result.accepted_at is not None
    assert len(result.qa_evaluations) == 1
    assert client.calls[0][1][0] == result.output_path
    assert "five dimension scores" in client.calls[0][0]
    assert "intentional change from previous beat" in client.calls[0][0]

    repeated = asyncio.run(executor.execute_beat(project_id, execution.id, "beat_1"))
    assert repeated.id == result.id
    assert len(client.calls) == 1


def test_complete_asset_graph_produces_visual_qa_debug_summary(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id, _, execution = _setup_execution(session)
    provider = FakeImageProvider()
    client = FakeVisualQAClient([_qa_pass()])
    monkeypatch.setattr(
        execution_module,
        "resolve_project_visual_operations",
        lambda session, project_id, provider_resolver: execution,
    )
    executor = _executor(
        session,
        tmp_path,
        provider,
        qa_service=VisualQAService(client),
    )

    results = asyncio.run(executor.execute_project(project_id, execution.id))
    all_candidates = list_beat_visual_results(
        session,
        project_id,
        execution_plan_id=execution.id,
    )
    summary = build_visual_qa_execution_summary(all_candidates)

    assert len(results) == 8
    assert len(client.calls) == 5
    assert summary.beats == 5
    assert summary.generated_candidates == 5
    assert summary.pass_first_attempt == 5
    assert summary.pass_with_warning == 0
    assert summary.regenerated == 0
    assert summary.continuity_failures == 0
    assert summary.style_drift_failures == 0
    assert summary.composition_failures == 0
    assert summary.average_attempts_per_accepted_generated_beat == 1.0


def test_edit_qa_retry_preserves_source_references_and_original_constraints(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id, _, execution = _setup_execution(session)
    provider = FakeImageProvider()
    monkeypatch.setattr(
        execution_module,
        "resolve_project_visual_operations",
        lambda session, project_id, provider_resolver: execution,
    )
    plain_executor = _executor(session, tmp_path, provider)
    asyncio.run(plain_executor.execute_beat(project_id, execution.id, "beat_1"))
    source = asyncio.run(
        plain_executor.execute_beat(project_id, execution.id, "beat_2")
    )
    client = FakeVisualQAClient(
        [
            _qa_regenerate(
                "EDIT_CHANGED_TOO_MUCH",
                "The shaft camera and fixed platforms changed",
                "Restore the original camera and both fixed platforms",
            ),
            _qa_pass(),
        ]
    )
    executor = _executor(
        session,
        tmp_path,
        provider,
        qa_service=VisualQAService(client),
    )

    accepted = asyncio.run(
        executor.execute_beat(project_id, execution.id, "beat_3")
    )
    attempts = list_beat_visual_results(
        session,
        project_id,
        execution_plan_id=execution.id,
        beat_id="beat_3",
    )

    assert len(attempts) == 2
    assert attempts[0].qa_status == "REGENERATE"
    assert attempts[0].is_accepted is False
    assert attempts[1].qa_status == "PASS"
    assert accepted.id == attempts[1].id
    assert {item.source_result_id for item in attempts} == {source.id}
    edit_calls = [call for call in provider.calls if call[0] == "edit"]
    assert len(edit_calls) == 2
    assert [reference.reference_id for reference in edit_calls[0][2]] == [
        reference.reference_id for reference in edit_calls[1][2]
    ]
    assert "KEEP UNCHANGED" in edit_calls[1][1]
    assert "CHANGE ONLY" in edit_calls[1][1]
    assert "Restore the original camera" in edit_calls[1][1]
    assert "STYLE CONTRACT [rough_explainer_v1]" in edit_calls[1][1]


def test_hard_qa_failures_stop_at_limit_and_rejected_never_becomes_source(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id, _, execution = _setup_execution(session)
    provider = FakeImageProvider()
    client = FakeVisualQAClient(
        [
            _qa_regenerate(
                "LOCATION_DRIFT",
                "The rectangular shaft became a circular cavern",
                "Restore the master shaft geometry",
                severity="critical",
            )
        ]
    )
    monkeypatch.setattr(
        execution_module,
        "resolve_project_visual_operations",
        lambda session, project_id, provider_resolver: execution,
    )
    executor = _executor(
        session,
        tmp_path,
        provider,
        qa_service=VisualQAService(client),
        max_visual_qa_attempts=2,
    )

    with pytest.raises(BeatVisualExecutionError, match="rejected all 2"):
        asyncio.run(executor.execute_beat(project_id, execution.id, "beat_1"))

    candidates = list_beat_visual_results(
        session,
        project_id,
        execution_plan_id=execution.id,
        beat_id="beat_1",
    )
    assert len(candidates) == 2
    assert all(item.generation_status == "SUCCEEDED" for item in candidates)
    assert all(item.qa_status == "REGENERATE" for item in candidates)
    assert not any(item.is_accepted for item in candidates)
    assert list_beat_visual_results(
        session,
        project_id,
        execution_plan_id=execution.id,
        accepted_only=True,
    ) == []


def test_retry_limit_can_accept_only_a_minor_best_available_candidate(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id, _, execution = _setup_execution(session)
    provider = FakeImageProvider()
    client = FakeVisualQAClient(
        [
            _qa_regenerate(
                "EXCESSIVE_CLUTTER",
                "A few unnecessary background props remain",
                "Remove the unnecessary background props",
                severity="minor",
            )
        ]
    )
    monkeypatch.setattr(
        execution_module,
        "resolve_project_visual_operations",
        lambda session, project_id, provider_resolver: execution,
    )
    executor = _executor(
        session,
        tmp_path,
        provider,
        qa_service=VisualQAService(client),
        max_visual_qa_attempts=2,
    )

    result = asyncio.run(executor.execute_beat(project_id, execution.id, "beat_1"))
    candidates = list_beat_visual_results(
        session,
        project_id,
        execution_plan_id=execution.id,
        beat_id="beat_1",
    )

    assert len(candidates) == 2
    assert result.qa_status == "PASS_WITH_WARNING"
    assert result.is_accepted is True
    assert result.qa_warning is not None
    assert len(result.qa_evaluations) == 1
    assert result.qa_evaluations[0].result == "REGENERATE"


def test_non_generated_operations_skip_vision_qa(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id, _, execution = _setup_execution(session)
    provider = FakeImageProvider()
    client = FakeVisualQAClient([_qa_pass()])
    monkeypatch.setattr(
        execution_module,
        "resolve_project_visual_operations",
        lambda session, project_id, provider_resolver: execution,
    )
    executor = _executor(
        session,
        tmp_path,
        provider,
        qa_service=VisualQAService(client),
    )

    first = asyncio.run(executor.execute_beat(project_id, execution.id, "beat_1"))
    transformed = asyncio.run(
        executor.execute_beat(project_id, execution.id, "beat_2")
    )

    assert first.qa_status == "PASS"
    assert transformed.qa_status == "NOT_RUN"
    assert transformed.is_accepted is True
    assert len(client.calls) == 1


def test_manual_qa_overrides_are_separate_from_automated_decision(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id, _, execution = _setup_execution(session)
    provider = FakeImageProvider()
    client = FakeVisualQAClient([_qa_pass()])
    monkeypatch.setattr(
        execution_module,
        "resolve_project_visual_operations",
        lambda session, project_id, provider_resolver: execution,
    )
    result = asyncio.run(
        _executor(
            session,
            tmp_path,
            provider,
            qa_service=VisualQAService(client),
        ).execute_beat(project_id, execution.id, "beat_1")
    )
    service = VisualQAManualOverrideService(session)

    rejected = service.reject(result.id, "Visible object is still ambiguous")
    assert rejected.is_accepted is False
    assert rejected.manual_qa_override == "REJECTED"
    assert rejected.qa_result == "PASS"

    accepted = service.accept(result.id, "Reviewed against the master scene")
    assert accepted.is_accepted is True
    assert accepted.manual_qa_override == "ACCEPTED"
    requested = service.request_regeneration(result.id, "Try a clearer crop")
    assert requested.is_accepted is False
    assert requested.manual_qa_override == "REGENERATE_REQUESTED"


def test_visual_qa_summary_reports_regeneration_and_failure_categories() -> None:
    first = type("Candidate", (), {})()
    first.resolved_operation = "NEW_IMAGE"
    first.generation_status = "SUCCEEDED"
    first.is_accepted = False
    first.beat_id = "beat_1"
    first.qa_problem_categories = ["STYLE_DRIFT_REALISM"]
    first.qa_result = "REGENERATE"
    first.qa_attempt = 1
    first.attempt = 1
    second = type("Candidate", (), {})()
    second.resolved_operation = "NEW_IMAGE"
    second.generation_status = "SUCCEEDED"
    second.is_accepted = True
    second.beat_id = "beat_1"
    second.qa_problem_categories = []
    second.qa_result = "PASS"
    second.qa_attempt = 2
    second.attempt = 2

    summary = build_visual_qa_execution_summary([first, second])

    assert summary.beats == 1
    assert summary.generated_candidates == 2
    assert summary.regenerated == 1
    assert summary.style_drift_failures == 1
    assert summary.average_attempts_per_accepted_generated_beat == 2.0


def test_visual_qa_result_and_immutable_evaluation_survive_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_sqlite_engine(tmp_path / "qa-restart.db")
    init_database(engine)
    factory = create_session_factory(engine)
    provider = FakeImageProvider()
    client = FakeVisualQAClient([_qa_pass()])
    with factory() as first_session:
        project_id, _, execution = _setup_execution(first_session)
        monkeypatch.setattr(
            execution_module,
            "resolve_project_visual_operations",
            lambda session, project_id, provider_resolver: execution,
        )
        result = asyncio.run(
            _executor(
                first_session,
                tmp_path,
                provider,
                qa_service=VisualQAService(client),
            ).execute_beat(project_id, execution.id, "beat_1")
        )
        result_id = result.id

    init_database(engine)
    with factory() as restarted_session:
        persisted = restarted_session.get(type(result), result_id)
        assert persisted is not None
        assert persisted.qa_status == "PASS"
        assert persisted.qa_scores["story_clarity"] == 0.95
        assert persisted.qa_revision is not None
        assert persisted.is_accepted is True
        assert len(persisted.qa_evaluations) == 1
        assert persisted.qa_evaluations[0].prompt_version == "visual_qa_v1"
    engine.dispose()


def _hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
