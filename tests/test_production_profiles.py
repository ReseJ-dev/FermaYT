"""Draft execution policy tests."""

from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest
from sqlalchemy.orm import Session
from test_visual_asset_execution import FakeImageProvider, _plan_payload

from app.database import create_session_factory, create_sqlite_engine, init_database
from app.models.visual_plan import VisualPlan
from app.pipeline.visual_operation_engine import VisualDecisionEvidence
from app.production_profiles import select_draft_key_beats
from app.repositories import create_project, save_project_visual_plan_record
from app.services.visual_operations import resolve_project_visual_operations
from app.services.visual_planning import (
    VISUAL_DIRECTOR_VERSION,
    VISUAL_PLAN_SCHEMA_VERSION,
    hash_story_text,
)


@pytest.fixture
def session(tmp_path: Path) -> Iterator[Session]:
    engine = create_sqlite_engine(tmp_path / "app.db")
    init_database(engine)
    factory = create_session_factory(engine)
    with factory() as value:
        yield value
    engine.dispose()


def test_draft_selection_keeps_semantic_plan_and_reduces_paid_beats() -> None:
    plan = VisualPlan.model_validate(_plan_payload())
    original = plan.model_dump(mode="json")

    selection = select_draft_key_beats(plan, paid_visual_ratio=0.3)

    assert plan.model_dump(mode="json") == original
    assert selection.actual_paid_beats < len(plan.visual_beats)
    assert selection.target_paid_beats == 3


def test_draft_selection_keeps_critical_progressive_changes() -> None:
    plan = VisualPlan.model_validate(_plan_payload())

    selection = select_draft_key_beats(plan, paid_visual_ratio=0.3)

    progressive = {
        beat.id for beat in plan.visual_beats if beat.progressive_change is not None
    }
    assert progressive <= selection.key_beat_ids
    assert plan.visual_beats[0].id in selection.key_beat_ids
    assert plan.visual_beats[-1].id in selection.key_beat_ids


def test_draft_execution_has_fewer_paid_operations_than_final(
    session: Session,
) -> None:
    project = create_project(
        session,
        name="Draft policy",
        story_text="A complete mine story",
        image_provider="seedream",
    )
    payload = _plan_payload()
    payload["visual_beats"][1]["preferred_visual_operation"] = (
        "REFERENCE_GENERATION"
    )
    plan = VisualPlan.model_validate(payload)
    save_project_visual_plan_record(
        session,
        project_id=project.id,
        schema_version=VISUAL_PLAN_SCHEMA_VERSION,
        visual_director_version=VISUAL_DIRECTOR_VERSION,
        story_text_hash=hash_story_text(project.story_text),
        plan_json=plan.model_dump(mode="json"),
    )
    provider = FakeImageProvider()

    def resolver(
        name: str,
        config: Mapping[str, object] | None,
    ) -> FakeImageProvider:
        del name, config
        return provider

    final = resolve_project_visual_operations(
        session,
        project.id,
        provider_resolver=resolver,
        production_profile="FINAL",
        evidence_by_beat={
            "beat_2": VisualDecisionEvidence(
                substantially_different_composition=True,
                new_image_improves_understanding=True,
            )
        },
    )
    draft = resolve_project_visual_operations(
        session,
        project.id,
        provider_resolver=resolver,
        production_profile="DRAFT",
        draft_paid_visual_ratio=0.3,
        evidence_by_beat={
            "beat_2": VisualDecisionEvidence(
                substantially_different_composition=True,
                new_image_improves_understanding=True,
            )
        },
    )
    paid = {"NEW_IMAGE", "REFERENCE_GENERATION", "EDIT_EXISTING"}
    final_paid = sum(item.resolved_operation in paid for item in final.decisions)
    draft_paid = sum(item.resolved_operation in paid for item in draft.decisions)
    critical_ids = {
        beat.id for beat in plan.visual_beats if beat.progressive_change is not None
    }

    assert draft_paid < final_paid
    assert critical_ids <= {
        item.beat_id for item in draft.decisions if item.resolved_operation in paid
    }
    assert draft.visual_plan.plan_json == final.visual_plan.plan_json
