import asyncio
from datetime import UTC, datetime

import pytest

from app.budgets import (
    BUDGET_ESTIMATE_EXCEEDS_LIMIT,
    BUDGET_EXHAUSTED,
    BUDGET_EXHAUSTED_DURING_QA,
    BUDGET_COST_UNKNOWN,
    GenerationBudgetError,
    ProjectBudgetGuard,
)
from app.costs import (
    CostEstimate,
    PricingUnit,
    UsageStatus,
    configure_provider_pricing,
    record_provider_usage,
)
from app.database import create_session_factory, create_sqlite_engine, init_database
from app.jobs import GenerationJobManager, GenerationJobStatus, GenerationJobType
from app.persistence import Project, Scene
from app.repositories import update_project


@pytest.fixture
def session(tmp_path):
    engine = create_sqlite_engine(tmp_path / "budget.db")
    init_database(engine)
    factory = create_session_factory(engine)
    with factory() as value:
        value.add(Project(
            id="project-1", name="Budget", story_text="Story",
            image_provider="qwen", image_model="qwen-image-3.0",
            tts_provider="qwen", tts_model="qwen3-tts-flash",
            tts_voice="Cherry", tts_language="Russian",
            width=1080, height=1920, fps=30, image_fit="cover",
            generation_budget_enabled=True,
            generation_budget_amount=0.10,
            generation_budget_currency="USD",
            generation_budget_warning_threshold=0.8,
        ))
        value.commit()
        configure_provider_pricing(
            value, provider="qwen", model="qwen-image-3.0",
            operation="NEW_IMAGE", pricing_unit=PricingUnit.PER_IMAGE,
            price=0.06, currency="USD", version="v1",
            effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        )
        configure_provider_pricing(
            value, provider="dashscope", model="qwen-vl-max",
            operation="VISUAL_QA", pricing_unit=PricingUnit.PER_REQUEST,
            price=0.03, currency="USD", version="v1",
            effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        )
        yield value


def _check_image(guard: ProjectBudgetGuard, *, retry: bool = False) -> None:
    guard.check_paid_call(
        pipeline_stage="VISUAL_GENERATION", provider="qwen",
        model="qwen-image-3.0", operation="NEW_IMAGE",
        unit_type=PricingUnit.PER_IMAGE, input_units=1,
        is_qa_retry=retry, beat_id="beat-1",
    )


def _record_image(session, revision: str, cost_status=UsageStatus.SUCCEEDED):
    return record_provider_usage(
        session, project_id="project-1", job_id="job-1",
        pipeline_stage="VISUAL_GENERATION", provider="qwen",
        model="qwen-image-3.0", operation="NEW_IMAGE",
        request_revision=revision, unit_type=PricingUnit.PER_IMAGE,
        input_units=1, status=cost_status,
    )


def test_project_below_budget_allows_paid_call(session):
    _check_image(ProjectBudgetGuard(session, "project-1"))


def test_preflight_estimate_above_budget_blocks(session):
    estimate = CostEstimate(
        minimum=0.08, maximum=0.14, currency="USD", paid_image_calls=2,
        free_visual_operations=0, expected_qa_calls=2,
        maximum_qa_retry_calls=2, unpriced_operations=(),
    )
    with pytest.raises(GenerationBudgetError) as error:
        ProjectBudgetGuard(session, "project-1").check_preflight(estimate)
    assert error.value.code == BUDGET_ESTIMATE_EXCEEDS_LIMIT


def test_next_call_exceeding_limit_is_stopped_before_execution(session):
    _record_image(session, "first")
    provider_calls = 0
    with pytest.raises(GenerationBudgetError) as error:
        _check_image(ProjectBudgetGuard(session, "project-1"))
        provider_calls += 1
    assert error.value.code == BUDGET_EXHAUSTED
    assert provider_calls == 0


def test_qa_retry_has_specific_budget_pause_reason(session):
    _record_image(session, "first")
    with pytest.raises(GenerationBudgetError) as error:
        ProjectBudgetGuard(session, "project-1").check_paid_call(
            pipeline_stage="VISUAL_QA", provider="dashscope",
            model="qwen-vl-max", operation="VISUAL_QA",
            unit_type=PricingUnit.PER_REQUEST, input_units=2,
            is_qa_retry=True, beat_id="beat-1",
        )
    assert error.value.code == BUDGET_EXHAUSTED_DURING_QA


def test_cached_work_does_not_consume_budget_again(session):
    _record_image(session, "historical")
    _record_image(session, "cache", UsageStatus.CACHED)
    snapshot = ProjectBudgetGuard(session, "project-1").snapshot()
    assert snapshot.spent == pytest.approx(0.06)
    assert snapshot.available == pytest.approx(0.04)


def test_budget_increase_resumes_without_invalidating_scene_asset(session):
    scene = Scene(
        project_id="project-1", position=1, text="Narration",
        image_prompt="Prompt", image_path="existing.png",
    )
    session.add(scene)
    session.commit()
    _record_image(session, "first")
    with pytest.raises(GenerationBudgetError):
        _check_image(ProjectBudgetGuard(session, "project-1"))
    update_project(session, "project-1", generation_budget_amount=0.20)
    _check_image(ProjectBudgetGuard(session, "project-1"))
    session.refresh(scene)
    assert scene.image_path == "existing.png"


def test_soft_threshold_is_visible_without_stopping(session):
    _record_image(session, "first")
    project = session.get(Project, "project-1")
    project.generation_budget_warning_threshold = 0.5
    session.commit()
    snapshot = ProjectBudgetGuard(session, "project-1").snapshot()
    assert snapshot.warning == "Project has used 60% of generation budget."


def test_explicit_override_allows_unknown_or_over_budget_call(session):
    _record_image(session, "first")
    _check_image(ProjectBudgetGuard(session, "project-1", override=True))


def test_unknown_provider_price_pauses_conservatively(session):
    with pytest.raises(GenerationBudgetError) as error:
        ProjectBudgetGuard(session, "project-1").check_paid_call(
            pipeline_stage="TTS", provider="unknown", model="unknown",
            operation="TTS", unit_type=PricingUnit.PER_CHARACTER,
            input_units=100,
        )
    assert error.value.code == BUDGET_COST_UNKNOWN


def test_job_manager_persists_paused_budget_status(session, tmp_path):
    error = None
    try:
        _record_image(session, "first")
        _check_image(ProjectBudgetGuard(session, "project-1"))
    except GenerationBudgetError as exc:
        error = exc
    assert error is not None

    async def scenario() -> None:
        manager = GenerationJobManager(tmp_path / "jobs.db")

        async def operation(job_id: str) -> None:
            del job_id
            raise error

        job = await manager.enqueue(
            "project-1", GenerationJobType.GENERATE_VIDEO, operation
        )
        stored = await manager.wait(job.id)
        assert stored is not None
        assert stored.status is GenerationJobStatus.PAUSED_BUDGET
        assert stored.error == BUDGET_EXHAUSTED
        assert stored.report["budget_pause"]["code"] == BUDGET_EXHAUSTED

    asyncio.run(scenario())
