from datetime import UTC, datetime

import pytest

from app.costs import (
    CostCertainty,
    PricingUnit,
    UsageStatus,
    configure_provider_pricing,
    load_pricing_config,
    record_provider_usage,
    summarize_project_cost,
)
from app.database import create_session_factory, create_sqlite_engine, init_database
from app.persistence import Project


@pytest.fixture
def session(tmp_path):
    engine = create_sqlite_engine(tmp_path / "costs.db")
    init_database(engine)
    factory = create_session_factory(engine)
    with factory() as value:
        value.add(Project(
            id="project-1", name="Cost test", story_text="Story",
            image_provider="qwen", image_model="qwen-image-3.0",
            tts_provider="qwen", tts_model="qwen3-tts-flash",
            tts_voice="Cherry", tts_language="Russian",
            width=1080, height=1920, fps=30, image_fit="cover",
        ))
        value.commit()
        yield value


def add_price(session, operation, unit=PricingUnit.PER_IMAGE, price=0.04):
    return configure_provider_pricing(
        session, provider="qwen", model="qwen-image-3.0",
        operation=operation, pricing_unit=unit, price=price,
        currency="USD", version="2026-09-v1",
        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
    )


@pytest.mark.parametrize("operation", ["NEW_IMAGE", "EDIT", "REFERENCE_GENERATION"])
def test_paid_image_operations_add_versioned_usage(session, operation):
    pricing = add_price(session, operation)
    usage = record_provider_usage(
        session, project_id="project-1", job_id="job-1",
        pipeline_stage="VISUAL_GENERATION", beat_id="beat-1",
        provider="qwen", model="qwen-image-3.0", operation=operation,
        request_revision=f"{operation}-1", unit_type=PricingUnit.PER_IMAGE,
        input_units=1,
    )
    assert float(usage.estimated_cost) == pytest.approx(0.04)
    assert usage.pricing_version == pricing.version
    assert usage.cost_certainty == CostCertainty.ESTIMATED.value


@pytest.mark.parametrize("operation", ["REUSE", "TRANSFORM", "OVERLAY"])
def test_free_visual_operations_cost_zero(session, operation):
    usage = record_provider_usage(
        session, project_id="project-1", job_id="job-1",
        pipeline_stage="VISUAL_GENERATION", provider="qwen", model="qwen-image-3.0",
        operation=operation, request_revision=operation,
        unit_type=PricingUnit.PER_IMAGE, input_units=1,
        status=UsageStatus.SKIPPED,
    )
    assert float(usage.actual_cost) == 0
    assert usage.cost_certainty == CostCertainty.ACTUAL.value


def test_cached_asset_does_not_add_historical_cost_again(session):
    add_price(session, "NEW_IMAGE")
    record_provider_usage(
        session, project_id="project-1", job_id="old-job",
        pipeline_stage="VISUAL_GENERATION", provider="qwen", model="qwen-image-3.0",
        operation="NEW_IMAGE", request_revision="paid", unit_type=PricingUnit.PER_IMAGE,
        input_units=1,
    )
    record_provider_usage(
        session, project_id="project-1", job_id="new-job",
        pipeline_stage="VISUAL_GENERATION", provider="qwen", model="qwen-image-3.0",
        operation="NEW_IMAGE", request_revision="cached", unit_type=PricingUnit.PER_IMAGE,
        input_units=1, status=UsageStatus.CACHED,
    )
    current = summarize_project_cost(session, "project-1", job_id="new-job")
    assert current.run_cost == 0
    assert current.historical_project_cost == pytest.approx(0.04)


def test_qa_retry_planning_and_tts_are_separately_counted(session):
    add_price(session, "VISUAL_QA", PricingUnit.PER_REQUEST, 0.01)
    configure_provider_pricing(
        session, provider="qwen", model="qwen-image-3.0", operation="PLANNING",
        pricing_unit=PricingUnit.PER_REQUEST, price=0.02, currency="USD",
        version="v1", effective_from=datetime(2026, 1, 1, tzinfo=UTC),
    )
    configure_provider_pricing(
        session, provider="qwen", model="qwen-image-3.0", operation="TTS",
        pricing_unit=PricingUnit.PER_CHARACTER, price=0.001, currency="USD",
        version="v1", effective_from=datetime(2026, 1, 1, tzinfo=UTC),
    )
    for operation, stage, units, retry in [
        ("PLANNING", "PLANNING", 1, False),
        ("VISUAL_QA", "VISUAL_QA", 1, True),
        ("TTS", "TTS", 5, False),
    ]:
        record_provider_usage(
            session, project_id="project-1", job_id="job-1",
            pipeline_stage=stage, provider="qwen", model="qwen-image-3.0",
            operation=operation, request_revision=operation,
            unit_type=(PricingUnit.PER_CHARACTER if operation == "TTS" else PricingUnit.PER_REQUEST),
            input_units=units, is_qa_retry=retry,
        )
    summary = summarize_project_cost(session, "project-1", job_id="job-1")
    assert summary.run_cost == pytest.approx(0.035)
    assert summary.qa_retry_cost == pytest.approx(0.01)


def test_failed_unpriced_call_is_unknown_and_safe(session):
    usage = record_provider_usage(
        session, project_id="project-1", job_id="job-1",
        pipeline_stage="VISUAL_GENERATION", provider="qwen", model="qwen-image-3.0",
        operation="NEW_IMAGE", request_revision="failed",
        unit_type=PricingUnit.PER_IMAGE, input_units=1, status=UsageStatus.FAILED,
    )
    assert usage.estimated_cost is None
    assert usage.actual_cost is None
    assert usage.cost_certainty == CostCertainty.UNKNOWN.value
    assert summarize_project_cost(session, "project-1", job_id="job-1").unpriced_records == 1


def test_same_request_revision_is_idempotent(session):
    add_price(session, "NEW_IMAGE")
    kwargs = dict(
        project_id="project-1", job_id="job-1", pipeline_stage="VISUAL_GENERATION",
        provider="qwen", model="qwen-image-3.0", operation="NEW_IMAGE",
        request_revision="same", unit_type=PricingUnit.PER_IMAGE, input_units=1,
    )
    first = record_provider_usage(session, **kwargs)
    second = record_provider_usage(session, **kwargs)
    assert second.id == first.id
    assert summarize_project_cost(session, "project-1", job_id="job-1").run_cost == pytest.approx(0.04)


def test_pricing_json_is_idempotent_and_version_is_immutable(session, tmp_path):
    path = tmp_path / "pricing.json"
    path.write_text(
        '{"prices":[{"provider":"qwen","model":"qwen-image-3.0",'
        '"operation":"NEW_IMAGE","pricing_unit":"PER_IMAGE","price":0.04,'
        '"currency":"USD","version":"v1",'
        '"effective_from":"2026-01-01T00:00:00+00:00"}]}',
        encoding="utf-8",
    )
    first = load_pricing_config(session, path)
    second = load_pricing_config(session, path)
    assert second[0].id == first[0].id
    first[0].price = 0.05
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()
