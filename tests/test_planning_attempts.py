"""Cost-control regression tests for paid Visual Planning attempts."""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from app.budgets import PlanningBudgetError
from app.clients.structured_completion import StructuredCompletionMetadata
from app.costs import PricingUnit, configure_provider_pricing
from app.database import create_session_factory, create_sqlite_engine, init_database
from app.errors import (
    PlanningAttemptLimitError,
    PlanningBillingUncertainError,
    PlanningTooLargeError,
    StructuredAIProviderError,
)
from app.persistence import PlanningProviderAttempt
from app.provider_diagnostics import StructuredAIProviderDiagnostic
from app.repositories import create_project
from app.services.planning_attempts import (
    PlanningAttemptController,
    planning_budget_snapshot,
)


def _session(tmp_path: Path):
    engine = create_sqlite_engine(tmp_path / "planning.db")
    init_database(engine)
    return engine, create_session_factory(engine)()


def _project(session, **changes):
    return create_project(
        session,
        name="Planning cost test",
        story_text="A mining story",
        planning_provider="kimi",
        planning_model="kimi-k2.6",
        **changes,
    )


def _price(session, amount: float, unit: PricingUnit = PricingUnit.PER_REQUEST):
    return configure_provider_pricing(
        session,
        provider="kimi",
        model="kimi-k2.6",
        operation="PLANNING",
        pricing_unit=unit,
        price=amount,
        currency="EUR",
        version=f"test-{unit.value}-{amount}",
        effective_from=datetime(2024, 1, 1, tzinfo=UTC),
    )


class _Client:
    provider = "kimi"
    model = "kimi-k2.6"
    max_output_tokens = 1000

    def __init__(self, outcomes: list[object], session=None) -> None:
        self.outcomes = outcomes
        self.calls = 0
        self.session = session
        self.last_completion_metadata = None

    async def generate(self, prompt: str) -> str:
        del prompt
        self.calls += 1
        if self.session is not None:
            pending = list(self.session.scalars(select(PlanningProviderAttempt)))
            assert len(pending) == self.calls
            assert pending[-1].status == "PENDING"
        outcome = self.outcomes[self.calls - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return str(outcome)


def _timeout() -> StructuredAIProviderError:
    diagnostic = StructuredAIProviderDiagnostic(
        provider="kimi",
        model="kimi-k2.6",
        operation="visual_planning",
        category="PLANNING_TIMEOUT",
        attempt=1,
        max_attempts=1,
        timeout_seconds=600,
    )
    return StructuredAIProviderError(
        "timed out",
        diagnostic=diagnostic,
        user_summary="Visual planning provider failed",
    )


def test_one_timeout_is_one_persisted_attempt_and_no_resend(tmp_path: Path) -> None:
    engine, session = _session(tmp_path)
    project = _project(session)
    client = _Client([_timeout()], session)
    controller = PlanningAttemptController(
        session,
        project_id=project.id,
        job_id="job-1",
        planning_run_id="run-1",
        client=client,
    )

    with pytest.raises(PlanningBillingUncertainError):
        asyncio.run(controller.execute("large prompt", "INITIAL"))

    attempts = list(session.scalars(select(PlanningProviderAttempt)))
    assert client.calls == 1
    assert len(attempts) == 1
    assert attempts[0].status == "TIMED_OUT_BILLING_UNKNOWN"
    assert attempts[0].billing_status == "UNKNOWN"
    assert attempts[0].progress_state == "PAUSED_AFTER_TIMEOUT"
    engine.dispose()


def test_repair_uses_same_global_limit_without_nested_multiplication(
    tmp_path: Path,
) -> None:
    engine, session = _session(tmp_path)
    project = _project(session)
    client = _Client(["{}", "{}"])
    controller = PlanningAttemptController(
        session,
        project_id=project.id,
        job_id="job-2",
        planning_run_id="run-2",
        client=client,
        max_paid_requests=2,
    )

    asyncio.run(controller.execute("initial", "INITIAL"))
    controller.mark_validation("REPAIR_REQUIRED")
    asyncio.run(controller.execute("repair", "REPAIR"))
    controller.mark_validation("SCHEMA_INVALID")
    with pytest.raises(PlanningAttemptLimitError):
        asyncio.run(controller.execute("hidden third call", "REPAIR"))

    attempts = list(
        session.scalars(
            select(PlanningProviderAttempt).order_by(
                PlanningProviderAttempt.attempt_number
            )
        )
    )
    assert client.calls == 2
    assert [item.attempt_kind for item in attempts] == ["INITIAL", "REPAIR"]
    assert [item.status for item in attempts] == ["REPAIR_REQUIRED", "SCHEMA_INVALID"]
    engine.dispose()


def test_restart_marks_pending_unknown_and_does_not_resubmit(tmp_path: Path) -> None:
    database_path = tmp_path / "restart.db"
    engine = create_sqlite_engine(database_path)
    init_database(engine)
    factory = create_session_factory(engine)
    with factory() as session:
        project = _project(session)
        session.add(
            PlanningProviderAttempt(
                project_id=project.id,
                job_id="job-3",
                planning_run_id="run-3",
                provider="kimi",
                model="kimi-k2.6",
                attempt_kind="INITIAL",
                attempt_number=1,
                request_revision="revision",
                input_token_estimate=100,
                configured_max_output_tokens=1000,
                status="PENDING",
                billing_status="UNKNOWN",
            )
        )
        session.commit()
    engine.dispose()

    restarted = create_sqlite_engine(database_path)
    init_database(restarted)
    restarted_factory = create_session_factory(restarted)
    with restarted_factory() as session:
        attempt = session.scalar(select(PlanningProviderAttempt))
        assert attempt is not None
        assert attempt.status == "INTERRUPTED_BILLING_UNKNOWN"
        client = _Client(["{}"])
        controller = PlanningAttemptController(
            session,
            project_id=attempt.project_id,
            job_id="job-4",
            planning_run_id="run-3",
            client=client,
        )
        with pytest.raises(PlanningBillingUncertainError):
            asyncio.run(controller.execute("no implicit retry", "REPAIR"))
        assert client.calls == 0
    restarted.dispose()


def test_explicit_retry_creates_second_visible_attempt_and_records_usage(
    tmp_path: Path,
) -> None:
    engine, session = _session(tmp_path)
    project = _project(session)
    first = _Client([_timeout()])
    initial = PlanningAttemptController(
        session,
        project_id=project.id,
        job_id="job-5",
        planning_run_id="run-5",
        client=first,
    )
    with pytest.raises(PlanningBillingUncertainError):
        asyncio.run(initial.execute("initial", "INITIAL"))

    second = _Client(["{}"])
    second.last_completion_metadata = StructuredCompletionMetadata(
        provider_request_id="provider-request-5",
        input_tokens=120,
        output_tokens=30,
        total_tokens=150,
    )
    retry = PlanningAttemptController(
        session,
        project_id=project.id,
        job_id="job-6",
        planning_run_id="run-5",
        client=second,
        allow_after_uncertain=True,
    )
    asyncio.run(retry.execute("explicit retry", "REPAIR"))

    attempts = list(
        session.scalars(
            select(PlanningProviderAttempt).order_by(
                PlanningProviderAttempt.attempt_number
            )
        )
    )
    assert len(attempts) == 2
    assert attempts[1].job_id == "job-6"
    assert attempts[1].provider_request_id == "provider-request-5"
    assert attempts[1].total_tokens == 150
    engine.dispose()


def test_cancellation_prevents_provider_dispatch(tmp_path: Path) -> None:
    engine, session = _session(tmp_path)
    project = _project(session)
    client = _Client(["{}"])
    controller = PlanningAttemptController(
        session,
        project_id=project.id,
        job_id="job-7",
        planning_run_id="run-7",
        client=client,
        cancelled=lambda: True,
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(controller.execute("cancelled", "INITIAL"))
    assert client.calls == 0
    assert list(session.scalars(select(PlanningProviderAttempt))) == []
    engine.dispose()


def test_in_flight_local_cancellation_marks_remote_execution_unknown(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, session = _session(tmp_path)
        project = _project(session)
        cancel = asyncio.Event()
        started = asyncio.Event()

        class SlowClient(_Client):
            async def generate(self, prompt: str) -> str:
                del prompt
                self.calls += 1
                started.set()
                await asyncio.Event().wait()
                return "{}"

        client = SlowClient([])
        controller = PlanningAttemptController(
            session,
            project_id=project.id,
            job_id="job-cancel",
            planning_run_id="run-cancel",
            client=client,
            cancelled=cancel.is_set,
        )
        task = asyncio.create_task(controller.execute("prompt", "INITIAL"))
        await started.wait()
        cancel.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        attempt = session.scalar(select(PlanningProviderAttempt))
        assert attempt is not None
        assert attempt.status == "CANCELLED_LOCALLY_BILLING_UNKNOWN"
        assert attempt.remote_execution_status == "REMOTE_EXECUTION_STATUS_UNKNOWN"
        engine.dispose()

    asyncio.run(scenario())


def test_safe_error_never_persists_api_key(tmp_path: Path) -> None:
    engine, session = _session(tmp_path)
    project = _project(session)
    client = _Client(
        [StructuredAIProviderError("api_key=super-secret", user_summary="api_key=super-secret")]
    )
    controller = PlanningAttemptController(
        session,
        project_id=project.id,
        job_id="job-8",
        planning_run_id="run-8",
        client=client,
    )
    with pytest.raises(StructuredAIProviderError):
        asyncio.run(controller.execute("prompt", "INITIAL"))
    attempt = session.scalar(select(PlanningProviderAttempt))
    assert attempt is not None
    assert "super-secret" not in (attempt.safe_error or "")
    engine.dispose()


def test_preflight_blocks_oversized_planning_request(tmp_path: Path) -> None:
    engine, session = _session(tmp_path)
    project = _project(session)
    client = _Client(["{}"])
    controller = PlanningAttemptController(
        session,
        project_id=project.id,
        job_id="job-large",
        planning_run_id="run-large",
        client=client,
        max_input_tokens=2,
    )

    with pytest.raises(PlanningTooLargeError, match="PLANNING_TOO_LARGE"):
        asyncio.run(controller.execute("this prompt cannot fit", "INITIAL"))

    assert client.calls == 0
    assert list(session.scalars(select(PlanningProviderAttempt))) == []
    engine.dispose()


def test_unknown_billing_timeout_keeps_reserved_budget(tmp_path: Path) -> None:
    engine, session = _session(tmp_path)
    project = _project(session, planning_budget_amount=0.25)
    _price(session, 0.10)
    controller = PlanningAttemptController(
        session,
        project_id=project.id,
        job_id="job-timeout-budget",
        planning_run_id="run-timeout-budget",
        client=_Client([_timeout()]),
    )

    with pytest.raises(PlanningBillingUncertainError):
        asyncio.run(controller.execute("prompt", "INITIAL"))

    snapshot = planning_budget_snapshot(
        session, project.id, "run-timeout-budget"
    )
    assert snapshot.spent == 0
    assert snapshot.reserved_unknown == pytest.approx(0.10)
    assert snapshot.remaining == pytest.approx(0.15)
    engine.dispose()


def test_second_call_is_blocked_when_unknown_reservation_would_exceed_budget(
    tmp_path: Path,
) -> None:
    engine, session = _session(tmp_path)
    project = _project(session, planning_budget_amount=0.15)
    _price(session, 0.10)
    first = PlanningAttemptController(
        session,
        project_id=project.id,
        job_id="job-first",
        planning_run_id="run-shared-budget",
        client=_Client([_timeout()]),
    )
    with pytest.raises(PlanningBillingUncertainError):
        asyncio.run(first.execute("prompt", "INITIAL"))

    second_client = _Client(["{}"])
    retry = PlanningAttemptController(
        session,
        project_id=project.id,
        job_id="job-second",
        planning_run_id="run-shared-budget",
        client=second_client,
        allow_after_uncertain=True,
    )
    with pytest.raises(PlanningBudgetError) as error:
        asyncio.run(retry.execute("repair", "REPAIR"))
    assert error.value.code == "PLANNING_BUDGET_EXHAUSTED"
    assert second_client.calls == 0
    assert len(list(session.scalars(select(PlanningProviderAttempt)))) == 1
    engine.dispose()


def test_successful_provider_usage_reconciles_reservation(tmp_path: Path) -> None:
    engine, session = _session(tmp_path)
    project = _project(session, planning_budget_amount=1.0)
    _price(session, 0.001, PricingUnit.PER_INPUT_TOKEN)
    _price(session, 0.002, PricingUnit.PER_OUTPUT_TOKEN)
    client = _Client(["{}"])
    client.max_output_tokens = 100
    client.last_completion_metadata = StructuredCompletionMetadata(
        input_tokens=10, output_tokens=5, total_tokens=15
    )
    controller = PlanningAttemptController(
        session,
        project_id=project.id,
        job_id="job-reconcile",
        planning_run_id="run-reconcile",
        client=client,
    )

    asyncio.run(controller.execute("prompt", "INITIAL"))

    attempt = session.scalar(select(PlanningProviderAttempt))
    assert attempt is not None
    assert float(attempt.estimated_max_cost) > float(attempt.actual_cost)
    assert float(attempt.actual_cost) == pytest.approx(0.02)
    assert attempt.cost_certainty == "ESTIMATED"
    snapshot = planning_budget_snapshot(session, project.id, "run-reconcile")
    assert snapshot.spent == pytest.approx(0.02)
    assert snapshot.reserved_unknown == 0
    engine.dispose()


def test_cached_plan_has_zero_planning_cost_for_new_run(tmp_path: Path) -> None:
    engine, session = _session(tmp_path)
    project = _project(session, planning_budget_amount=0.25)
    snapshot = planning_budget_snapshot(session, project.id, "cached-run")
    assert snapshot.spent == 0
    assert snapshot.reserved_unknown == 0
    assert snapshot.remaining == pytest.approx(0.25)
    engine.dispose()


def test_repair_consumes_remaining_planning_budget(tmp_path: Path) -> None:
    engine, session = _session(tmp_path)
    project = _project(
        session, planning_budget_amount=0.15, planning_max_paid_requests=5
    )
    _price(session, 0.10)
    client = _Client(["{}", "{}"])
    controller = PlanningAttemptController(
        session,
        project_id=project.id,
        job_id="job-repair-budget",
        planning_run_id="run-repair-budget",
        client=client,
    )
    asyncio.run(controller.execute("initial", "INITIAL"))
    controller.mark_validation("REPAIR_REQUIRED")

    with pytest.raises(PlanningBudgetError):
        asyncio.run(controller.execute("repair", "REPAIR"))
    assert client.calls == 1
    engine.dispose()


def test_retries_cannot_exceed_planning_hard_guard(tmp_path: Path) -> None:
    engine, session = _session(tmp_path)
    project = _project(
        session, planning_budget_amount=0.25, planning_max_paid_requests=10
    )
    _price(session, 0.10)
    client = _Client(["{}", "{}", "{}"])
    controller = PlanningAttemptController(
        session,
        project_id=project.id,
        job_id="job-hard-guard",
        planning_run_id="run-hard-guard",
        client=client,
    )
    asyncio.run(controller.execute("one", "INITIAL"))
    asyncio.run(controller.execute("two", "REPAIR"))
    with pytest.raises(PlanningBudgetError):
        asyncio.run(controller.execute("three", "REPAIR"))
    assert client.calls == 2
    snapshot = planning_budget_snapshot(session, project.id, "run-hard-guard")
    assert snapshot.spent == pytest.approx(0.20)
    assert snapshot.remaining == pytest.approx(0.05)
    engine.dispose()


def test_running_request_and_success_have_observable_lifecycle(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        engine, session = _session(tmp_path)
        project = _project(session)
        started = asyncio.Event()
        release = asyncio.Event()

        class ObservableClient(_Client):
            async def generate(self, prompt: str) -> str:
                del prompt
                self.calls += 1
                started.set()
                await release.wait()
                return "{}"

        client = ObservableClient([])
        controller = PlanningAttemptController(
            session,
            project_id=project.id,
            job_id="job-observable",
            planning_run_id="run-observable",
            client=client,
        )
        with caplog.at_level(logging.INFO, logger="app.services.planning_attempts"):
            task = asyncio.create_task(controller.execute("prompt", "INITIAL"))
            await started.wait()
            attempt = session.scalar(select(PlanningProviderAttempt))
            assert attempt is not None
            assert attempt.progress_state == "WAITING_FOR_PROVIDER"
            assert attempt.dispatched_at is not None
            assert attempt.finished_at is None
            release.set()
            await task
            session.refresh(attempt)
            assert attempt.progress_state == "VALIDATING_RESPONSE"
            assert attempt.response_received_at is not None
            controller.mark_validation_started()
            controller.mark_validation("SUCCEEDED")
            session.refresh(attempt)
            assert attempt.progress_state == "COMPLETED"
            assert attempt.finished_at is not None

        events = {record.getMessage() for record in caplog.records}
        assert {
            "planning_attempt_created",
            "planning_request_dispatched",
            "planning_response_received",
            "planning_validation_started",
            "planning_completed",
        } <= events
        logged = next(
            record
            for record in caplog.records
            if record.getMessage() == "planning_request_dispatched"
        )
        assert logged.job_id == "job-observable"
        assert logged.project_id == project.id
        assert logged.planning_attempt == 1
        assert logged.provider == "kimi"
        assert logged.model == "kimi-k2.6"
        assert logged.elapsed_seconds >= 0
        engine.dispose()

    asyncio.run(scenario())


def test_repair_and_provider_failure_have_distinct_progress_states(
    tmp_path: Path,
) -> None:
    engine, session = _session(tmp_path)
    project = _project(session)
    client = _Client(["{}", StructuredAIProviderError("safe failure")])
    controller = PlanningAttemptController(
        session,
        project_id=project.id,
        job_id="job-repair-state",
        planning_run_id="run-repair-state",
        client=client,
    )
    asyncio.run(controller.execute("initial", "INITIAL"))
    controller.mark_validation(
        "REPAIR_REQUIRED", "unknown reference", "UNKNOWN_REFERENCE"
    )
    first = session.scalar(
        select(PlanningProviderAttempt).where(
            PlanningProviderAttempt.attempt_number == 1
        )
    )
    assert first is not None
    assert first.progress_state == "REPAIRING_PLAN"
    assert first.validation_category == "UNKNOWN_REFERENCE"

    with pytest.raises(StructuredAIProviderError):
        asyncio.run(controller.execute("repair", "REPAIR"))
    second = session.scalar(
        select(PlanningProviderAttempt).where(
            PlanningProviderAttempt.attempt_number == 2
        )
    )
    assert second is not None
    assert second.attempt_kind == "REPAIR"
    assert second.progress_state == "FAILED"
    engine.dispose()
