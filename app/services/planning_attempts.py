"""Single authoritative controller for paid Visual Planning provider attempts."""

from __future__ import annotations

import asyncio
import logging
import math
import os
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from hashlib import sha256

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.budgets import (
    PlanningBudgetError,
    PlanningBudgetSnapshot,
    ProjectBudgetGuard,
)
from app.clients.structured_completion import StructuredCompletionMetadata
from app.costs import (
    CostCertainty,
    PricingUnit,
    UsageStatus,
    find_pricing,
    record_provider_usage,
)
from app.errors import (
    PlanningAttemptLimitError,
    PlanningBillingUncertainError,
    PlanningTooLargeError,
    StructuredAIProviderError,
)
from app.persistence import PlanningProviderAttempt, Project
from app.provider_diagnostics import (
    find_structured_ai_provider_diagnostic,
    sanitize_provider_message,
)

DEFAULT_MAX_PAID_PLANNING_REQUESTS = 2
DEFAULT_MAX_INPUT_TOKENS = 20_000
DEFAULT_MAX_OUTPUT_TOKENS = 32_768
DEFAULT_MAX_TOTAL_ESTIMATED_TOKENS = 100_000
PLANNING_BUDGET_EXHAUSTED = "PLANNING_BUDGET_EXHAUSTED"
PLANNING_COST_UNKNOWN = "PLANNING_COST_UNKNOWN"
UNKNOWN_BILLING_STATUSES = frozenset(
    {
        "TIMED_OUT_BILLING_UNKNOWN",
        "CANCELLED_LOCALLY_BILLING_UNKNOWN",
        "INTERRUPTED_BILLING_UNKNOWN",
        "FAILED_BILLING_POSSIBLE",
    }
)
logger = logging.getLogger(__name__)


class PlanningAttemptKind(str, Enum):
    INITIAL = "INITIAL"
    REPAIR = "REPAIR"


class PlanningAttemptStatus(str, Enum):
    PENDING = "PENDING"
    SUCCEEDED = "SUCCEEDED"
    FAILED_NOT_BILLED_CONFIRMED = "FAILED_NOT_BILLED_CONFIRMED"
    FAILED_BILLING_POSSIBLE = "FAILED_BILLING_POSSIBLE"
    TIMED_OUT_BILLING_UNKNOWN = "TIMED_OUT_BILLING_UNKNOWN"
    CANCELLED_LOCALLY_BILLING_UNKNOWN = "CANCELLED_LOCALLY_BILLING_UNKNOWN"
    INTERRUPTED_BILLING_UNKNOWN = "INTERRUPTED_BILLING_UNKNOWN"
    RATE_LIMITED_NOT_SUBMITTED = "RATE_LIMITED_NOT_SUBMITTED"
    INVALID_PROVIDER_RESPONSE = "INVALID_PROVIDER_RESPONSE"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    REPAIR_REQUIRED = "REPAIR_REQUIRED"


class BillingStatus(str, Enum):
    KNOWN = "KNOWN"
    ESTIMATED = "ESTIMATED"
    UNKNOWN = "UNKNOWN"


class PlanningProgressState(str, Enum):
    PREPARING_SCOPE = "PREPARING_SCOPE"
    ESTIMATING_COST = "ESTIMATING_COST"
    WAITING_FOR_PROVIDER = "WAITING_FOR_PROVIDER"
    VALIDATING_RESPONSE = "VALIDATING_RESPONSE"
    REPAIRING_PLAN = "REPAIRING_PLAN"
    PAUSED_AFTER_TIMEOUT = "PAUSED_AFTER_TIMEOUT"
    PAUSED_BUDGET = "PAUSED_BUDGET"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class _PlanningPrice:
    def __init__(
        self,
        maximum: float | None,
        currency: str | None,
        mode: str,
    ) -> None:
        self.maximum = maximum
        self.currency = currency
        self.mode = mode


class PlanningAttemptController:
    """Persist, dispatch, classify, and globally limit planning POSTs."""

    def __init__(
        self,
        session: Session,
        *,
        project_id: str,
        job_id: str,
        planning_run_id: str,
        client: object,
        budget_guard: ProjectBudgetGuard | None = None,
        max_paid_requests: int | None = None,
        allow_after_uncertain: bool = False,
        cancelled: Callable[[], bool] | None = None,
        provider: str | None = None,
        model: str | None = None,
        expected_beat_count: int | None = None,
        planning_budget_amount: float | None = None,
        max_input_tokens: int | None = None,
        max_output_tokens: int | None = None,
        max_total_estimated_tokens: int | None = None,
    ) -> None:
        project = session.get(Project, project_id)
        if project is None:
            raise ValueError("Project not found")
        configured = (
            max_paid_requests
            if max_paid_requests is not None
            else int(project.planning_max_paid_requests)
        )
        if configured < 1:
            raise ValueError("max_paid_planning_requests must be positive")
        self.session = session
        self.project_id = project_id
        self.job_id = job_id
        self.planning_run_id = planning_run_id
        self.client = client
        self.budget_guard = budget_guard
        self.max_paid_requests = configured
        self.allow_after_uncertain = allow_after_uncertain
        self.cancelled = cancelled or (lambda: False)
        self._provider = provider
        self._model = model
        self.expected_beat_count = expected_beat_count
        self.planning_budget_amount = (
            planning_budget_amount
            if planning_budget_amount is not None
            else (
                float(project.planning_budget_amount)
                if project.planning_budget_amount is not None
                else None
            )
        )
        self.planning_currency = project.generation_budget_currency
        self.max_input_tokens = max_input_tokens or int(project.planning_max_input_tokens)
        self.max_output_tokens = max_output_tokens or int(project.planning_max_output_tokens)
        self.max_total_estimated_tokens = (
            max_total_estimated_tokens
            or int(project.planning_max_total_estimated_tokens)
        )
        if min(
            self.max_input_tokens,
            self.max_output_tokens,
            self.max_total_estimated_tokens,
        ) < 1:
            raise ValueError("planning token limits must be positive")
        self.current_attempt: PlanningProviderAttempt | None = None

    @property
    def provider(self) -> str:
        return self._provider or str(getattr(self.client, "provider", "unknown"))

    @property
    def model(self) -> str:
        return self._model or str(getattr(self.client, "model", "unknown"))

    @property
    def can_dispatch(self) -> bool:
        return len(self._attempts()) < self.max_paid_requests and not self.cancelled()

    async def execute(
        self, prompt: str, kind: PlanningAttemptKind | str
    ) -> str:
        kind = PlanningAttemptKind(kind)
        if self.cancelled():
            raise asyncio.CancelledError
        input_estimate = max(1, math.ceil(len(prompt) / 4))
        requested_output = int(getattr(self.client, "max_output_tokens", 0) or 0)
        self._check_token_limits(input_estimate, requested_output)
        price = _estimate_planning_price(
            self.session,
            self.provider,
            self.model,
            input_estimate,
            requested_output,
        )
        previous = self._attempts()
        uncertain = next(
            (item for item in reversed(previous) if item.status in UNKNOWN_BILLING_STATUSES),
            None,
        )
        if uncertain is not None and not self.allow_after_uncertain:
            raise PlanningBillingUncertainError(uncertain.id, uncertain.status)
        if len(previous) >= self.max_paid_requests:
            raise PlanningAttemptLimitError(self.max_paid_requests)
        snapshot = planning_budget_snapshot(
            self.session,
            self.project_id,
            self.planning_run_id,
            amount=self.planning_budget_amount,
            currency=self.planning_currency,
            next_estimated_cost=price.maximum,
        )
        if self.planning_budget_amount is not None:
            if price.maximum is None or price.currency is None:
                raise PlanningBudgetError(
                    PLANNING_COST_UNKNOWN,
                    "Planning cost cannot be estimated from configured pricing; request was not sent.",
                    planning_snapshot=snapshot,
                    provider=self.provider,
                    model=self.model,
                )
            if price.currency != self.planning_currency:
                raise PlanningBudgetError(
                    PLANNING_COST_UNKNOWN,
                    "Planning price currency does not match the project budget currency.",
                    planning_snapshot=snapshot,
                    provider=self.provider,
                    model=self.model,
                )
            if snapshot.remaining is None or price.maximum > snapshot.remaining + 1e-9:
                raise PlanningBudgetError(
                    PLANNING_BUDGET_EXHAUSTED,
                    "Planning budget is exhausted; increase it explicitly to resume.",
                    planning_snapshot=snapshot,
                    provider=self.provider,
                    model=self.model,
                )
        if self.budget_guard is not None:
            self.budget_guard.check_paid_call(
                pipeline_stage="PLANNING",
                provider=self.provider,
                model=self.model,
                operation="PLANNING",
                unit_type=PricingUnit.PER_REQUEST,
                input_units=1,
            )

        attempt_number = len(previous) + 1
        request_revision = sha256(
            f"{self.planning_run_id}\x1f{attempt_number}\x1f{kind.value}\x1f{prompt}".encode()
        ).hexdigest()
        record = PlanningProviderAttempt(
            project_id=self.project_id,
            job_id=self.job_id,
            planning_run_id=self.planning_run_id,
            provider=self.provider,
            model=self.model,
            attempt_kind=kind.value,
            attempt_number=attempt_number,
            request_revision=request_revision,
            input_token_estimate=input_estimate,
            configured_max_output_tokens=requested_output,
            expected_beat_count=self.expected_beat_count,
            estimated_max_cost=price.maximum,
            currency=price.currency,
            status=PlanningAttemptStatus.PENDING.value,
            billing_status=BillingStatus.UNKNOWN.value,
            cost_certainty=CostCertainty.UNKNOWN.value,
            progress_state=PlanningProgressState.WAITING_FOR_PROVIDER.value,
            started_at=datetime.now(UTC),
        )
        self.session.add(record)
        self.session.commit()  # Required durability boundary before network dispatch.
        self.session.refresh(record)
        self.current_attempt = record
        self._log_event("planning_attempt_created", record)

        try:
            record.dispatched_at = datetime.now(UTC)
            self.session.commit()
            self._log_event("planning_request_dispatched", record)
            provider_task = asyncio.create_task(
                self.client.generate(prompt)  # type: ignore[attr-defined]
            )
            while not provider_task.done():
                await asyncio.wait({provider_task}, timeout=0.2)
                if self.cancelled():
                    provider_task.cancel()
                    try:
                        await provider_task
                    except asyncio.CancelledError:
                        pass
                    raise asyncio.CancelledError
            raw = provider_task.result()
            record.response_received_at = datetime.now(UTC)
            record.progress_state = PlanningProgressState.VALIDATING_RESPONSE.value
            self.session.commit()
            self._log_event("planning_response_received", record)
        except asyncio.CancelledError:
            self._finish(
                record,
                PlanningAttemptStatus.CANCELLED_LOCALLY_BILLING_UNKNOWN,
                BillingStatus.UNKNOWN,
                safe_error="Local wait cancelled; remote execution and billing are unknown",
                remote_execution_status="REMOTE_EXECUTION_STATUS_UNKNOWN",
            )
            raise
        except StructuredAIProviderError as exc:
            diagnostic = find_structured_ai_provider_diagnostic(exc)
            category = diagnostic.category if diagnostic is not None else ""
            if category == "PLANNING_TIMEOUT":
                status = PlanningAttemptStatus.TIMED_OUT_BILLING_UNKNOWN
            elif category in {
                "PLANNING_INVALID_JSON",
                "PLANNING_EMPTY_RESPONSE",
                "PLANNING_TRUNCATED_OUTPUT",
                "PLANNING_STRUCTURED_JSON_INCOMPATIBILITY",
            }:
                status = PlanningAttemptStatus.INVALID_PROVIDER_RESPONSE
            elif category in {"PLANNING_AUTH_ERROR", "PLANNING_BAD_REQUEST"}:
                status = PlanningAttemptStatus.FAILED_NOT_BILLED_CONFIRMED
            else:
                status = PlanningAttemptStatus.FAILED_BILLING_POSSIBLE
            billing = (
                BillingStatus.KNOWN
                if status is PlanningAttemptStatus.FAILED_NOT_BILLED_CONFIRMED
                else BillingStatus.UNKNOWN
            )
            self._finish(
                record,
                status,
                billing,
                provider_request_id=(diagnostic.request_id if diagnostic else None),
                safe_error=_safe_error(exc),
                remote_execution_status=(
                    "REMOTE_EXECUTION_STATUS_UNKNOWN"
                    if billing is BillingStatus.UNKNOWN
                    else None
                ),
            )
            if status is PlanningAttemptStatus.TIMED_OUT_BILLING_UNKNOWN:
                self._log_event("planning_timeout_billing_unknown", record)
                raise PlanningBillingUncertainError(record.id, status.value) from exc
            raise
        except Exception as exc:
            self._finish(
                record,
                PlanningAttemptStatus.FAILED_BILLING_POSSIBLE,
                BillingStatus.UNKNOWN,
                safe_error=_safe_error(exc),
                remote_execution_status="REMOTE_EXECUTION_STATUS_UNKNOWN",
            )
            raise

        metadata = getattr(self.client, "last_completion_metadata", None)
        if not isinstance(metadata, StructuredCompletionMetadata):
            metadata = StructuredCompletionMetadata()
        self._finish(
            record,
            PlanningAttemptStatus.SUCCEEDED,
            BillingStatus.ESTIMATED if price.maximum is not None else BillingStatus.UNKNOWN,
            provider_request_id=metadata.provider_request_id,
            metadata=metadata,
            price_mode=price.mode,
            terminal=False,
        )
        return raw

    def mark_validation_started(self) -> None:
        if self.current_attempt is None:
            return
        self.current_attempt.progress_state = (
            PlanningProgressState.VALIDATING_RESPONSE.value
        )
        self.current_attempt.validation_started_at = datetime.now(UTC)
        self.session.commit()
        self._log_event("planning_validation_started", self.current_attempt)

    def mark_validation(
        self,
        status: PlanningAttemptStatus | str,
        safe_error: str | None = None,
        validation_category: str | None = None,
    ) -> None:
        status = PlanningAttemptStatus(status)
        if status not in {
            PlanningAttemptStatus.SUCCEEDED,
            PlanningAttemptStatus.SCHEMA_INVALID,
            PlanningAttemptStatus.REPAIR_REQUIRED,
        }:
            raise ValueError("invalid validation outcome")
        if self.current_attempt is None:
            return
        self.current_attempt.status = status.value
        self.current_attempt.safe_error = (
            (sanitize_provider_message(safe_error) or "Schema validation failed")[:500]
            if safe_error
            else None
        )
        safe_category = sanitize_provider_message(validation_category)
        self.current_attempt.validation_category = (
            safe_category[:100] if safe_category else None
        )
        if status is PlanningAttemptStatus.REPAIR_REQUIRED:
            self.current_attempt.progress_state = (
                PlanningProgressState.REPAIRING_PLAN.value
            )
            self._log_event("planning_repair_requested", self.current_attempt)
        elif status is PlanningAttemptStatus.SUCCEEDED:
            self.current_attempt.progress_state = PlanningProgressState.COMPLETED.value
            self.current_attempt.finished_at = datetime.now(UTC)
            self._log_event("planning_completed", self.current_attempt)
        else:
            self.current_attempt.progress_state = PlanningProgressState.FAILED.value
            self.current_attempt.finished_at = datetime.now(UTC)
        self.session.commit()

    def _log_event(
        self, event: str, record: PlanningProviderAttempt
    ) -> None:
        now = datetime.now(UTC)
        start = record.dispatched_at or record.started_at
        end = record.response_received_at or record.finished_at or now
        elapsed = max((end - start).total_seconds(), 0.0)
        logger.info(
            event,
            extra={
                "planning_event": event,
                "job_id": self.job_id,
                "project_id": self.project_id,
                "planning_attempt": record.attempt_number,
                "planning_attempt_kind": record.attempt_kind,
                "provider": record.provider,
                "model": record.model,
                "elapsed_seconds": round(elapsed, 3),
                "cost_certainty": record.cost_certainty,
                "estimated_max_cost": (
                    float(record.estimated_max_cost)
                    if record.estimated_max_cost is not None
                    else None
                ),
                "currency": record.currency,
            },
        )

    def _attempts(self) -> list[PlanningProviderAttempt]:
        return list(
            self.session.scalars(
                select(PlanningProviderAttempt)
                .where(PlanningProviderAttempt.planning_run_id == self.planning_run_id)
                .order_by(PlanningProviderAttempt.attempt_number)
            )
        )

    def _check_token_limits(self, input_tokens: int, output_tokens: int) -> None:
        if output_tokens < 1:
            raise PlanningTooLargeError("provider output limit is not configured")
        failures: list[str] = []
        if input_tokens > self.max_input_tokens:
            failures.append(
                f"input estimate {input_tokens} exceeds {self.max_input_tokens}"
            )
        if output_tokens > self.max_output_tokens:
            failures.append(
                f"max output {output_tokens} exceeds {self.max_output_tokens}"
            )
        prior_total = sum(
            item.input_token_estimate + item.configured_max_output_tokens
            for item in self._attempts()
        )
        total = prior_total + input_tokens + output_tokens
        if total > self.max_total_estimated_tokens:
            failures.append(
                f"planning-run token estimate {total} exceeds "
                f"{self.max_total_estimated_tokens}"
            )
        if failures:
            raise PlanningTooLargeError("; ".join(failures))

    def _finish(
        self,
        record: PlanningProviderAttempt,
        status: PlanningAttemptStatus,
        billing_status: BillingStatus,
        *,
        provider_request_id: str | None = None,
        metadata: StructuredCompletionMetadata | None = None,
        safe_error: str | None = None,
        remote_execution_status: str | None = None,
        price_mode: str | None = None,
        terminal: bool = True,
    ) -> None:
        record.status = status.value
        record.billing_status = billing_status.value
        record.provider_request_id = provider_request_id
        record.remote_execution_status = remote_execution_status
        record.safe_error = safe_error
        if terminal:
            record.finished_at = datetime.now(UTC)
            record.progress_state = (
                PlanningProgressState.PAUSED_AFTER_TIMEOUT.value
                if status
                in {
                    PlanningAttemptStatus.TIMED_OUT_BILLING_UNKNOWN,
                    PlanningAttemptStatus.CANCELLED_LOCALLY_BILLING_UNKNOWN,
                    PlanningAttemptStatus.INTERRUPTED_BILLING_UNKNOWN,
                }
                else PlanningProgressState.FAILED.value
            )
        confirmed_no_charge = (
            billing_status is BillingStatus.KNOWN
            and status
            in {
                PlanningAttemptStatus.FAILED_NOT_BILLED_CONFIRMED,
                PlanningAttemptStatus.RATE_LIMITED_NOT_SUBMITTED,
            }
        )
        if confirmed_no_charge:
            record.actual_cost = 0
            record.cost_certainty = CostCertainty.ACTUAL.value
        if metadata is not None:
            record.input_tokens = metadata.input_tokens
            record.output_tokens = metadata.output_tokens
            record.total_tokens = metadata.total_tokens
            reconciled = _reconcile_planning_cost(
                self.session,
                self.provider,
                self.model,
                metadata,
                price_mode or "",
                record.estimated_max_cost,
            )
            if reconciled is not None:
                record.actual_cost = reconciled
                record.cost_certainty = CostCertainty.ESTIMATED.value
            elif record.estimated_max_cost is not None:
                # No reliable token usage: retain the conservative estimate as cost.
                record.actual_cost = record.estimated_max_cost
                record.cost_certainty = CostCertainty.ESTIMATED.value
        self.session.commit()
        record_provider_usage(
            self.session,
            project_id=self.project_id,
            job_id=self.job_id,
            pipeline_stage="PLANNING",
            provider=self.provider,
            model=self.model,
            operation="PLANNING",
            request_revision=record.request_revision,
            unit_type=PricingUnit.PER_REQUEST,
            input_units=1,
            output_units=0,
            status=(
                UsageStatus.SUCCEEDED
                if status is PlanningAttemptStatus.SUCCEEDED
                else UsageStatus.FAILED
            ),
            actual_cost=(float(record.actual_cost) if record.actual_cost is not None else None),
            cost_certainty=record.cost_certainty,
        )


def planning_budget_snapshot(
    session: Session,
    project_id: str,
    planning_run_id: str,
    *,
    amount: float | None = None,
    currency: str | None = None,
    next_estimated_cost: float | None = None,
) -> PlanningBudgetSnapshot:
    project = session.get(Project, project_id)
    if project is None:
        raise ValueError("Project not found")
    configured_amount = (
        amount
        if amount is not None
        else (
            float(project.planning_budget_amount)
            if project.planning_budget_amount is not None
            else None
        )
    )
    configured_currency = currency or project.generation_budget_currency
    attempts = list(
        session.scalars(
            select(PlanningProviderAttempt).where(
                PlanningProviderAttempt.planning_run_id == planning_run_id
            )
        )
    )
    spent = 0.0
    reserved_unknown = 0.0
    for attempt in attempts:
        maximum = (
            float(attempt.estimated_max_cost)
            if attempt.estimated_max_cost is not None
            else 0.0
        )
        if attempt.cost_certainty == CostCertainty.UNKNOWN.value:
            reserved_unknown += maximum
        else:
            value = attempt.actual_cost
            spent += float(value) if value is not None else maximum
    committed = spent + reserved_unknown
    remaining = (
        max(configured_amount - committed, 0.0)
        if configured_amount is not None
        else None
    )
    return PlanningBudgetSnapshot(
        enabled=configured_amount is not None,
        amount=configured_amount,
        currency=configured_currency,
        spent=round(spent, 8),
        reserved_unknown=round(reserved_unknown, 8),
        remaining=round(remaining, 8) if remaining is not None else None,
        next_estimated_cost=next_estimated_cost,
    )


def _estimate_planning_price(
    session: Session,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> _PlanningPrice:
    per_request = find_pricing(
        session, provider, model, "PLANNING", PricingUnit.PER_REQUEST
    )
    if per_request is not None:
        return _PlanningPrice(float(per_request.price), per_request.currency, "request")
    per_token = find_pricing(
        session, provider, model, "PLANNING", PricingUnit.PER_TOKEN
    )
    if per_token is not None:
        return _PlanningPrice(
            round(float(per_token.price) * (input_tokens + output_tokens), 8),
            per_token.currency,
            "token",
        )
    input_price = find_pricing(
        session, provider, model, "PLANNING", PricingUnit.PER_INPUT_TOKEN
    )
    output_price = find_pricing(
        session, provider, model, "PLANNING", PricingUnit.PER_OUTPUT_TOKEN
    )
    if input_price is None or output_price is None:
        return _PlanningPrice(None, None, "unknown")
    currencies = {input_price.currency, output_price.currency}
    reasoning_price = find_pricing(
        session, provider, model, "PLANNING", PricingUnit.PER_REASONING_TOKEN
    )
    if reasoning_price is not None:
        currencies.add(reasoning_price.currency)
    if len(currencies) != 1:
        return _PlanningPrice(None, None, "unknown")
    maximum = (
        float(input_price.price) * input_tokens
        + float(output_price.price) * output_tokens
        + (float(reasoning_price.price) * output_tokens if reasoning_price else 0)
    )
    return _PlanningPrice(round(maximum, 8), currencies.pop(), "split_token")


def _reconcile_planning_cost(
    session: Session,
    provider: str,
    model: str,
    metadata: StructuredCompletionMetadata,
    mode: str,
    reserved: object,
) -> float | None:
    if mode == "request":
        return float(reserved) if reserved is not None else None
    if metadata.input_tokens is None or metadata.output_tokens is None:
        return None
    price = _estimate_planning_price(
        session,
        provider,
        model,
        metadata.input_tokens,
        metadata.output_tokens,
    )
    return price.maximum


def latest_uncertain_planning_attempt(
    session: Session, project_id: str
) -> PlanningProviderAttempt | None:
    return session.scalar(
        select(PlanningProviderAttempt)
        .where(
            PlanningProviderAttempt.project_id == project_id,
            PlanningProviderAttempt.status.in_(UNKNOWN_BILLING_STATUSES),
        )
        .order_by(PlanningProviderAttempt.started_at.desc())
        .limit(1)
    )


def configured_planning_attempt_limit() -> int:
    raw = os.getenv("MAX_PAID_PLANNING_REQUESTS", "").strip()
    try:
        value = int(raw) if raw else DEFAULT_MAX_PAID_PLANNING_REQUESTS
    except ValueError as exc:
        raise ValueError("MAX_PAID_PLANNING_REQUESTS must be an integer") from exc
    return value


def _safe_error(error: Exception) -> str:
    summary = getattr(error, "user_summary", None)
    value = summary if isinstance(summary, str) else type(error).__name__
    return (sanitize_provider_message(value) or type(error).__name__)[:500]
