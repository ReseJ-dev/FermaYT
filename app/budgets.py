"""Conservative project generation-budget enforcement."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

from sqlalchemy.orm import Session

from app.costs import (
    CostEstimate,
    PricingUnit,
    find_pricing,
    summarize_project_cost,
)
from app.persistence import Project

logger = logging.getLogger(__name__)

BUDGET_ESTIMATE_EXCEEDS_LIMIT = "BUDGET_ESTIMATE_EXCEEDS_LIMIT"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
BUDGET_EXHAUSTED_DURING_QA = "BUDGET_EXHAUSTED_DURING_QA"
BUDGET_COST_UNKNOWN = "BUDGET_COST_UNKNOWN"
BUDGET_CURRENCY_MISMATCH = "BUDGET_CURRENCY_MISMATCH"


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    enabled: bool
    amount: float | None
    currency: str
    spent: float | None
    available: float | None
    used_percent: float | None
    warning_threshold_percent: float
    warning: str | None
    override: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class GenerationBudgetError(RuntimeError):
    """A safe, structured pause reason raised before a paid request."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        snapshot: BudgetSnapshot,
        pipeline_stage: str,
        next_call_cost: float | None = None,
        provider: str | None = None,
        model: str | None = None,
        operation: str | None = None,
        beat_id: str | None = None,
        master_scene_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.user_summary = message
        self.snapshot = snapshot
        self.pipeline_stage = pipeline_stage
        self.next_call_cost = next_call_cost
        self.provider = provider
        self.model = model
        self.operation = operation
        self.beat_id = beat_id
        self.master_scene_id = master_scene_id

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": str(self),
            "pipeline_stage": self.pipeline_stage,
            "next_call_cost": self.next_call_cost,
            "provider": self.provider,
            "model": self.model,
            "operation": self.operation,
            "beat_id": self.beat_id,
            "master_scene_id": self.master_scene_id,
            "budget": self.snapshot.as_dict(),
        }


class ProjectBudgetGuard:
    """Check the project-wide ledger immediately before every paid call."""

    def __init__(
        self,
        session: Session,
        project_id: str,
        *,
        override: bool = False,
    ) -> None:
        project = session.get(Project, project_id)
        if project is None:
            raise ValueError("Project not found")
        self.session = session
        self.project = project
        self.override = override

    @property
    def enabled(self) -> bool:
        return bool(self.project.generation_budget_enabled)

    def snapshot(self) -> BudgetSnapshot:
        amount = (
            float(self.project.generation_budget_amount)
            if self.project.generation_budget_amount is not None
            else None
        )
        summary = summarize_project_cost(self.session, self.project.id)
        spent = summary.historical_project_cost
        available = (
            max(amount - spent, 0.0)
            if amount is not None and spent is not None
            else None
        )
        used_percent = (
            spent / amount * 100
            if amount is not None and amount > 0 and spent is not None
            else None
        )
        threshold = float(self.project.generation_budget_warning_threshold) * 100
        warning = (
            f"Project has used {used_percent:.0f}% of generation budget."
            if used_percent is not None and used_percent >= threshold
            else None
        )
        return BudgetSnapshot(
            enabled=self.enabled,
            amount=amount,
            currency=self.project.generation_budget_currency,
            spent=spent,
            available=available,
            used_percent=used_percent,
            warning_threshold_percent=threshold,
            warning=warning,
            override=self.override,
        )

    def check_preflight(self, estimate: CostEstimate) -> None:
        if not self.enabled or self.override:
            return
        snapshot = self._validated_snapshot("BUDGET_PREFLIGHT")
        if estimate.maximum is None or estimate.currency is None:
            self._raise_unknown(snapshot, "BUDGET_PREFLIGHT")
        if estimate.currency != snapshot.currency:
            raise GenerationBudgetError(
                BUDGET_CURRENCY_MISMATCH,
                "Budget currency does not match configured provider pricing.",
                snapshot=snapshot,
                pipeline_stage="BUDGET_PREFLIGHT",
            )
        assert snapshot.available is not None
        if estimate.maximum > snapshot.available + 1e-9:
            raise GenerationBudgetError(
                BUDGET_ESTIMATE_EXCEEDS_LIMIT,
                "Estimated generation cost exceeds the project budget.",
                snapshot=snapshot,
                pipeline_stage="BUDGET_PREFLIGHT",
                next_call_cost=estimate.maximum,
            )

    def check_paid_call(
        self,
        *,
        pipeline_stage: str,
        provider: str,
        model: str | None,
        operation: str,
        unit_type: PricingUnit,
        input_units: float = 0,
        output_units: float = 0,
        is_qa_retry: bool = False,
        beat_id: str | None = None,
        master_scene_id: str | None = None,
    ) -> None:
        if not self.enabled or self.override:
            return
        snapshot = self._validated_snapshot(pipeline_stage)
        pricing = find_pricing(
            self.session, provider, model, operation, unit_type
        )
        if pricing is None:
            self._raise_unknown(
                snapshot, pipeline_stage, provider, model, operation,
                beat_id, master_scene_id,
            )
        assert pricing is not None
        if pricing.currency != snapshot.currency:
            raise GenerationBudgetError(
                BUDGET_CURRENCY_MISMATCH,
                "Budget currency does not match configured provider pricing.",
                snapshot=snapshot,
                pipeline_stage=pipeline_stage,
                provider=provider,
                model=model,
                operation=operation,
                beat_id=beat_id,
                master_scene_id=master_scene_id,
            )
        next_cost = float(pricing.price) * (input_units + output_units)
        assert snapshot.available is not None
        if next_cost > snapshot.available + 1e-9:
            code = (
                BUDGET_EXHAUSTED_DURING_QA
                if is_qa_retry
                else BUDGET_EXHAUSTED
            )
            raise GenerationBudgetError(
                code,
                (
                    "Budget exhausted during Visual QA retry."
                    if is_qa_retry
                    else "Project generation budget is exhausted."
                ),
                snapshot=snapshot,
                pipeline_stage=pipeline_stage,
                next_call_cost=next_cost,
                provider=provider,
                model=model,
                operation=operation,
                beat_id=beat_id,
                master_scene_id=master_scene_id,
            )
        projected_percent = (
            ((snapshot.spent or 0) + next_cost) / snapshot.amount * 100
            if snapshot.amount else None
        )
        if (
            projected_percent is not None
            and projected_percent >= snapshot.warning_threshold_percent
        ):
            logger.warning(
                "Project has used or reserved %.0f%% of generation budget",
                projected_percent,
                extra={"project_id": self.project.id, "pipeline_stage": pipeline_stage},
            )

    def _validated_snapshot(self, stage: str) -> BudgetSnapshot:
        snapshot = self.snapshot()
        if snapshot.amount is None:
            raise GenerationBudgetError(
                BUDGET_COST_UNKNOWN,
                "Generation budget is enabled but no budget amount is configured.",
                snapshot=snapshot,
                pipeline_stage=stage,
            )
        if snapshot.spent is None:
            self._raise_unknown(snapshot, stage)
        return snapshot

    def _raise_unknown(
        self,
        snapshot: BudgetSnapshot,
        stage: str,
        provider: str | None = None,
        model: str | None = None,
        operation: str | None = None,
        beat_id: str | None = None,
        master_scene_id: str | None = None,
    ) -> None:
        raise GenerationBudgetError(
            BUDGET_COST_UNKNOWN,
            "Provider cost is unknown; paid generation was paused conservatively.",
            snapshot=snapshot,
            pipeline_stage=stage,
            provider=provider,
            model=model,
            operation=operation,
            beat_id=beat_id,
            master_scene_id=master_scene_id,
        )
