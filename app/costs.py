"""Provider pricing, immutable usage ledger, summaries, and estimates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.persistence import (
    BeatVisualResult,
    MasterSceneAsset,
    Project,
    ProjectNarrationAsset,
    ProjectVisualExecutionPlan,
    ProviderPricing,
    ProviderUsageRecord,
    VisualOperationDecisionRecord,
)


class PricingUnit(str, Enum):
    PER_IMAGE = "PER_IMAGE"
    PER_CHARACTER = "PER_CHARACTER"
    PER_REQUEST = "PER_REQUEST"


class UsageStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CACHED = "CACHED"
    SKIPPED = "SKIPPED"


class CostCertainty(str, Enum):
    ACTUAL = "ACTUAL"
    ESTIMATED = "ESTIMATED"
    UNKNOWN = "UNKNOWN"


FREE_VISUAL_OPERATIONS = frozenset({"REUSE", "TRANSFORM", "OVERLAY"})


@dataclass(frozen=True, slots=True)
class CostSummary:
    run_cost: float | None
    historical_project_cost: float | None
    qa_retry_cost: float | None
    currency: str | None
    cost_by_stage: dict[str, float]
    cost_by_provider: dict[str, float]
    cost_by_model: dict[str, float]
    cost_by_beat: dict[str, float]
    unpriced_records: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CostEstimate:
    minimum: float | None
    maximum: float | None
    currency: str | None
    paid_image_calls: int
    free_visual_operations: int
    expected_qa_calls: int
    maximum_qa_retry_calls: int
    unpriced_operations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def configure_provider_pricing(
    session: Session,
    *,
    provider: str,
    model: str,
    operation: str,
    pricing_unit: PricingUnit | str,
    price: float,
    currency: str,
    version: str,
    effective_from: datetime,
) -> ProviderPricing:
    """Insert a new immutable price version; existing versions are never edited."""
    if price < 0:
        raise ValueError("price must not be negative")
    if effective_from.tzinfo is None:
        raise ValueError("effective_from must be timezone-aware")
    currency = currency.strip().upper()
    if len(currency) != 3:
        raise ValueError("currency must be a three-letter ISO code")
    normalized_provider = provider.strip().lower()
    normalized_model = model.strip()
    normalized_operation = operation.strip().upper()
    normalized_unit = PricingUnit(pricing_unit).value
    existing = session.scalar(select(ProviderPricing).where(
        ProviderPricing.provider == normalized_provider,
        ProviderPricing.model == normalized_model,
        ProviderPricing.operation == normalized_operation,
        ProviderPricing.pricing_unit == normalized_unit,
        ProviderPricing.version == version.strip(),
    ))
    if existing is not None:
        if (
            float(existing.price) != price or existing.currency != currency
            or existing.effective_from != effective_from.astimezone(UTC)
        ):
            raise ValueError("pricing version already exists with different values")
        return existing
    record = ProviderPricing(
        provider=normalized_provider,
        model=normalized_model,
        operation=normalized_operation,
        pricing_unit=normalized_unit,
        price=price,
        currency=currency,
        version=version.strip(),
        effective_from=effective_from.astimezone(UTC),
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def load_pricing_config(session: Session, path: str | Path) -> list[ProviderPricing]:
    """Load idempotent immutable price versions from a local JSON configuration."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("prices"), list):
        raise ValueError("pricing config must contain a prices list")
    records: list[ProviderPricing] = []
    for item in data["prices"]:
        if not isinstance(item, dict):
            raise ValueError("each pricing entry must be an object")
        records.append(configure_provider_pricing(
            session,
            provider=str(item["provider"]), model=str(item["model"]),
            operation=str(item["operation"]), pricing_unit=str(item["pricing_unit"]),
            price=float(item["price"]), currency=str(item["currency"]),
            version=str(item["version"]),
            effective_from=datetime.fromisoformat(str(item["effective_from"])),
        ))
    return records


def find_pricing(
    session: Session,
    provider: str,
    model: str | None,
    operation: str,
    unit_type: PricingUnit | str,
    *,
    at: datetime | None = None,
) -> ProviderPricing | None:
    """Return the latest effective configured price for an exact request shape."""
    moment = (at or datetime.now(UTC)).astimezone(UTC)
    return session.scalar(
        select(ProviderPricing)
        .where(
            ProviderPricing.provider == provider.strip().lower(),
            ProviderPricing.model == (model or ""),
            ProviderPricing.operation == operation.strip().upper(),
            ProviderPricing.pricing_unit == PricingUnit(unit_type).value,
            ProviderPricing.effective_from <= moment,
        )
        .order_by(ProviderPricing.effective_from.desc(), ProviderPricing.created_at.desc())
        .limit(1)
    )


def record_provider_usage(
    session: Session,
    *,
    project_id: str,
    job_id: str,
    pipeline_stage: str,
    provider: str,
    model: str | None,
    operation: str,
    request_revision: str,
    unit_type: PricingUnit | str,
    input_units: float = 0,
    output_units: float = 0,
    status: UsageStatus | str = UsageStatus.SUCCEEDED,
    beat_id: str | None = None,
    master_scene_id: str | None = None,
    actual_cost: float | None = None,
    is_qa_retry: bool = False,
) -> ProviderUsageRecord:
    """Append one idempotent ledger entry and snapshot the effective price version."""
    existing = session.scalar(
        select(ProviderUsageRecord).where(
            ProviderUsageRecord.job_id == job_id,
            ProviderUsageRecord.pipeline_stage == pipeline_stage,
            ProviderUsageRecord.request_revision == request_revision,
        )
    )
    if existing is not None:
        return existing
    normalized_status = UsageStatus(status)
    normalized_operation = operation.strip().upper()
    normalized_unit = PricingUnit(unit_type)
    is_free = normalized_status in {UsageStatus.CACHED, UsageStatus.SKIPPED} or (
        normalized_operation in FREE_VISUAL_OPERATIONS
    )
    pricing = None if is_free else find_pricing(
        session, provider, model, normalized_operation, normalized_unit
    )
    units = input_units + output_units
    estimated_cost = 0.0 if is_free else (
        round(float(pricing.price) * units, 8) if pricing is not None else None
    )
    if is_free:
        actual_cost = 0.0
        certainty = CostCertainty.ACTUAL
    elif normalized_status is UsageStatus.FAILED:
        certainty = CostCertainty.UNKNOWN
    elif pricing is None:
        certainty = CostCertainty.UNKNOWN
    elif actual_cost is not None:
        certainty = CostCertainty.ACTUAL
    else:
        certainty = CostCertainty.ESTIMATED
    record = ProviderUsageRecord(
        project_id=project_id,
        job_id=job_id,
        pipeline_stage=pipeline_stage,
        beat_id=beat_id,
        master_scene_id=master_scene_id,
        provider=provider.strip().lower(),
        model=model,
        operation=normalized_operation,
        request_revision=request_revision,
        estimated_cost=estimated_cost,
        actual_cost=actual_cost,
        currency=pricing.currency if pricing is not None else None,
        pricing_version=pricing.version if pricing is not None else None,
        input_units=input_units,
        output_units=output_units,
        unit_type=normalized_unit.value,
        status=normalized_status.value,
        cost_certainty=certainty.value,
        is_qa_retry=is_qa_retry,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def usage_revision(*parts: object) -> str:
    return sha256("\x1f".join(str(part) for part in parts).encode()).hexdigest()


def summarize_project_cost(
    session: Session, project_id: str, *, job_id: str | None = None
) -> CostSummary:
    all_records = list(session.scalars(
        select(ProviderUsageRecord)
        .where(ProviderUsageRecord.project_id == project_id)
        .order_by(ProviderUsageRecord.created_at)
    ))
    run_records = [record for record in all_records if job_id is None or record.job_id == job_id]
    currencies = {record.currency for record in all_records if record.currency}
    currency = next(iter(currencies)) if len(currencies) == 1 else None

    def valued(record: ProviderUsageRecord) -> float | None:
        value = record.actual_cost if record.actual_cost is not None else record.estimated_cost
        return float(value) if value is not None else None

    def total(records: list[ProviderUsageRecord]) -> float | None:
        values = [valued(record) for record in records]
        if not records:
            return 0.0
        if any(value is None for value in values):
            return None
        return round(sum(value for value in values if value is not None), 8)

    by_stage: defaultdict[str, float] = defaultdict(float)
    by_provider: defaultdict[str, float] = defaultdict(float)
    by_model: defaultdict[str, float] = defaultdict(float)
    by_beat: defaultdict[str, float] = defaultdict(float)
    for record in run_records:
        value = valued(record)
        if value is None:
            continue
        by_stage[record.pipeline_stage] += value
        by_provider[record.provider] += value
        by_model[record.model or "unknown"] += value
        if record.beat_id:
            by_beat[record.beat_id] += value
    return CostSummary(
        run_cost=total(run_records),
        historical_project_cost=total(all_records),
        qa_retry_cost=total([record for record in run_records if record.is_qa_retry]),
        currency=currency,
        cost_by_stage={key: round(value, 8) for key, value in by_stage.items()},
        cost_by_provider={key: round(value, 8) for key, value in by_provider.items()},
        cost_by_model={key: round(value, 8) for key, value in by_model.items()},
        cost_by_beat={key: round(value, 8) for key, value in by_beat.items()},
        unpriced_records=sum(
            record.estimated_cost is None and record.actual_cost is None
            and record.status not in {UsageStatus.CACHED.value, UsageStatus.SKIPPED.value}
            for record in run_records
        ),
    )


def estimate_operations(
    session: Session,
    *,
    provider: str,
    model: str | None,
    operations: list[str],
    qa_provider: str | None = None,
    qa_model: str | None = None,
    qa_enabled: bool = True,
    max_qa_attempts: int = 3,
) -> CostEstimate:
    """Estimate a resolved visual plan without claiming retries will occur."""
    normalized = [_pricing_operation(op) for op in operations]
    paid = [op for op in normalized if op not in FREE_VISUAL_OPERATIONS]
    free = len(operations) - len(paid)
    minimum = 0.0
    maximum = 0.0
    currencies: set[str] = set()
    missing: set[str] = set()
    for operation in paid:
        price = find_pricing(session, provider, model, operation, PricingUnit.PER_IMAGE)
        if price is None:
            missing.add(f"{provider}/{model or ''}/{operation}")
            continue
        minimum += float(price.price)
        maximum += float(price.price) * max_qa_attempts
        currencies.add(price.currency)
    expected_qa = len(paid) if qa_enabled else 0
    max_retries = len(paid) * max(max_qa_attempts - 1, 0) if qa_enabled else 0
    if qa_enabled and qa_provider:
        qa_price = find_pricing(
            session, qa_provider, qa_model, "VISUAL_QA", PricingUnit.PER_REQUEST
        )
        if qa_price is None and expected_qa:
            missing.add(f"{qa_provider}/{qa_model or ''}/VISUAL_QA")
        elif qa_price is not None:
            minimum += float(qa_price.price) * expected_qa
            maximum += float(qa_price.price) * (expected_qa + max_retries)
            currencies.add(qa_price.currency)
    known_currency = next(iter(currencies)) if len(currencies) == 1 else None
    return CostEstimate(
        minimum=None if missing else round(minimum, 8),
        maximum=None if missing else round(maximum, 8),
        currency=known_currency,
        paid_image_calls=len(paid),
        free_visual_operations=free,
        expected_qa_calls=expected_qa,
        maximum_qa_retry_calls=max_retries,
        unpriced_operations=tuple(sorted(missing)),
    )


def estimate_project_generation_cost(
    session: Session,
    project_id: str,
    *,
    max_qa_attempts: int = 3,
) -> CostEstimate:
    """Estimate remaining paid work from the latest persisted resolved plan."""
    project = session.get(Project, project_id)
    if project is None:
        raise ValueError("Project not found")
    execution = session.scalar(
        select(ProjectVisualExecutionPlan)
        .where(ProjectVisualExecutionPlan.project_id == project_id)
        .order_by(ProjectVisualExecutionPlan.created_at.desc())
        .limit(1)
    )
    if execution is None or project.visual_plan is None:
        return CostEstimate(
            minimum=None, maximum=None, currency=None, paid_image_calls=0,
            free_visual_operations=0, expected_qa_calls=0,
            maximum_qa_retry_calls=0,
            unpriced_operations=("persisted_visual_plan_required",),
        )
    accepted_beats = set(session.scalars(
        select(BeatVisualResult.beat_id).where(
            BeatVisualResult.execution_plan_id == execution.id,
            BeatVisualResult.is_accepted.is_(True),
        )
    ))
    decisions = list(session.scalars(
        select(VisualOperationDecisionRecord).where(
            VisualOperationDecisionRecord.execution_plan_id == execution.id
        )
    ))
    remaining_operations = [
        decision.resolved_operation
        for decision in decisions
        if decision.beat_id not in accepted_beats
    ]
    existing_masters = set(session.scalars(
        select(MasterSceneAsset.master_scene_id).where(
            MasterSceneAsset.project_id == project_id
        )
    ))
    beats = project.visual_plan.plan_json.get("visual_beats", [])
    required_masters = {
        beat.get("master_scene_id") for beat in beats if beat.get("master_scene_id")
    }
    remaining_operations.extend(
        "NEW_IMAGE" for master_id in required_masters - existing_masters
    )
    estimate = estimate_operations(
        session,
        provider=execution.provider,
        model=execution.model,
        operations=remaining_operations,
        qa_provider=project.visual_qa_provider,
        qa_model=project.visual_qa_model,
        qa_enabled=project.visual_qa_enabled,
        max_qa_attempts=max_qa_attempts,
    )
    story_hash = sha256(project.story_text.strip().encode()).hexdigest()
    narration_cached = session.scalar(
        select(ProjectNarrationAsset.id).where(
            ProjectNarrationAsset.project_id == project_id,
            ProjectNarrationAsset.story_text_hash == story_hash,
        ).limit(1)
    ) is not None
    if narration_cached:
        return estimate
    tts_price = find_pricing(
        session, project.tts_provider, project.tts_model, "TTS",
        PricingUnit.PER_CHARACTER,
    )
    missing = set(estimate.unpriced_operations)
    minimum = estimate.minimum
    maximum = estimate.maximum
    currency = estimate.currency
    if tts_price is None:
        missing.add(f"{project.tts_provider}/{project.tts_model or ''}/TTS")
        minimum = maximum = None
    elif minimum is not None and maximum is not None:
        tts_cost = float(tts_price.price) * len(project.story_text.strip())
        minimum = round(minimum + tts_cost, 8)
        maximum = round(maximum + tts_cost, 8)
        currency = tts_price.currency if currency in {None, tts_price.currency} else None
    return CostEstimate(
        minimum=minimum,
        maximum=maximum,
        currency=currency,
        paid_image_calls=estimate.paid_image_calls,
        free_visual_operations=estimate.free_visual_operations,
        expected_qa_calls=estimate.expected_qa_calls,
        maximum_qa_retry_calls=estimate.maximum_qa_retry_calls,
        unpriced_operations=tuple(sorted(missing)),
    )


def _pricing_operation(operation: str) -> str:
    normalized = operation.strip().upper()
    return "EDIT" if normalized == "EDIT_EXISTING" else normalized
