"""Persisted Project integration for the existing Visual Director."""

import hashlib
import logging
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session

from app.costs import PricingUnit, UsageStatus, record_provider_usage, usage_revision
from app.errors import (
    ProjectVisualPlanError,
    StaleProjectVisualPlanError,
    VisualDirectorError,
)
from app.generators.visual_director import VisualDirector, VisualPlanningClient
from app.models.visual_plan import VisualPlan
from app.repositories import (
    get_project,
    get_project_visual_plan_record,
    save_project_visual_plan_record,
)

VISUAL_PLAN_SCHEMA_VERSION = "visual_plan_v1"
VISUAL_DIRECTOR_VERSION = "visual_director_v1"

logger = logging.getLogger(__name__)


class VisualPlanStatus(str, Enum):
    CURRENT = "CURRENT"
    STALE = "STALE"


@dataclass(frozen=True, slots=True)
class ProjectVisualPlanState:
    """Validated semantic plan together with its story freshness."""

    plan: VisualPlan
    record_id: str
    status: VisualPlanStatus

    @property
    def is_current(self) -> bool:
        return self.status is VisualPlanStatus.CURRENT


def hash_story_text(story_text: str) -> str:
    """Hash the exact persisted narration text as UTF-8."""
    return hashlib.sha256(story_text.encode("utf-8")).hexdigest()


async def create_project_visual_plan(
    session: Session,
    project_id: str,
    planning_client: VisualPlanningClient,
    *,
    job_id: str | None = None,
) -> VisualPlan:
    """Generate, validate, and atomically persist a Project's current plan."""
    project = get_project(session, project_id)
    if project is None:
        raise ValueError(f"Project not found: {project_id}")

    story_text = project.story_text.strip()
    if not story_text:
        raise ValueError("Project story_text must not be empty")

    _log(
        logging.INFO,
        "Visual planning started",
        project_id=project_id,
        event="planning_start",
    )
    try:
        plan = await VisualDirector(planning_client).create_plan(story_text)
    except VisualDirectorError as exc:
        if job_id is not None:
            record_provider_usage(
                session, project_id=project_id, job_id=job_id,
                pipeline_stage="PLANNING", provider=project.planning_provider,
                model=project.planning_model, operation="PLANNING",
                request_revision=usage_revision(hash_story_text(story_text), "planning"),
                unit_type=PricingUnit.PER_REQUEST, input_units=1,
                status=UsageStatus.FAILED,
            )
        failure_type = (
            "provider_failure"
            if str(exc) == "Visual planning provider failed"
            else "validation_failure"
        )
        _log(
            logging.WARNING,
            "Visual planning failed",
            project_id=project_id,
            event=failure_type,
        )
        raise

    if job_id is not None:
        record_provider_usage(
            session, project_id=project_id, job_id=job_id,
            pipeline_stage="PLANNING", provider=project.planning_provider,
            model=project.planning_model, operation="PLANNING",
            request_revision=usage_revision(hash_story_text(story_text), "planning"),
            unit_type=PricingUnit.PER_REQUEST, input_units=1,
            status=UsageStatus.SUCCEEDED,
        )

    _log(
        logging.INFO,
        "Visual planning succeeded",
        project_id=project_id,
        event="planning_success",
        beat_count=len(plan.visual_beats),
    )
    try:
        record = save_project_visual_plan_record(
            session,
            project_id=project_id,
            schema_version=VISUAL_PLAN_SCHEMA_VERSION,
            visual_director_version=VISUAL_DIRECTOR_VERSION,
            story_text_hash=hash_story_text(story_text),
            plan_json=plan.model_dump(mode="json"),
        )
    except Exception:
        _log(
            logging.ERROR,
            "Visual plan persistence failed",
            project_id=project_id,
            event="persistence_failure",
        )
        raise

    _log(
        logging.INFO,
        "Visual plan persisted",
        project_id=project_id,
        event="persistence_success",
        plan_id=record.id,
    )
    return VisualPlan.model_validate(record.plan_json)


def load_project_visual_plan(
    session: Session,
    project_id: str,
) -> VisualPlan | None:
    """Reload and revalidate a Project's current plan from durable storage."""
    record = get_project_visual_plan_record(session, project_id)
    if record is None:
        return None
    try:
        return VisualPlan.model_validate(record.plan_json)
    except Exception as exc:
        _log(
            logging.ERROR,
            "Persisted visual plan validation failed",
            project_id=project_id,
            event="persisted_validation_failure",
            plan_id=record.id,
        )
        raise ProjectVisualPlanError(
            "Persisted Project visual plan is invalid"
        ) from exc


def load_project_visual_plan_state(
    session: Session,
    project_id: str,
) -> ProjectVisualPlanState | None:
    """Load a plan for inspection and explicitly report CURRENT or STALE."""
    project = get_project(session, project_id)
    if project is None:
        raise ValueError(f"Project not found: {project_id}")
    record = get_project_visual_plan_record(session, project_id)
    if record is None:
        return None
    plan = load_project_visual_plan(session, project_id)
    assert plan is not None
    status = (
        VisualPlanStatus.CURRENT
        if record.story_text_hash == hash_story_text(project.story_text)
        else VisualPlanStatus.STALE
    )
    return ProjectVisualPlanState(
        plan=plan,
        record_id=record.id,
        status=status,
    )


def require_current_project_visual_plan(
    session: Session,
    project_id: str,
) -> ProjectVisualPlanState:
    """Reject missing or stale plans at an execution boundary."""
    state = load_project_visual_plan_state(session, project_id)
    if state is None:
        raise ProjectVisualPlanError("Project does not have a visual plan")
    if not state.is_current:
        _log(
            logging.WARNING,
            "Stale visual plan rejected for execution",
            project_id=project_id,
            event="stale_plan_rejected",
            plan_id=state.record_id,
        )
        raise StaleProjectVisualPlanError(
            "Project story changed; regenerate the visual plan before execution"
        )
    return state


def _log(
    level: int,
    message: str,
    *,
    project_id: str,
    event: str,
    beat_count: int | None = None,
    plan_id: str | None = None,
) -> None:
    extra: dict[str, str | int] = {
        "project_id": project_id,
        "visual_planning_event": event,
        "schema_version": VISUAL_PLAN_SCHEMA_VERSION,
        "visual_director_version": VISUAL_DIRECTOR_VERSION,
    }
    if beat_count is not None:
        extra["beat_count"] = beat_count
    if plan_id is not None:
        extra["plan_id"] = plan_id
    logger.log(level, message, extra=extra)
