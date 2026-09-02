"""Data-level manual controls for persisted visual QA candidates."""

from sqlalchemy.orm import Session

from app.models.beat_visual import ManualVisualQAOverride
from app.persistence import BeatVisualResult
from app.repositories import get_beat_visual_result, set_manual_visual_qa_override


class VisualQAManualOverrideService:
    """Explicit human decisions kept separate from automated QA evidence."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def accept(self, result_id: str, reason: str) -> BeatVisualResult:
        return self._set(result_id, ManualVisualQAOverride.ACCEPTED, reason)

    def reject(self, result_id: str, reason: str) -> BeatVisualResult:
        return self._set(result_id, ManualVisualQAOverride.REJECTED, reason)

    def request_regeneration(
        self,
        result_id: str,
        reason: str,
    ) -> BeatVisualResult:
        return self._set(
            result_id,
            ManualVisualQAOverride.REGENERATE_REQUESTED,
            reason,
        )

    def _set(
        self,
        result_id: str,
        override: ManualVisualQAOverride,
        reason: str,
    ) -> BeatVisualResult:
        result = get_beat_visual_result(self.session, result_id)
        if result is None:
            raise ValueError(f"Beat visual candidate not found: {result_id}")
        return set_manual_visual_qa_override(
            self.session,
            result,
            override,
            reason=reason,
        )
