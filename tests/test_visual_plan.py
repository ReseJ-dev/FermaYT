"""Schema-level tests for semantic visual plans."""

import pytest
from pydantic import ValidationError

from app.models.visual_plan import (
    CameraMovement,
    ShotFraming,
    VisualBeat,
    VisualOperation,
)


def test_visual_operation_serializes_as_requested_string() -> None:
    assert VisualOperation.REFERENCE_GENERATION.value == "REFERENCE_GENERATION"


def test_dependent_visual_operation_requires_a_source() -> None:
    with pytest.raises(ValidationError, match="requires source_visual_id"):
        VisualBeat(
            id="beat",
            narration_segment="Narration",
            visual_purpose="Purpose",
            what_viewer_should_understand="Meaning",
            location_id="location",
            camera_framing=ShotFraming.WIDE,
            camera_view="Wide",
            framing_reason="Establish geography",
            camera_movement=CameraMovement.STATIC,
            physical_state="Stable",
            change_from_previous_beat="Initial",
            information_added_beyond_narration="Shows distance",
            preferred_visual_operation=VisualOperation.REUSE,
            estimated_duration_seconds=4,
        )


def test_visual_beat_rejects_extra_prompt_field() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        VisualBeat(
            id="beat",
            narration_segment="Narration",
            visual_purpose="Purpose",
            what_viewer_should_understand="Meaning",
            location_id="location",
            camera_framing=ShotFraming.WIDE,
            camera_view="Wide",
            framing_reason="Establish geography",
            camera_movement=CameraMovement.STATIC,
            physical_state="Stable",
            change_from_previous_beat="Initial",
            information_added_beyond_narration="Shows distance",
            preferred_visual_operation=VisualOperation.NEW_IMAGE,
            estimated_duration_seconds=4,
            image_prompt="This layer must not create prompts yet",
        )
