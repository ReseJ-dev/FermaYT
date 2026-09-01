"""Tests for continuity-aware visual operation decisions."""

import logging

import pytest

from app.models.visual_plan import VisualOperation, VisualPlan
from app.pipeline.visual_operation_engine import (
    VisualDecisionEvidence,
    VisualOperationDecisionEngine,
    VisualProviderCapabilities,
)


def _beat(
    beat_id: str,
    operation: VisualOperation,
    *,
    camera_view: str,
    progressive_change: dict[str, str] | None = None,
    overlay_description: str | None = None,
) -> dict[str, object]:
    source_id = None if operation is VisualOperation.NEW_IMAGE else "beat_1"
    return {
        "id": beat_id,
        "narration_segment": f"Narration for {beat_id}",
        "visual_purpose": "Keep the changing route understandable",
        "what_viewer_should_understand": "The same shaft remains visible",
        "location_id": "shaft",
        "characters_visible": ["miners"],
        "important_objects": ["ladder"],
        "camera_framing": "WIDE" if beat_id == "beat_1" else "MEDIUM",
        "camera_view": camera_view,
        "framing_reason": "Show the relevant geography",
        "camera_movement": "STATIC",
        "geography_established_by": None,
        "master_scene_id": "shaft_master",
        "physical_state": (
            "The ladder is broken" if progressive_change else "The ladder is intact"
        ),
        "progressive_change": progressive_change,
        "safety_geography": None,
        "anticipated_consequence": None,
        "change_from_previous_beat": (
            "The ladder changes state" if progressive_change else "No physical change"
        ),
        "information_added_beyond_narration": "Shows the route relative to the miners",
        "preferred_visual_operation": operation.value,
        "source_visual_id": source_id,
        "overlay_description": overlay_description,
        "estimated_duration_seconds": 4,
    }


def _plan(
    second_operation: VisualOperation = VisualOperation.REUSE,
    *,
    second_camera_view: str = "Wide shaft",
    progressive_change: dict[str, str] | None = None,
    overlay_description: str | None = None,
) -> VisualPlan:
    return VisualPlan.model_validate(
        {
            "story_summary": "A mine route becomes dangerous",
            "visual_strategy": "Maintain one recognizable shaft",
            "characters": [
                {
                    "id": "miners",
                    "name": "Miners",
                    "description": "Workers below ground",
                }
            ],
            "locations": [
                {
                    "id": "shaft",
                    "name": "Mine shaft",
                    "description": "A deep vertical shaft",
                    "spatial_layout": "Surface above and miners below",
                }
            ],
            "important_objects": [
                {
                    "id": "ladder",
                    "name": "Ladder",
                    "description": "Vertical escape ladder",
                    "story_function": "Route to safety",
                }
            ],
            "recurring_environments": [
                {
                    "id": "mine_environment",
                    "location_id": "shaft",
                    "continuity_requirements": "Keep the shaft layout stable",
                }
            ],
            "possible_master_scenes": [
                {
                    "id": "shaft_master",
                    "location_id": "shaft",
                    "description": "Wide master cutaway",
                    "environment_geometry": "Vertical shaft from surface to tunnel",
                    "recurring_object_positions": "Ladder follows the right wall",
                    "color_palette": "Dark rock with amber work lights",
                    "basic_composition": "Surface above, miners below",
                    "characters_visible": ["miners"],
                    "important_objects": ["ladder"],
                }
            ],
            "visual_beats": [
                _beat(
                    "beat_1",
                    VisualOperation.NEW_IMAGE,
                    camera_view="Wide shaft",
                ),
                _beat(
                    "beat_2",
                    second_operation,
                    camera_view=second_camera_view,
                    progressive_change=progressive_change,
                    overlay_description=overlay_description,
                ),
            ],
        }
    )


def _state_change() -> dict[str, str]:
    return {
        "subject_id": "ladder",
        "previous_state": "Intact",
        "current_state": "Broken",
        "progression": "Open route becomes blocked",
    }


def test_first_environment_establishment_uses_new_image() -> None:
    decision = VisualOperationDecisionEngine().decide(
        _plan(),
        0,
        capabilities=VisualProviderCapabilities(reference_generation=True),
        available_visuals={"shaft_master": "/media/master.png"},
    )

    assert decision.operation is VisualOperation.NEW_IMAGE
    assert decision.requires_image_api is True


def test_unchanged_scene_is_reused_without_image_api_call() -> None:
    decision = VisualOperationDecisionEngine().decide(
        _plan(),
        1,
        capabilities=VisualProviderCapabilities(),
        available_visuals={"beat_1": "/media/beat-1.png"},
    )

    assert decision.operation is VisualOperation.REUSE
    assert decision.requires_image_api is False
    assert decision.source_image_paths == ("/media/beat-1.png",)


def test_camera_change_without_state_change_uses_transform() -> None:
    decision = VisualOperationDecisionEngine().decide(
        _plan(VisualOperation.TRANSFORM, second_camera_view="Closer shaft framing"),
        1,
        capabilities=VisualProviderCapabilities(),
        available_visuals={"beat_1": "/media/beat-1.png"},
    )

    assert decision.operation is VisualOperation.TRANSFORM
    assert decision.requires_image_api is False


def test_same_scene_with_physical_change_uses_supported_edit() -> None:
    decision = VisualOperationDecisionEngine().decide(
        _plan(VisualOperation.EDIT_EXISTING, progressive_change=_state_change()),
        1,
        capabilities=VisualProviderCapabilities(
            reference_generation=True,
            image_editing=True,
        ),
        available_visuals={"beat_1": "/media/beat-1.png"},
    )

    assert decision.operation is VisualOperation.EDIT_EXISTING
    assert "physical state changed" in decision.reasons


def test_unsupported_edit_falls_back_to_reference_generation() -> None:
    decision = VisualOperationDecisionEngine().decide(
        _plan(VisualOperation.EDIT_EXISTING, progressive_change=_state_change()),
        1,
        capabilities=VisualProviderCapabilities(reference_generation=True),
        available_visuals={"beat_1": "/media/beat-1.png"},
    )

    assert decision.operation is VisualOperation.REFERENCE_GENERATION
    assert decision.fallback_from is VisualOperation.EDIT_EXISTING
    assert decision.source_image_paths == ("/media/beat-1.png",)


def test_unsupported_edit_without_reference_support_falls_back_to_new_image() -> None:
    decision = VisualOperationDecisionEngine().decide(
        _plan(VisualOperation.EDIT_EXISTING, progressive_change=_state_change()),
        1,
        capabilities=VisualProviderCapabilities(),
        available_visuals={"beat_1": "/media/beat-1.png"},
    )

    assert decision.operation is VisualOperation.NEW_IMAGE
    assert decision.fallback_from is VisualOperation.EDIT_EXISTING
    assert decision.source_image_paths == ()


def test_programmatic_overlay_does_not_call_image_api() -> None:
    decision = VisualOperationDecisionEngine().decide(
        _plan(
            VisualOperation.OVERLAY,
            overlay_description="Draw a route arrow ending at the broken ladder",
        ),
        1,
        capabilities=VisualProviderCapabilities(),
        available_visuals={"beat_1": "/media/beat-1.png"},
    )

    assert decision.operation is VisualOperation.OVERLAY
    assert decision.requires_image_api is False


def test_new_composition_can_use_reference_identity() -> None:
    decision = VisualOperationDecisionEngine().decide(
        _plan(
            VisualOperation.REFERENCE_GENERATION,
            second_camera_view="Opposite side composition",
        ),
        1,
        capabilities=VisualProviderCapabilities(reference_generation=True),
        available_visuals={"beat_1": "/media/beat-1.png"},
        evidence=VisualDecisionEvidence(
            substantially_different_composition=True,
            new_image_improves_understanding=True,
        ),
    )

    assert decision.operation is VisualOperation.REFERENCE_GENERATION
    assert decision.source_visual_ids == ("beat_1",)


def test_invalid_similarity_evidence_is_rejected() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        VisualDecisionEvidence(visual_similarity=1.2)


def test_decision_reason_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        decision = VisualOperationDecisionEngine().decide(
            _plan(VisualOperation.EDIT_EXISTING, progressive_change=_state_change()),
            1,
            capabilities=VisualProviderCapabilities(image_editing=True),
            available_visuals={"beat_1": "/media/beat-1.png"},
        )

    assert decision.operation is VisualOperation.EDIT_EXISTING
    assert "Beat beat_2:" in caplog.text
    assert "EDIT_EXISTING" in caplog.text
    assert "physical state changed" in caplog.text
