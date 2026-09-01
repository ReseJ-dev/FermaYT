"""Tests for permanent versioned image style enforcement."""

import pytest

from app.errors import StyleContractError
from app.style_contracts import (
    DEFAULT_IMAGE_STYLE_ID,
    ROUGH_EXPLAINER_V1,
    apply_image_style_contract,
    get_image_style_contract,
)


def test_default_contract_has_stable_versioned_id_and_required_rules() -> None:
    rendered = ROUGH_EXPLAINER_V1.render()

    assert DEFAULT_IMAGE_STYLE_ID == "rough_explainer_v1"
    assert "thick slightly uneven black outlines" in rendered
    assert "dot eyes" in rendered
    assert "extremely simplified machinery and architecture" in rendered
    assert "photorealism" in rendered
    assert "STYLE HAS PRIORITY OVER DETAIL" in rendered


def test_contract_is_appended_after_scene_prompt() -> None:
    result = apply_image_style_contract("A wide mine cutaway")

    assert result.startswith("A wide mine cutaway\n\n")
    assert result.endswith(
        "This contract overrides any conflicting style instruction elsewhere in the request."
    )


def test_contract_injection_is_idempotent() -> None:
    once = apply_image_style_contract("A wide mine cutaway")

    assert apply_image_style_contract(once) == once
    assert once.count("STYLE CONTRACT [rough_explainer_v1]") == 1


@pytest.mark.parametrize(
    "prompt",
    [
        "A photorealistic mine",
        "Use cinematic lighting",
        "A polished vector art scene",
        "Highly detailed machinery",
        "Render with depth of field",
    ],
)
def test_scene_prompt_cannot_override_contract(prompt: str) -> None:
    with pytest.raises(StyleContractError, match="conflicts"):
        apply_image_style_contract(prompt)


def test_explicit_negative_constraint_is_not_treated_as_override() -> None:
    result = apply_image_style_contract("A mine scene, not photorealistic")

    assert "STYLE CONTRACT [rough_explainer_v1]" in result


def test_unknown_future_style_version_fails_explicitly() -> None:
    with pytest.raises(StyleContractError, match="Unknown image style contract"):
        get_image_style_contract("rough_explainer_v2")


def test_prompt_cannot_mix_style_versions() -> None:
    prompt = "Scene\n\nSTYLE CONTRACT [rough_explainer_v2]"

    with pytest.raises(StyleContractError, match="different"):
        apply_image_style_contract(prompt, "rough_explainer_v1")


def test_bare_matching_marker_cannot_bypass_contract() -> None:
    prompt = "Photorealistic mine\n\nSTYLE CONTRACT [rough_explainer_v1]"

    with pytest.raises(StyleContractError, match="different or duplicated"):
        apply_image_style_contract(prompt)


def test_later_conflict_is_detected_even_if_first_occurrence_is_negated() -> None:
    prompt = "Not photorealistic at first, then make it photorealistic"

    with pytest.raises(StyleContractError, match="conflicts"):
        apply_image_style_contract(prompt)
