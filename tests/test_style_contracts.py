"""Tests for permanent versioned image style enforcement."""

from types import MappingProxyType

import pytest

from app import style_contracts
from app.errors import StyleContractError
from app.style_contracts import (
    DEFAULT_IMAGE_STYLE_ID,
    ROUGH_EXPLAINER_V1,
    ImageStyleContract,
    StyleConflictRule,
    apply_image_style_contract,
    get_image_style_contract,
    prepare_image_prompt_for_provider,
    validate_image_style_prompt,
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
        "These permanent drawing rules override any conflicting style request."
    )


def test_contract_injection_is_idempotent() -> None:
    once = apply_image_style_contract("A wide mine cutaway")

    assert apply_image_style_contract(once) == once
    assert once.count("Use this permanent drawing style") == 1


def test_provider_boundary_strips_raw_planner_metadata_labels() -> None:
    prompt = prepare_image_prompt_for_provider(
        "Purpose: show miners. State: rising water. Change: blocked exit."
    )

    assert "Purpose:" not in prompt
    assert "State:" not in prompt
    assert "Change:" not in prompt
    assert "STYLE CONTRACT" not in prompt


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

    assert "Use this permanent drawing style" in result


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


@pytest.mark.parametrize(
    "prompt",
    [
        "photorealistic mine",
        "highly photorealistic scene",
        "realistic stone walls",
        "realistic human anatomy",
        "use realistic material textures",
        "cinematic lighting",
        "cinematic flashlight lighting",
        "cinematic realistic lighting",
        "complex shadows",
        "detailed wood grain",
        "sophisticated perspective",
        "polished editorial illustration",
        "polished vector art",
        "3d render",
        "depth of field",
        "strong depth of field",
        "complex gradients",
    ],
)
def test_real_style_requests_remain_conflicts(prompt: str) -> None:
    with pytest.raises(StyleContractError, match="conflicts"):
        validate_image_style_prompt(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "no photorealism",
        "not photorealistic",
        "non-photorealistic cartoon",
        "avoid realistic materials",
        "without realistic materials",
        "without cinematic lighting",
        "do not use cinematic lighting",
        "avoid depth of field",
        "simple flat colors, no complex gradients",
        "no realistic anatomy",
        "do not use wood grain",
        "don't use complex shadows",
        "don’t use complex gradients",
        "avoid sophisticated perspective",
        "no polished vector art",
        "without depth of field",
        "keep surfaces simple rather than realistic",
        "non-photorealistic cartoon mine",
        "non realistic materials",
        "simple flat lighting, not cinematic lighting",
        "excluding complex gradients",
    ],
)
def test_negated_style_concepts_are_allowed(prompt: str) -> None:
    assert validate_image_style_prompt(prompt) == prompt


@pytest.mark.parametrize(
    "prompt",
    [
        "no photorealism",
        "avoid realistic materials",
        "without realistic anatomy",
        "do not use cinematic lighting",
        "no polished editorial illustration",
        "avoid polished vector art",
        "no 3d render",
    ],
)
def test_required_negative_style_instructions_pass(prompt: str) -> None:
    assert validate_image_style_prompt(prompt) == prompt


def test_negative_section_heading_applies_to_each_bullet() -> None:
    prompt = """AVOID:
- photorealism
- realistic materials
- realistic anatomy
- cinematic lighting
- polished editorial illustration
- polished vector art
- 3D render
"""

    assert validate_image_style_prompt(prompt) == prompt.strip()


@pytest.mark.parametrize(
    "prompt",
    [
        (
            "Do not mix realistic tunnel walls, realistic lighting, or realistic "
            "materials with cartoon foreground elements."
        ),
        "remove realistic stone, concrete, wood, or metal rendering",
        "Remove fine surface detail and realistic stone texture.",
    ],
)
def test_negative_imperatives_cover_coordinated_lists(prompt: str) -> None:
    assert validate_image_style_prompt(prompt) == prompt


@pytest.mark.parametrize(
    "prompt, expected_concept",
    [
        ("remove flat colors, then use realistic materials", "realistic materials"),
        ("avoid complex shadows, but add realistic stone", "realistic stone"),
    ],
)
def test_positive_instruction_ends_coordinated_negation_scope(
    prompt: str,
    expected_concept: str,
) -> None:
    with pytest.raises(StyleContractError, match=expected_concept):
        validate_image_style_prompt(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "photorealistic mine shaft",
        "use realistic materials",
        "realistic human anatomy",
        "cinematic lighting",
        "polished editorial illustration",
        "polished vector art",
        "3d render",
    ],
)
def test_required_positive_conflicts_still_fail(prompt: str) -> None:
    with pytest.raises(StyleContractError, match="conflicts"):
        validate_image_style_prompt(prompt)


@pytest.mark.parametrize(
    "alias",
    [
        "photorealism",
        "photorealistic",
        "photo realistic",
        "photo-realistic",
        "3-d render",
    ],
)
def test_style_concept_aliases_are_blocked(alias: str) -> None:
    with pytest.raises(StyleContractError, match="conflicts"):
        validate_image_style_prompt(f"Create a {alias} mine")


def test_mixed_negated_and_positive_concepts_still_conflict() -> None:
    prompt = "no photorealism, but use cinematic lighting"

    with pytest.raises(StyleContractError, match="cinematic lighting"):
        validate_image_style_prompt(prompt)


def test_negation_does_not_leak_into_a_later_positive_instruction() -> None:
    prompt = "avoid flat colors and use photorealistic textures"

    with pytest.raises(StyleContractError, match="photorealism"):
        validate_image_style_prompt(prompt)


def test_unrelated_no_does_not_blanket_allow_a_later_conflict() -> None:
    prompt = "no flat colors and photorealistic textures"

    with pytest.raises(StyleContractError, match="photorealism"):
        validate_image_style_prompt(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "no flat illustration, make the scene photorealistic",
        "avoid cartoon styling and use cinematic realistic lighting",
    ],
)
def test_unrelated_style_negation_does_not_allow_conflict(prompt: str) -> None:
    with pytest.raises(StyleContractError, match="conflicts"):
        validate_image_style_prompt(prompt)


def test_gapped_alias_does_not_match_across_clause_boundary() -> None:
    prompt = "Use a cinematic composition. Lighting must remain simple and flat."

    assert validate_image_style_prompt(prompt) == prompt


@pytest.mark.parametrize(
    "prompt, expected_concept",
    [
        ("no photorealism, use depth of field", "depth of field"),
        (
            "avoid cinematic lighting, use realistic stone textures",
            "realistic stone",
        ),
    ],
)
def test_mixed_prompt_blocks_its_positive_conflict(
    prompt: str,
    expected_concept: str,
) -> None:
    with pytest.raises(StyleContractError, match=expected_concept):
        validate_image_style_prompt(prompt)


def test_mixed_prompt_with_only_negated_conflict_passes() -> None:
    prompt = "no realistic anatomy, use simplified cartoon anatomy"

    assert validate_image_style_prompt(prompt) == prompt


def test_not_only_is_not_treated_as_style_negation() -> None:
    prompt = "not only photorealistic but cinematic"

    with pytest.raises(StyleContractError, match="photorealism"):
        validate_image_style_prompt(prompt)


def test_validation_precedes_contract_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_prompts: list[str] = []
    real_validator = style_contracts.validate_image_style_prompt

    def observe_dynamic_prompt(prompt: str, style_id: str) -> str:
        observed_prompts.append(prompt)
        assert "STYLE CONTRACT [" not in prompt
        return real_validator(prompt, style_id)

    monkeypatch.setattr(
        style_contracts,
        "validate_image_style_prompt",
        observe_dynamic_prompt,
    )

    result = style_contracts.apply_image_style_contract("A flat cartoon mine")

    assert observed_prompts == ["A flat cartoon mine"]
    assert "Use this permanent drawing style" in result


def test_existing_permanent_contract_is_not_validated_as_dynamic_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contracted = apply_image_style_contract("A flat cartoon mine")
    observed_prompts: list[str] = []
    real_validator = style_contracts.validate_image_style_prompt

    def observe_dynamic_prompt(prompt: str, style_id: str) -> str:
        observed_prompts.append(prompt)
        return real_validator(prompt, style_id)

    monkeypatch.setattr(
        style_contracts,
        "validate_image_style_prompt",
        observe_dynamic_prompt,
    )

    assert style_contracts.apply_image_style_contract(contracted) == contracted
    assert observed_prompts == ["A flat cartoon mine"]


def test_provider_boundary_does_not_rescan_assembled_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assembled = apply_image_style_contract("A flat cartoon mine")

    def fail_if_called(prompt: str, style_id: str) -> str:
        raise AssertionError(f"assembled prompt was rescanned: {style_id}: {prompt}")

    monkeypatch.setattr(
        style_contracts,
        "validate_image_style_prompt",
        fail_if_called,
    )

    prepared = prepare_image_prompt_for_provider(assembled)
    assert prepared.endswith(assembled)
    assert prepared.startswith(
        "Create one illustration containing absolutely no visible text"
    )


def test_validator_uses_rules_from_selected_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_contract = ImageStyleContract(
        style_id="custom_flat_v1",
        description="Custom flat style.",
        required=("flat colors",),
        prohibited=("oil painting",),
        priority_rule="Flat style wins.",
        conflict_rules=(StyleConflictRule("oil painting", ("oil painting",)),),
    )
    monkeypatch.setattr(
        style_contracts,
        "IMAGE_STYLE_CONTRACTS",
        MappingProxyType({custom_contract.style_id: custom_contract}),
    )

    assert (
        validate_image_style_prompt("photorealistic reference", custom_contract.style_id)
        == "photorealistic reference"
    )
    with pytest.raises(StyleContractError, match="oil painting"):
        validate_image_style_prompt("use oil painting", custom_contract.style_id)
