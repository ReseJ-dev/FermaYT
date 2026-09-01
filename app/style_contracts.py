"""Versioned, non-negotiable style contracts for stateless image requests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from app.errors import StyleContractError


@dataclass(frozen=True, slots=True)
class ImageStyleContract:
    style_id: str
    description: str
    required: tuple[str, ...]
    prohibited: tuple[str, ...]
    priority_rule: str

    def render(self) -> str:
        required = "\n".join(f"- {item}" for item in self.required)
        prohibited = "\n".join(f"- {item}" for item in self.prohibited)
        return f"""STYLE CONTRACT [{self.style_id}] — NON-NEGOTIABLE
{self.description}

REQUIRED:
{required}

DO NOT ADD:
{prohibited}

Historical accuracy applies to important layout and recognizable objects, not to
detailed rendering. Objects must use the minimum number of shapes needed for quick
understanding. {self.priority_rule}

This contract overrides any conflicting style instruction elsewhere in the request."""


ROUGH_EXPLAINER_V1: Final = ImageStyleContract(
    style_id="rough_explainer_v1",
    description=(
        "Simple amateur hand-drawn 2D YouTube explainer illustration. Slightly "
        "imperfect and handmade, but not children's artwork."
    ),
    required=(
        "thick slightly uneven black outlines",
        "simple crude geometry",
        "simple cartoon people",
        "minimal facial features and dot eyes",
        "flat muted colors",
        "almost no shading",
        "lots of visual simplicity and sparse backgrounds",
        "slightly imperfect perspective",
        "slightly awkward handmade proportions",
        "extremely simplified machinery and architecture",
    ),
    prohibited=(
        "photorealism or realistic materials",
        "realistic anatomy",
        "cinematic lighting or complex shadows",
        "realistic stone, wood grain, or scratches",
        "tiny mechanical details",
        "sophisticated perspective",
        "polished editorial illustration or polished vector art",
        "3D or depth of field",
        "excessive architecture details",
        "complex gradients",
    ),
    priority_rule=(
        "When DETAIL conflicts with VISUAL SIMPLICITY, choose simplicity while "
        "preserving factual information. STYLE HAS PRIORITY OVER DETAIL."
    ),
)

DEFAULT_IMAGE_STYLE_ID: Final = ROUGH_EXPLAINER_V1.style_id
IMAGE_STYLE_CONTRACTS: Final = MappingProxyType(
    {ROUGH_EXPLAINER_V1.style_id: ROUGH_EXPLAINER_V1}
)

_STYLE_MARKER = re.compile(r"STYLE CONTRACT \[([^\]]+)]")
_CONFLICTING_STYLE_PHRASES: Final = (
    "photorealistic",
    "photorealism",
    "realistic materials",
    "realistic anatomy",
    "cinematic lighting",
    "complex shadows",
    "realistic stone",
    "wood grain",
    "tiny mechanical details",
    "sophisticated perspective",
    "polished editorial illustration",
    "polished vector art",
    "3d render",
    "depth of field",
    "excessive architecture details",
    "complex gradients",
    "highly detailed",
    "intricate details",
)
_NEGATIONS: Final = ("not ", "no ", "avoid ", "without ", "do not ")


def get_image_style_contract(style_id: str) -> ImageStyleContract:
    normalized_id = style_id.strip()
    try:
        return IMAGE_STYLE_CONTRACTS[normalized_id]
    except KeyError as exc:
        raise StyleContractError(f"Unknown image style contract: {normalized_id}") from exc


def apply_image_style_contract(
    prompt: str,
    style_id: str = DEFAULT_IMAGE_STYLE_ID,
) -> str:
    """Append one authoritative contract or reject a conflicting override."""
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise ValueError("image prompt must not be empty")

    contract = get_image_style_contract(style_id)
    rendered_contract = contract.render()
    existing_markers = _STYLE_MARKER.findall(normalized_prompt)
    if existing_markers:
        if (
            existing_markers == [contract.style_id]
            and normalized_prompt.endswith(rendered_contract)
            and normalized_prompt.count(rendered_contract) == 1
        ):
            return normalized_prompt
        raise StyleContractError(
            "Image prompt already contains a different or duplicated style contract"
        )

    conflicts = _find_conflicting_style_phrases(normalized_prompt)
    if conflicts:
        phrases = ", ".join(conflicts)
        raise StyleContractError(
            f"Image prompt conflicts with {contract.style_id}: {phrases}"
        )
    return f"{normalized_prompt}\n\n{rendered_contract}"


def _find_conflicting_style_phrases(prompt: str) -> list[str]:
    lowered = prompt.lower()
    conflicts: list[str] = []
    for phrase in _CONFLICTING_STYLE_PHRASES:
        for match in re.finditer(re.escape(phrase), lowered):
            prefix = lowered[max(0, match.start() - 16) : match.start()]
            if not any(prefix.endswith(negation) for negation in _NEGATIONS):
                conflicts.append(phrase)
                break
    return conflicts
