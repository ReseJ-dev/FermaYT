"""Versioned, non-negotiable style contracts for stateless image requests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from app.errors import StyleContractError

NO_VISIBLE_TEXT_INSTRUCTION: Final = (
    "Create one illustration containing absolutely no visible text of any kind: no "
    "letters, words, numbers, captions, signs, labels, titles, watermarks, prompt "
    "wording, technical metadata, or UI. All instructions are metadata only."
)


@dataclass(frozen=True, slots=True)
class StyleConflictRule:
    """One prohibited visual concept and the phrases that can request it."""

    concept: str
    aliases: tuple[str, ...]
    max_intervening_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ImageStyleContract:
    style_id: str
    description: str
    required: tuple[str, ...]
    prohibited: tuple[str, ...]
    priority_rule: str
    conflict_rules: tuple[StyleConflictRule, ...] = ()

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


_ROUGH_EXPLAINER_CONFLICT_RULES: Final = (
    StyleConflictRule(
        "photorealism",
        ("photorealism", "photorealistic", "photo realistic", "photo-realistic"),
    ),
    StyleConflictRule(
        "realistic materials",
        ("realistic materials", "realistic material"),
    ),
    StyleConflictRule(
        "realistic anatomy",
        ("realistic anatomy", "realistic human anatomy"),
    ),
    StyleConflictRule(
        "cinematic lighting",
        ("cinematic lighting",),
        max_intervening_tokens=1,
    ),
    StyleConflictRule("complex shadows", ("complex shadows",)),
    StyleConflictRule("realistic stone", ("realistic stone",)),
    StyleConflictRule("wood grain", ("wood grain",)),
    StyleConflictRule("tiny mechanical details", ("tiny mechanical details",)),
    StyleConflictRule("sophisticated perspective", ("sophisticated perspective",)),
    StyleConflictRule(
        "polished editorial illustration",
        ("polished editorial illustration",),
    ),
    StyleConflictRule("polished vector art", ("polished vector art",)),
    StyleConflictRule("3d render", ("3d render", "3-d render")),
    StyleConflictRule("depth of field", ("depth of field",)),
    StyleConflictRule(
        "excessive architecture details",
        ("excessive architecture details",),
    ),
    StyleConflictRule("complex gradients", ("complex gradients",)),
    StyleConflictRule("highly detailed", ("highly detailed",)),
    StyleConflictRule("intricate details", ("intricate details",)),
)


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
    conflict_rules=_ROUGH_EXPLAINER_CONFLICT_RULES,
)

DEFAULT_IMAGE_STYLE_ID: Final = ROUGH_EXPLAINER_V1.style_id
IMAGE_STYLE_CONTRACTS: Final = MappingProxyType(
    {ROUGH_EXPLAINER_V1.style_id: ROUGH_EXPLAINER_V1}
)

_STYLE_MARKER = re.compile(r"STYLE CONTRACT \[([^\]]+)]")
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_HARD_CLAUSE_BOUNDARY = re.compile(r"[\n.;:!?]+")
_SOFT_CLAUSE_BOUNDARY = re.compile(r"[,]+")
_CLAUSE_BOUNDARY_TOKEN: Final = "clauseboundarytoken"
_HARD_BOUNDARY_TOKEN: Final = "hardboundarytoken"
_NEGATION_TOKENS: Final = frozenset(
    {
        "no",
        "not",
        "don't",
        "avoid",
        "avoiding",
        "without",
        "never",
        "exclude",
        "excluding",
        "remove",
        "removing",
    }
)
_NEGATION_SCOPE_STARTERS: Final = frozenset(
    {"avoid", "avoiding", "exclude", "excluding", "remove", "removing", "without"}
)
_POSITIVE_INSTRUCTION_RESETS: Final = frozenset(
    {
        "add",
        "apply",
        "but",
        "create",
        "depict",
        "however",
        "include",
        "instead",
        "make",
        "render",
        "request",
        "show",
        "then",
        "use",
        "using",
    }
)
_NEGATION_SCOPE_TOKENS: Final = 5
_NEGATION_FILLER_TOKENS: Final = frozenset(
    {
        "a",
        "add",
        "all",
        "an",
        "any",
        "at",
        "create",
        "ever",
        "explicitly",
        "include",
        "make",
        "of",
        "please",
        "remotely",
        "render",
        "request",
        "the",
        "use",
        "using",
    }
)
_NEGATIVE_SECTION_HEADER = re.compile(
    r"\b(?:avoid|do not add|do not use|exclude|excluding)\s*:\s*$",
    re.IGNORECASE,
)
_NEGATIVE_INLINE_HEADER = re.compile(
    r"\b(?:avoid|do not add|do not use|exclude|excluding)\s*:\s*(.+)$",
    re.IGNORECASE,
)
_SECTION_HEADER = re.compile(r"^[A-Z][A-Z0-9 _/-]*:\s*$")
_BULLET_LINE = re.compile(r"^\s*[-*•]\s+(.+)$")


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
    """Validate dynamic content, then append one authoritative style contract."""
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
            dynamic_prompt = normalized_prompt[: -len(rendered_contract)].rstrip()
            validate_image_style_prompt(dynamic_prompt, style_id)
            return normalized_prompt
        raise StyleContractError(
            "Image prompt already contains a different or duplicated style contract"
        )

    validate_image_style_prompt(normalized_prompt, style_id)
    return append_image_style_contract(normalized_prompt, style_id)


def append_image_style_contract(
    validated_prompt: str,
    style_id: str = DEFAULT_IMAGE_STYLE_ID,
) -> str:
    """Append a contract to dynamic content that was already fully validated."""
    normalized_prompt = validated_prompt.strip()
    if not normalized_prompt:
        raise ValueError("image prompt must not be empty")
    if _STYLE_MARKER.search(normalized_prompt):
        raise StyleContractError(
            "Validated image prompt must not already contain a style contract"
        )
    contract = get_image_style_contract(style_id)
    return f"{normalized_prompt}\n\n{contract.render()}"


def prepare_image_prompt_for_provider(
    prompt: str,
    style_id: str = DEFAULT_IMAGE_STYLE_ID,
) -> str:
    """Finalize a prompt at the provider boundary without revalidating a contract."""
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise ValueError("image prompt must not be empty")
    contract = get_image_style_contract(style_id)
    rendered_contract = contract.render()
    existing_markers = _STYLE_MARKER.findall(normalized_prompt)
    if not existing_markers:
        assembled = apply_image_style_contract(normalized_prompt, style_id)
        return _with_no_visible_text_instruction(assembled)
    if (
        existing_markers == [contract.style_id]
        and normalized_prompt.endswith(rendered_contract)
        and normalized_prompt.count(rendered_contract) == 1
    ):
        return _with_no_visible_text_instruction(normalized_prompt)
    raise StyleContractError(
        "Image prompt contains a different, duplicated, or misplaced style contract"
    )


def _with_no_visible_text_instruction(prompt: str) -> str:
    if prompt.startswith(NO_VISIBLE_TEXT_INSTRUCTION):
        return prompt
    return f"{NO_VISIBLE_TEXT_INSTRUCTION}\n\n{prompt}"


def validate_image_style_prompt(
    prompt: str,
    style_id: str = DEFAULT_IMAGE_STYLE_ID,
) -> str:
    """Validate only dynamic/user/generated prompt content against a style."""
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise ValueError("image prompt must not be empty")
    contract = get_image_style_contract(style_id)
    conflicts = _find_conflicting_style_concepts(
        normalized_prompt,
        contract.conflict_rules,
    )
    if conflicts:
        phrases = ", ".join(conflicts)
        raise StyleContractError(
            f"Image prompt conflicts with {contract.style_id}: {phrases}"
        )
    return normalized_prompt


def _find_conflicting_style_concepts(
    prompt: str,
    rules: tuple[StyleConflictRule, ...],
) -> list[str]:
    tokens = _tokenize(prompt)
    conflicts: list[str] = []
    for rule in rules:
        if any(
            _contains_non_negated_alias(
                tokens,
                _normalized_alias(alias),
                rule.max_intervening_tokens,
            )
            for alias in rule.aliases
        ):
            conflicts.append(rule.concept)
    return conflicts


def _normalized_alias(alias: str) -> tuple[str, ...]:
    return _tokenize(alias)


def _tokenize(value: str) -> tuple[str, ...]:
    normalized = _expand_negative_sections(value)
    normalized = normalized.lower().replace("’", "'").replace("‘", "'")
    normalized = _HARD_CLAUSE_BOUNDARY.sub(
        f" {_HARD_BOUNDARY_TOKEN} ", normalized
    )
    normalized = _SOFT_CLAUSE_BOUNDARY.sub(
        f" {_CLAUSE_BOUNDARY_TOKEN} ", normalized
    )
    return tuple(match.group() for match in _TOKEN.finditer(normalized))


def _expand_negative_sections(value: str) -> str:
    """Carry an AVOID/DO NOT heading's negation into each bullet item."""
    lines: list[str] = []
    negative_section = False
    for line in value.splitlines():
        stripped = line.strip()
        inline_match = _NEGATIVE_INLINE_HEADER.search(stripped)
        if inline_match is not None:
            lines.append(f"avoid {inline_match.group(1)}")
            negative_section = False
            continue
        if _NEGATIVE_SECTION_HEADER.search(stripped):
            lines.append(line)
            negative_section = True
            continue
        bullet_match = _BULLET_LINE.fullmatch(line)
        if negative_section and bullet_match is not None:
            lines.append(f"avoid {bullet_match.group(1)}")
            continue
        if stripped and (
            _SECTION_HEADER.fullmatch(stripped) or bullet_match is None
        ):
            negative_section = False
        lines.append(line)
    return "\n".join(lines)


def _contains_non_negated_alias(
    tokens: tuple[str, ...],
    alias: tuple[str, ...],
    max_intervening_tokens: int,
) -> bool:
    for start, token in enumerate(tokens):
        if token != alias[0]:
            continue
        end = _match_alias_end(tokens, start, alias, max_intervening_tokens)
        if end is not None and not _is_locally_negated(tokens, start):
            return True
    return False


def _match_alias_end(
    tokens: tuple[str, ...],
    start: int,
    alias: tuple[str, ...],
    max_intervening_tokens: int,
) -> int | None:
    position = start
    remaining_gap = max_intervening_tokens
    for expected in alias[1:]:
        position += 1
        while position < len(tokens) and tokens[position] != expected:
            if tokens[position] in {
                _CLAUSE_BOUNDARY_TOKEN,
                _HARD_BOUNDARY_TOKEN,
            } or remaining_gap == 0:
                return None
            remaining_gap -= 1
            position += 1
        if position >= len(tokens):
            return None
    return position


def _is_locally_negated(tokens: tuple[str, ...], concept_start: int) -> bool:
    window_start = max(0, concept_start - _NEGATION_SCOPE_TOKENS)
    prefix = tokens[window_start:concept_start]
    negation_positions = [
        index for index, token in enumerate(prefix) if token in _NEGATION_TOKENS
    ]
    if not negation_positions:
        # Hyphen normalization turns "non-photorealistic" into two tokens.
        if prefix and prefix[-1] == "non":
            return True
        return _is_governed_by_negative_instruction(tokens, concept_start)

    negation_index = negation_positions[-1]
    negation = prefix[negation_index]
    following = prefix[negation_index + 1 :]
    if negation == "not" and following[:1] == ("only",):
        return False
    if all(token in _NEGATION_FILLER_TOKENS for token in following):
        return True
    return _is_governed_by_negative_instruction(tokens, concept_start)


def _is_governed_by_negative_instruction(
    tokens: tuple[str, ...],
    concept_start: int,
) -> bool:
    """Recognize negative imperatives that govern a coordinated list."""
    sentence_start = 0
    for index in range(concept_start - 1, -1, -1):
        if tokens[index] == _HARD_BOUNDARY_TOKEN:
            sentence_start = index + 1
            break
    prefix = tokens[sentence_start:concept_start]
    scope_start: int | None = None
    for index, token in enumerate(prefix):
        if token in _NEGATION_SCOPE_STARTERS:
            scope_start = index
        elif token in {"not", "never", "don't"}:
            if token == "not" and prefix[index + 1 : index + 2] == ("only",):
                continue
            # The following verb ("use", "mix", "add", etc.) belongs to the
            # negative command and must not reset its own scope.
            scope_start = min(index + 1, len(prefix) - 1)
    if scope_start is None:
        return False
    return not any(
        token in _POSITIVE_INSTRUCTION_RESETS
        for token in prefix[scope_start + 1 :]
    )
