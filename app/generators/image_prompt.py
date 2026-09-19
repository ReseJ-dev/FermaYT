"""Build concise provider-ready image prompts from semantic visual state."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from app.errors import ImagePromptBuildError
from app.generators.style_reference import build_reference_role_instruction
from app.models.visual_plan import (
    VisualBeat,
    VisualOperation,
    VisualPlan,
    assess_visual_beat_complexity,
)
from app.providers import ImageReference
from app.style_contracts import (
    DEFAULT_IMAGE_STYLE_ID,
    NO_VISIBLE_TEXT_INSTRUCTION,
    append_image_style_contract,
    validate_image_style_prompt,
)


@dataclass(frozen=True, slots=True)
class ImagePromptBuilder:
    """Translate semantic directing decisions without using narration text."""

    max_semantic_characters: int = 4_500

    def build(
        self,
        plan: VisualPlan,
        beat: VisualBeat,
        operation: VisualOperation,
        *,
        references: tuple[ImageReference, ...] = (),
        style_id: str = DEFAULT_IMAGE_STYLE_ID,
        project_style_prompt: str | None = None,
    ) -> str:
        # The operation controls which provider method the pipeline invokes. It is
        # never visual content and must not be exposed to an image model.
        del operation
        core = build_simplified_visual_core(plan, beat)
        sections: list[tuple[str, str]] = []
        reference_section = self._reference_section(references)
        if reference_section is not None:
            sections.append(reference_section)
        sections.extend(self._core_sections(core))
        if project_style_prompt is not None and project_style_prompt.strip():
            sections.insert(
                1,
                ("PROJECT STYLE DIRECTION", project_style_prompt.strip()),
            )
        excluded = [*beat.must_not_show, *beat.optional_entities_to_omit]
        if excluded:
            sections.append(("DO NOT SHOW", "; ".join(dict.fromkeys(excluded))))
        semantic_prompt = _render_sections(NO_VISIBLE_TEXT_INSTRUCTION, sections)
        validate_image_style_prompt(semantic_prompt, style_id)
        semantic_prompt = _fit_semantic_sections(
            NO_VISIBLE_TEXT_INSTRUCTION,
            sections,
            self.max_semantic_characters,
        )
        return append_image_style_contract(semantic_prompt, style_id)

    def build_edit(
        self,
        plan: VisualPlan,
        beat: VisualBeat,
        *,
        references: tuple[ImageReference, ...],
        style_id: str = DEFAULT_IMAGE_STYLE_ID,
        project_style_prompt: str | None = None,
    ) -> str:
        """Build a concise state delta while protecting existing composition."""
        if not references:
            raise ImagePromptBuildError("Image editing requires a source reference")
        core = build_simplified_visual_core(plan, beat)
        keep = [
            "the source composition and identities not named in the change",
        ]
        change = _clean_join((core.main_visual_idea, core.visible_physical_state))
        if beat.progressive_change is not None:
            progression = beat.progressive_change
            change = _clean_join(
                (
                    change,
                    (
                        f"{progression.subject_id}, {progression.previous_state} to "
                        f"{progression.current_state}; {progression.progression}"
                    ),
                )
            )
        sections = [
            (
                "Attached image guidance",
                build_reference_role_instruction(references),
            ),
            ("KEEP UNCHANGED", "; ".join(keep)),
            ("CHANGE ONLY", change),
            *self._core_sections(core, include_action=False),
        ]
        if project_style_prompt is not None and project_style_prompt.strip():
            sections.insert(
                1,
                ("PROJECT STYLE DIRECTION", project_style_prompt.strip()),
            )
        excluded = [*beat.must_not_show, *beat.optional_entities_to_omit]
        if excluded:
            sections.append(("DO NOT SHOW", "; ".join(dict.fromkeys(excluded))))
        prefix = (
            f"{NO_VISIBLE_TEXT_INSTRUCTION} Edit the provided source image; preserve "
            "everything not explicitly changed below."
        )
        prompt = _render_sections(prefix, sections)
        validate_image_style_prompt(prompt, style_id)
        prompt = _fit_semantic_sections(
            prefix,
            sections,
            self.max_semantic_characters,
        )
        return append_image_style_contract(prompt, style_id)

    @staticmethod
    def _reference_section(
        references: tuple[ImageReference, ...],
    ) -> tuple[str, str] | None:
        if not references:
            return None
        return (
            "Attached image guidance",
            build_reference_role_instruction(references),
        )

    @staticmethod
    def _core_sections(
        core: SimplifiedVisualCore, *, include_action: bool = True
    ) -> list[tuple[str, str]]:
        sections: list[tuple[str, str]] = []
        if core.main_subject:
            sections.append(("MAIN SUBJECT", core.main_subject))
        if include_action:
            sections.append(
                (
                    "MAIN ACTION / STATE",
                    _clean_join((core.main_visual_idea, core.visible_physical_state)),
                )
            )
        if core.essential_objects:
            sections.append(("ESSENTIAL OBJECTS", "; ".join(core.essential_objects)))
        sections.extend(
            [
                ("MINIMAL LOCATION", core.minimal_location),
                ("SIMPLE FRAMING", core.simple_framing),
            ]
        )
        return sections


@dataclass(frozen=True, slots=True)
class SimplifiedVisualCore:
    main_visual_idea: str
    main_subject: str | None
    visible_physical_state: str
    essential_objects: tuple[str, ...]
    minimal_location: str
    simple_framing: str
    character_count_target: int
    background_complexity: str
    omitted_entities: tuple[str, ...]
    visible_entities: tuple[str, ...]
    environment_cues_used: tuple[str, ...]
    complexity_budget: dict[str, int]
    complexity_status: str
    split_required: bool
    complexity_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_simplified_visual_core(
    plan: VisualPlan, beat: VisualBeat
) -> SimplifiedVisualCore:
    """Resolve only beat-visible facts; continuity references carry everything else."""
    location = next(
        (item for item in plan.locations if item.id == beat.location_id), None
    )
    if location is None:
        raise ImagePromptBuildError(f"Unknown beat location: {beat.location_id}")
    assessment = assess_visual_beat_complexity(beat)
    if assessment.budget_exceeded:
        reason = "; ".join(assessment.reasons)
        raise ImagePromptBuildError(f"COMPLEXITY_BUDGET_EXCEEDED: {reason}")
    characters = {item.id: item for item in plan.characters}
    objects = {item.id: item for item in plan.important_objects}
    omitted = tuple(dict.fromkeys((*beat.must_not_show, *beat.optional_entities_to_omit)))
    subject_parts: list[str] = []
    for character_id in beat.characters_visible:
        character = characters[character_id]
        descriptor = _minimal_descriptor(character.description, 12)
        subject_parts.append(
            f"{character.name}{f', {descriptor}' if descriptor else ''}"
        )
    main_subject = None
    if subject_parts:
        count_word = "one" if beat.character_count_target == 1 else str(
            beat.character_count_target
        )
        main_subject = f"{count_word} visible character: {'; '.join(subject_parts)}"
    elif beat.character_count_target == 0:
        main_subject = "no people visible"
    essential_objects = tuple(
        dict.fromkeys(
            objects[item].name
            for item in beat.important_objects
            if not _matches_omitted(item, objects[item].name, omitted)
        )
    )
    environment_cues = tuple(
        cue
        for cue in (
            _normalize_environment_cue(item, omitted)
            for item in beat.essential_environment_cues
        )
        if cue and not _semantically_repeated(cue, (location.name,))
    )
    environment_cues = _deduplicate_phrases(environment_cues)
    minimal_location = _clean_join((location.name, *environment_cues))
    simple_framing = _clean_join(
        _deduplicate_phrases((beat.camera_framing.value, beat.camera_view))
    )
    visible_entities = tuple(
        dict.fromkeys((*beat.characters_visible, *beat.important_objects))
    )
    return SimplifiedVisualCore(
        main_visual_idea=sanitize_provider_visual_text(beat.main_visual_idea),
        main_subject=main_subject,
        visible_physical_state=sanitize_provider_visual_text(
            beat.visible_physical_state
        ),
        essential_objects=essential_objects,
        minimal_location=minimal_location,
        simple_framing=simple_framing,
        character_count_target=beat.character_count_target,
        background_complexity=beat.background_complexity.value,
        omitted_entities=omitted,
        visible_entities=visible_entities,
        environment_cues_used=environment_cues,
        complexity_budget=beat.complexity_budget.model_dump(mode="json"),
        complexity_status="WITHIN_BUDGET",
        split_required=assessment.split_required,
        complexity_reasons=tuple(assessment.reasons),
    )


def _matches_omitted(identity: str, name: str, omitted: tuple[str, ...]) -> bool:
    candidates = {identity.casefold(), name.casefold()}
    for value in omitted:
        lowered = value.casefold()
        if any(candidate in lowered or lowered in candidate for candidate in candidates):
            return True
    return False


def _normalize_environment_cue(value: str, omitted: tuple[str, ...]) -> str:
    """Keep setting recognition while removing omitted inventory and camera prose."""
    cleaned = sanitize_provider_visual_text(value)
    cleaned = re.sub(
        r"\b(?:in|from)\s+(?:a\s+)?(?:wide|close|detail|medium|low-angle|high-angle)?\s*"
        r"(?:side|front|overhead)?\s*(?:view|perspective)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    parts = re.split(r"\s*(?:;|,|\bwith\b|\bfeaturing\b|\bcontaining\b)\s*", cleaned)
    normalized_parts = [
        re.sub(r"^(?:and|or)\s+", "", part.strip(" .,-"), flags=re.IGNORECASE)
        for part in parts
    ]
    kept = [
        part
        for part in normalized_parts
        if part
        and not any(
            omitted_value.casefold() in part.casefold()
            or part.casefold() in omitted_value.casefold()
            for omitted_value in omitted
        )
    ]
    return _clean_join(kept)


def _phrase_tokens(value: str) -> set[str]:
    stop = {"a", "an", "the", "from", "in", "of", "camera", "view"}
    return {
        item
        for item in re.findall(r"[a-z0-9]+", value.casefold())
        if item not in stop
    }


def _semantically_repeated(value: str, existing: tuple[str, ...]) -> bool:
    tokens = _phrase_tokens(value)
    return bool(tokens) and any(tokens <= _phrase_tokens(item) for item in existing)


def _deduplicate_phrases(values: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        cleaned = sanitize_provider_visual_text(value).strip(" .;")
        if not cleaned:
            continue
        tokens = _phrase_tokens(cleaned)
        duplicate_index = next(
            (
                index
                for index, current in enumerate(result)
                if tokens <= _phrase_tokens(current)
                or _phrase_tokens(current) <= tokens
            ),
            None,
        )
        if duplicate_index is None:
            result.append(cleaned)
        elif len(tokens) > len(_phrase_tokens(result[duplicate_index])):
            result[duplicate_index] = cleaned
    return tuple(result)


def _minimal_descriptor(value: str, maximum_words: int) -> str:
    cleaned = sanitize_provider_visual_text(value)
    clause = re.split(r"[.;]", cleaned, maxsplit=1)[0].strip(" ,")
    words = clause.split()
    return " ".join(words[:maximum_words])


def _clean_join(values: object) -> str:
    return "; ".join(
        sanitize_provider_visual_text(str(item)).strip(" .;")
        for item in values  # type: ignore[union-attr]
        if item is not None and str(item).strip()
    )


def _render_sections(
    prefix: str,
    sections: list[tuple[str, str]],
) -> str:
    entries = [
        (heading, sanitize_provider_visual_text(content))
        for heading, content in sections
        if content.strip()
    ]
    return prefix + "".join(
        f"\n\n{_PROVIDER_SECTION_LEADS[heading]} {content}"
        for heading, content in entries
    )


def _fit_semantic_sections(
    prefix: str,
    sections: list[tuple[str, str]],
    maximum: int,
) -> str:
    """Compact verbose model output while retaining every semantic section."""
    entries = [
        (heading, sanitize_provider_visual_text(content))
        for heading, content in sections
        if content.strip()
    ]
    rendered = _render_sections(prefix, entries)
    if len(rendered) <= maximum:
        return rendered

    fixed_length = len(prefix) + sum(
        len(f"\n\n{_PROVIDER_SECTION_LEADS[heading]} ") for heading, _ in entries
    )
    content_budget = maximum - fixed_length
    # Represent every section, then protect the simplified visible core before
    # optional continuity/style prose receives space. This is deterministic and
    # prevents equal-share compaction from erasing the requested physical state.
    base_per_section = 4
    if content_budget < base_per_section * len(entries):
        raise ImagePromptBuildError(
            "Semantic image prompt is too long for the configured prompt budget"
        )

    lengths = [len(content) for _, content in entries]
    allocations = [min(length, base_per_section) for length in lengths]
    remaining = content_budget - sum(allocations)
    protected_targets = {
        "MAIN SUBJECT": 60,
        "MAIN ACTION / STATE": 90,
        "ESSENTIAL OBJECTS": 45,
        "MINIMAL LOCATION": 45,
        "SIMPLE FRAMING": 35,
        "CHANGE ONLY": 80,
    }
    for heading, target in protected_targets.items():
        index = next(
            (i for i, (candidate, _) in enumerate(entries) if candidate == heading),
            None,
        )
        if index is None or remaining <= 0:
            continue
        wanted = min(lengths[index], target)
        addition = min(wanted - allocations[index], remaining)
        allocations[index] += max(0, addition)
        remaining -= max(0, addition)
    while remaining > 0:
        active = [
            index for index, length in enumerate(lengths) if allocations[index] < length
        ]
        if not active:
            break
        share = max(1, remaining // len(active))
        for index in active:
            addition = min(share, lengths[index] - allocations[index], remaining)
            allocations[index] += addition
            remaining -= addition
            if remaining == 0:
                break

    compacted = [
        (heading, _ellipsize_middle(content, allocation))
        for (heading, content), allocation in zip(entries, allocations, strict=True)
    ]
    result = _render_sections(prefix, compacted)
    if len(result) > maximum:
        raise ImagePromptBuildError(
            "Semantic image prompt is too long for the configured prompt budget"
        )
    return result


_PROVIDER_SECTION_LEADS = {
    "Attached image guidance": "Use the attached images only as visual guidance.",
    "MAIN SUBJECT": "Show only this main subject.",
    "MAIN ACTION / STATE": "Show this main action and visible state.",
    "ESSENTIAL OBJECTS": "Include these essential story objects.",
    "MINIMAL LOCATION": "Use only this minimal location context.",
    "SIMPLE FRAMING": "Frame the scene simply.",
    "PROJECT STYLE DIRECTION": "Follow this project drawing direction.",
    "DO NOT SHOW": "Exclude these story mistakes.",
    "KEEP UNCHANGED": "Preserve the following established elements.",
    "CHANGE ONLY": "Change only this physical state.",
}

_INTERNAL_LABEL = re.compile(
    r"(?i)\b(?:purpose|state|change|visual\s+operation|visual\s+reference|"
    r"reference\s+instructions?|image\s+reference|planner\s+notes?|qa\s+notes?|"
    r"style\s+contract)\s*:"
)


def sanitize_provider_visual_text(value: str) -> str:
    """Remove planner-style labels while preserving their concrete visual content."""
    normalized = " ".join(value.split())
    normalized = _INTERNAL_LABEL.sub("", normalized)
    normalized = normalized.replace(":", ",")
    return re.sub(r"\s+", " ", normalized).strip()


def _ellipsize_middle(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    head = (maximum - 3) * 2 // 3
    tail = maximum - head - 3
    return f"{value[:head].rstrip()}...{value[-tail:].lstrip()}"
