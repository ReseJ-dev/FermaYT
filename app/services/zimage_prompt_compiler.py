"""Deterministic semantic prompt compilation for Z-Image's small prompt window."""

from __future__ import annotations

from dataclasses import dataclass

from app.clients.image_api import KieZImageApiClient
from app.errors import ImagePromptBuildError
from app.generators.image_prompt import sanitize_provider_visual_text
from app.models.visual_plan import MasterScene, VisualBeat, VisualOperation, VisualPlan
from app.style_contracts import DEFAULT_IMAGE_STYLE_ID, get_image_style_contract

_COMPACT_STYLE = (
    "rough handmade 2D, thick uneven outlines, crude shapes, flat muted colors, "
    "sparse detail, minimal shading, imperfect perspective; no text, realism, 3D, "
    "gradients, or borders"
)


@dataclass(frozen=True, slots=True)
class ZImagePromptCompilation:
    prompt: str
    report: dict[str, object]


@dataclass(frozen=True, slots=True)
class _Section:
    field: str
    label: str
    value: str
    minimum: int
    preferred: int
    priority: str = "IMPORTANT"


def compile_zimage_semantic_prompt(
    plan: VisualPlan,
    *,
    target: VisualBeat | MasterScene,
    operation: VisualOperation,
    manual_scene_override: str | None,
    qa_correction: str | None,
    style_id: str = DEFAULT_IMAGE_STYLE_ID,
) -> ZImagePromptCompilation:
    """Compile provider text from semantic fields instead of truncating prose.

    The order below is deliberate. Earlier fields receive spare characters first,
    while every non-optional field receives enough space to remain identifiable.
    QA is a delta appended to the original specification, never its replacement.
    """
    get_image_style_contract(style_id)  # Reject unknown contracts before dispatch.
    sections, dropped_optional, required_anchors = (
        _beat_sections(plan, target, operation, manual_scene_override, qa_correction)
        if isinstance(target, VisualBeat)
        else _master_sections(plan, target, manual_scene_override, qa_correction)
    )
    # QA remains a targeted delta to the original specification. Style wording is
    # last so it can never displace subject, action, objects, location, or framing.
    if qa_correction is not None:
        correction = sanitize_provider_visual_text(qa_correction)
        if correction:
            sections.append(
                _Section(
                    "qa_correction",
                    "Correction",
                    correction,
                    len(correction),
                    len(correction),
                    "CORRECTION",
                )
            )
    sections.append(
        _Section(
            "style", "Style", _COMPACT_STYLE, len(_COMPACT_STYLE), len(_COMPACT_STYLE)
        )
    )

    prefix = "Purely pictorial scene."
    fixed = len(prefix) + sum(len(f" {item.label}: .") for item in sections)
    available = KieZImageApiClient.PROMPT_MAX_CHARACTERS - fixed
    required_length = sum(
        len(item.value)
        for item in sections
        if item.priority in {"REQUIRED", "CORRECTION"}
    )
    if required_length > available:
        raise _required_state_truncated(
            "required semantic fields cannot fit the provider boundary"
        )
    minimums = [
        len(item.value)
        if item.priority in {"REQUIRED", "CORRECTION"}
        else 0
        if item.priority == "OPTIONAL"
        else min(len(item.value), item.minimum)
        for item in sections
    ]
    if sum(minimums) > available:
        # Required content is intact; compress IMPORTANT before losing semantics.
        minimums = [
            len(item.value)
            if item.priority in {"REQUIRED", "CORRECTION"}
            else 0
            for item in sections
        ]

    allocations = minimums[:]
    remaining = available - sum(allocations)
    # Priority order is the section order. Fill each semantic section to its
    # preferred budget before lower-priority wording receives any extra space.
    for priority in ("IMPORTANT", "OPTIONAL"):
        for index, section in enumerate(sections):
            if section.priority != priority:
                continue
            wanted = min(len(section.value), section.preferred)
            addition = min(max(0, wanted - allocations[index]), remaining)
            allocations[index] += addition
            remaining -= addition

    rendered_values = [
        _truncate_at_word(section.value, allocation)
        for section, allocation in zip(sections, allocations, strict=True)
    ]
    prompt = prefix + "".join(
        f" {section.label}: {value}."
        for section, value in zip(sections, rendered_values, strict=True)
        if value
    )
    if len(prompt) > KieZImageApiClient.PROMPT_MAX_CHARACTERS:
        raise _too_complex("compiled prompt exceeds provider boundary")
    _validate_required_sections(sections, rendered_values, prompt, required_anchors)

    compressed = [
        section.field
        for section, value in zip(sections, rendered_values, strict=True)
        if value != section.value
    ]
    report: dict[str, object] = {
        "preserved_fields": [
            item.field
            for item, value in zip(sections, rendered_values, strict=True)
            if value == item.value
        ],
        "compressed_fields": compressed,
        "truncated_fields": compressed,
        "dropped_optional_fields": dropped_optional,
        "required_fields": [
            item.field for item in sections if item.priority == "REQUIRED"
        ],
        "required_state_status": "PRESERVED",
        "final_length": len(prompt),
        "maximum_characters": KieZImageApiClient.PROMPT_MAX_CHARACTERS,
    }
    return ZImagePromptCompilation(prompt=prompt, report=report)


def _beat_sections(
    plan: VisualPlan,
    beat: VisualBeat,
    operation: VisualOperation,
    manual: str | None,
    qa_correction: str | None,
) -> tuple[list[_Section], list[str], list[str]]:
    by_character = {item.id: item for item in plan.characters}
    by_object = {item.id: item for item in plan.important_objects}
    location = next(
        (item for item in plan.locations if item.id == beat.location_id), None
    )
    if location is None:
        raise _too_complex(f"unknown location {beat.location_id}")

    action_parts: list[str] = [
        f"main visual fact {beat.main_visual_idea}",
        f"required visible state {beat.visible_physical_state}",
    ]
    action = _clean_join(action_parts)
    character_names = [by_character[item].name for item in beat.characters_visible]
    character_details = [
        f"{by_character[item].name}, {_compact_descriptor(by_character[item].description, 12)}"
        for item in beat.characters_visible
    ]
    subject_identity = _identity_first(character_names, character_details)
    if subject_identity:
        count = "one" if beat.character_count_target == 1 else str(
            beat.character_count_target
        )
        subject = f"{count} visible character; {subject_identity}"
    else:
        subject = "no people visible"
    object_names = [by_object[item].name for item in beat.important_objects]
    object_details = [
        f"{by_object[item].name}, {_compact_descriptor(by_object[item].description, 8)}"
        for item in beat.important_objects
    ]
    objects = (
        _identity_first(object_names, object_details) or "no required story object"
    )
    location_value = _identity_with_details(
        location.name,
        *(
            _compact_descriptor(cue, 12)
            for cue in beat.essential_environment_cues
        ),
    )
    camera = _clean_join((beat.camera_framing.value, beat.camera_view))
    change = beat.change_from_previous_beat
    if beat.progressive_change is not None:
        progress = beat.progressive_change
        change = _clean_join(
            (
                change,
                f"{progress.subject_id} {progress.previous_state} to {progress.current_state}",
            )
        )
    if operation is VisualOperation.EDIT_EXISTING:
        change = f"edit only this change; preserve all other continuity; {change}"

    sections = [
        _Section("characters", "Subject", subject, len(subject), len(subject), "REQUIRED"),
        _Section("physical_state_action", "Action", action, len(action), len(action), "REQUIRED"),
        _Section("critical_objects", "Objects", objects, len(objects), len(objects), "REQUIRED"),
        _Section("location", "Location", location_value, len(location_value), len(location_value), "REQUIRED"),
        _Section("camera_composition", "Camera", camera, 42, 90),
        _Section(
            "change_from_previous",
            "Change",
            change,
            min(len(change), 70),
            82,
        ),
    ]
    if manual:
        sections.append(
            _Section(
                "manual_scene_override",
                "Scene",
                sanitize_provider_visual_text(manual),
                0,
                90,
                "OPTIONAL",
            )
        )
    excluded = [*beat.must_not_show, *beat.optional_entities_to_omit]
    if excluded:
        sections.append(
            _Section("must_not_show", "Avoid", _clean_join(excluded), 28, 55)
        )
    dropped = [
        "visual_purpose",
        "framing_reason",
        "information_added_beyond_narration",
        "project_style_prose",
        "reference_role_prose",
    ]
    if beat.anticipated_consequence is not None:
        dropped.append("anticipated_consequence")
    if qa_correction is None:
        dropped.append("qa_correction")
    anchors = character_names + [location.name] + object_names
    return sections, dropped, anchors


def _master_sections(
    plan: VisualPlan,
    master: MasterScene,
    manual: str | None,
    qa_correction: str | None,
) -> tuple[list[_Section], list[str], list[str]]:
    by_character = {item.id: item for item in plan.characters}
    by_object = {item.id: item for item in plan.important_objects}
    location = next(
        (item for item in plan.locations if item.id == master.location_id), None
    )
    if location is None:
        raise _too_complex(f"unknown location {master.location_id}")
    character_names = [by_character[item].name for item in master.characters_visible]
    character_details = [
        f"{by_character[item].name}, {by_character[item].description}"
        for item in master.characters_visible
    ]
    object_names = [by_object[item].name for item in master.important_objects]
    object_details = [
        f"{by_object[item].name}, {by_object[item].description}"
        for item in master.important_objects
    ]
    master_action = f"required master {_compact_descriptor(master.description, 22)}"
    master_subject = _identity_first(
        character_names,
        [
            f"{name}, {_compact_descriptor(detail, 10)}"
            for name, detail in zip(character_names, character_details, strict=True)
        ],
    ) or "no people visible"
    master_location = _identity_with_details(
        location.name,
        _compact_descriptor(location.description, 12),
        _compact_descriptor(master.environment_geometry, 12),
    )
    master_objects = (
        _identity_first(
            object_names,
            [
                f"{name}, {_compact_descriptor(detail, 8)}"
                for name, detail in zip(object_names, object_details, strict=True)
            ],
        )
        or _compact_descriptor(master.recurring_object_positions, 12)
    )
    sections = [
        _Section(
            "physical_state_action",
            "Action",
            master_action,
            len(master_action),
            len(master_action),
            "REQUIRED",
        ),
        _Section(
            "characters",
            "Subject",
            master_subject,
            len(master_subject),
            len(master_subject),
            "REQUIRED",
        ),
        _Section(
            "location",
            "Location",
            master_location,
            len(master_location),
            len(master_location),
            "REQUIRED",
        ),
        _Section(
            "critical_objects",
            "Objects",
            master_objects,
            len(master_objects),
            len(master_objects),
            "REQUIRED",
        ),
        _Section("camera_composition", "Camera", master.basic_composition, 42, 90),
        _Section(
            "change_from_previous", "Layout", master.recurring_object_positions, 34, 65
        ),
    ]
    if manual:
        sections.append(
            _Section(
                "manual_scene_override",
                "Scene",
                sanitize_provider_visual_text(manual),
                0,
                90,
                "OPTIONAL",
            )
        )
    dropped = ["color_palette", "project_style_prose", "reference_role_prose"]
    if qa_correction is None:
        dropped.append("qa_correction")
    anchors = character_names + [location.name] + object_names
    return sections, dropped, anchors


def _identity_first(names: list[str], details: list[str]) -> str:
    if not names:
        return ""
    descriptions = []
    for name, detail in zip(names, details, strict=True):
        descriptions.append(detail.removeprefix(f"{name}, "))
    return _clean_join((", ".join(names), "; ".join(descriptions)))


def _identity_with_details(identity: str, *details: str) -> str:
    cleaned = []
    for detail in details:
        value = sanitize_provider_visual_text(detail)
        start = value.casefold().find(identity.casefold())
        if start >= 0:
            value = (value[:start] + value[start + len(identity) :]).strip(" ,.;:-")
        if value:
            cleaned.append(value)
    return _clean_join((identity, *cleaned))


def _compact_descriptor(value: str, maximum_words: int) -> str:
    cleaned = sanitize_provider_visual_text(value)
    clause = cleaned.split(".", 1)[0].split(";", 1)[0].strip(" ,")
    return " ".join(clause.split()[:maximum_words])


def _clean_join(values: object) -> str:
    return "; ".join(
        sanitize_provider_visual_text(str(item)).strip(" .;")
        for item in values  # type: ignore[union-attr]
        if item is not None and str(item).strip()
    )


def _truncate_at_word(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    candidate = value[:maximum].rstrip(" ,.;:-")
    boundary = candidate.rfind(" ")
    if boundary >= max(12, maximum // 2):
        candidate = candidate[:boundary].rstrip(" ,.;:-")
    return candidate


def _validate_required_sections(
    sections: list[_Section],
    rendered_values: list[str],
    prompt: str,
    required_anchors: list[str],
) -> None:
    missing = [
        section.field
        for section, value in zip(sections, rendered_values, strict=True)
        if section.priority == "REQUIRED" and value != section.value
    ]
    missing_anchors = [
        anchor
        for anchor in required_anchors
        if anchor.casefold() not in prompt.casefold()
    ]
    if missing or missing_anchors or not prompt.strip():
        joined = ", ".join(missing + missing_anchors) or "prompt"
        raise _required_state_truncated(
            f"required concepts could not be represented: {joined}"
        )


def _too_complex(reason: str) -> ImagePromptBuildError:
    return ImagePromptBuildError(f"ZIMAGE_PROMPT_TOO_COMPLEX: {reason}")


def _required_state_truncated(reason: str) -> ImagePromptBuildError:
    return ImagePromptBuildError(f"PROMPT_REQUIRED_STATE_TRUNCATED: {reason}")
