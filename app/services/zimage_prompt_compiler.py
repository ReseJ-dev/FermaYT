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
    optional: bool = False


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
    sections.append(
        _Section(
            "style", "Style", _COMPACT_STYLE, len(_COMPACT_STYLE), len(_COMPACT_STYLE)
        )
    )
    # QA is intentionally last in semantic priority, but remains explicit on retry.
    if qa_correction is not None:
        correction = sanitize_provider_visual_text(qa_correction)
        if correction:
            sections.append(
                _Section(
                    "qa_correction",
                    "Correction",
                    correction,
                    min(len(correction), 75),
                    90,
                )
            )

    prefix = "Purely pictorial scene."
    fixed = len(prefix) + sum(len(f" {item.label}: .") for item in sections)
    available = KieZImageApiClient.PROMPT_MAX_CHARACTERS - fixed
    minimums = [min(len(item.value), item.minimum) for item in sections]
    if sum(minimums) > available:
        raise _too_complex("required semantic section minimums exceed 800 characters")

    allocations = minimums[:]
    remaining = available - sum(allocations)
    # Priority order is the section order. Fill each semantic section to its
    # preferred budget before lower-priority wording receives any extra space.
    for index, section in enumerate(sections):
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
        "preserved_fields": [item.field for item in sections],
        "compressed_fields": compressed,
        "dropped_optional_fields": dropped_optional,
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

    action_parts: list[str] = []
    if manual:
        action_parts.append(f"requested scene {_truncate_at_word(manual, 65)}")
    action_parts.append(f"required state {beat.physical_state}")
    if not _substantially_overlaps(
        beat.physical_state, beat.what_viewer_should_understand
    ):
        action_parts.append(beat.what_viewer_should_understand)
    action = _clean_join(action_parts)
    action_minimum = min(
        len(action),
        (76 if manual else 0) + min(62, len(beat.physical_state) + 15),
    )
    character_names = [by_character[item].name for item in beat.characters_visible]
    character_details = [
        f"{by_character[item].name}, {by_character[item].description}"
        for item in beat.characters_visible
    ]
    subject = _identity_first(character_names, character_details) or "no people visible"
    object_names = [by_object[item].name for item in beat.important_objects]
    object_details = [
        f"{by_object[item].name}, {by_object[item].description}"
        for item in beat.important_objects
    ]
    objects = (
        _identity_first(object_names, object_details) or "no required story object"
    )
    location_value = _identity_with_details(
        location.name, location.description, location.spatial_layout
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
        _Section("physical_state_action", "Action", action, action_minimum, 145),
        _Section("characters", "Subject", subject, 42, 105),
        _Section("location", "Location", location_value, 48, 105),
        _Section("critical_objects", "Objects", objects, 42, 105),
        _Section("camera_composition", "Camera", camera, 42, 90),
        _Section(
            "change_from_previous",
            "Change",
            change,
            min(len(change), 70),
            82,
        ),
    ]
    if beat.must_not_show:
        sections.append(
            _Section("must_not_show", "Avoid", _clean_join(beat.must_not_show), 28, 55)
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
    anchors = (
        character_names + [location.name] + object_names + [beat.camera_framing.value]
    )
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
    sections = [
        _Section(
            "physical_state_action",
            "Action",
            _clean_join(
                (
                    f"requested scene {_truncate_at_word(manual, 65)}",
                    f"required master {master.description}",
                )
                if manual
                else (f"required master {master.description}",)
            ),
            min(len(master.description) + (76 if manual else 16), 130),
            135,
        ),
        _Section(
            "characters",
            "Subject",
            _identity_first(character_names, character_details) or "no people visible",
            38,
            95,
        ),
        _Section(
            "location",
            "Location",
            _identity_with_details(
                location.name, location.description, master.environment_geometry
            ),
            48,
            115,
        ),
        _Section(
            "critical_objects",
            "Objects",
            _identity_first(object_names, object_details)
            or master.recurring_object_positions,
            40,
            100,
        ),
        _Section("camera_composition", "Camera", master.basic_composition, 42, 90),
        _Section(
            "change_from_previous", "Layout", master.recurring_object_positions, 34, 65
        ),
    ]
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


def _substantially_overlaps(left: str, right: str) -> bool:
    left_words = {item.casefold().strip(".,;:-") for item in left.split()}
    right_words = {item.casefold().strip(".,;:-") for item in right.split()}
    if not left_words or not right_words:
        return False
    return len(left_words & right_words) / min(len(left_words), len(right_words)) >= 0.5


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
        if not section.optional and not value.strip()
    ]
    missing_anchors = [
        anchor
        for anchor in required_anchors
        if anchor.casefold() not in prompt.casefold()
    ]
    if missing or missing_anchors or not prompt.strip():
        joined = ", ".join(missing + missing_anchors) or "prompt"
        raise _too_complex(f"required concepts could not be represented: {joined}")


def _too_complex(reason: str) -> ImagePromptBuildError:
    return ImagePromptBuildError(f"ZIMAGE_PROMPT_TOO_COMPLEX: {reason}")
