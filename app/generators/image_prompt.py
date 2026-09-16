"""Build concise provider-ready image prompts from semantic visual state."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.errors import ImagePromptBuildError
from app.generators.style_reference import build_reference_role_instruction
from app.models.visual_plan import (
    LocationDefinition,
    MasterScene,
    VisualBeat,
    VisualOperation,
    VisualPlan,
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
        location = next(
            (item for item in plan.locations if item.id == beat.location_id),
            None,
        )
        if location is None:
            raise ImagePromptBuildError(f"Unknown beat location: {beat.location_id}")
        master = next(
            (
                item
                for item in plan.possible_master_scenes
                if item.id == beat.master_scene_id
            ),
            None,
        )

        sections: list[tuple[str, str]] = []
        reference_section = self._reference_section(references)
        if reference_section is not None:
            sections.append(reference_section)
        sections.extend([
            (
                "LOCATION CONTINUITY",
                self._location_content(location, master),
            ),
            (
                "CHARACTER CONTINUITY",
                self._character_content(plan, beat),
            ),
            (
                "OBJECT CONTINUITY",
                self._object_content(plan, beat, master),
            ),
            (
                "CURRENT CAMERA / COMPOSITION",
                self._camera_content(beat),
            ),
            ("CURRENT PHYSICAL STATE", self._physical_state_content(beat)),
            ("WHAT CHANGED", self._change_content(beat)),
            (
                "VISUAL FOCUS",
                (
                    f"Make {beat.visual_focus or beat.what_viewer_should_understand} "
                    f"the first noticeable element. Clearly depict "
                    f"{beat.what_viewer_should_understand}. The frame should "
                    f"{beat.visual_purpose}."
                ),
            ),
        ])
        if project_style_prompt is not None and project_style_prompt.strip():
            sections.insert(
                1,
                ("PROJECT STYLE DIRECTION", project_style_prompt.strip()),
            )
        if beat.must_not_show:
            sections.append(("DO NOT SHOW", "; ".join(beat.must_not_show)))
        sections.append(
            (
                "SIMPLIFICATION RULE",
                (
                    "Use only the shapes needed to understand this beat. Keep one "
                    "obvious focal action, remove unrelated background clutter, and "
                    "preserve factual layout over decorative detail. A simple frame "
                    "that clearly advances the story is better than a beautiful "
                    "frame that only decorates the narration."
                ),
            )
        )
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
        location = next(
            (item for item in plan.locations if item.id == beat.location_id),
            None,
        )
        if location is None:
            raise ImagePromptBuildError(f"Unknown beat location: {beat.location_id}")
        master = next(
            (
                item
                for item in plan.possible_master_scenes
                if item.id == beat.master_scene_id
            ),
            None,
        )
        keep = [
            f"location identity and layout: {location.spatial_layout}",
            f"camera and framing: {beat.camera_framing.value}; {beat.camera_view}",
        ]
        if master is not None:
            keep.extend(
                (
                    f"environment geometry: {master.environment_geometry}",
                    f"recurring object positions: {master.recurring_object_positions}",
                )
            )
        change = beat.change_from_previous_beat
        if beat.progressive_change is not None:
            progression = beat.progressive_change
            change = (
                f"{progression.subject_id}: {progression.previous_state} -> "
                f"{progression.current_state}. {progression.progression}"
            )
        sections = [
            (
                "Attached image guidance",
                build_reference_role_instruction(references),
            ),
            ("KEEP UNCHANGED", "; ".join(keep)),
            ("CHANGE ONLY", change),
            (
                "REQUIRED RESULT",
                (
                    f"Make {beat.what_viewer_should_understand} immediately clear. "
                    "Do not redesign the environment or add unrelated detail."
                ),
            ),
        ]
        if project_style_prompt is not None and project_style_prompt.strip():
            sections.insert(
                1,
                ("PROJECT STYLE DIRECTION", project_style_prompt.strip()),
            )
        if beat.must_not_show:
            sections.append(("DO NOT SHOW", "; ".join(beat.must_not_show)))
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
    def _location_content(
        location: LocationDefinition,
        master: MasterScene | None,
    ) -> str:
        if master is None:
            return (
                f"{location.description}. Preserve this layout: "
                f"{location.spatial_layout}."
            )
        return (
            "Use the same recurring environment. "
            f"Preserve this geometry. {master.environment_geometry}. "
            f"Use this composition. {master.basic_composition}. "
            f"Keep this palette. {master.color_palette}. "
            "Do not redesign the environment."
        )

    @staticmethod
    def _character_content(plan: VisualPlan, beat: VisualBeat) -> str:
        by_id = {character.id: character for character in plan.characters}
        if not beat.characters_visible:
            return "No recurring characters visible; do not introduce extra people."
        return "; ".join(
            f"{by_id[character_id].name} appears as "
            f"{by_id[character_id].description}"
            for character_id in beat.characters_visible
        )

    @staticmethod
    def _object_content(
        plan: VisualPlan,
        beat: VisualBeat,
        master: MasterScene | None,
    ) -> str:
        by_id = {item.id: item for item in plan.important_objects}
        objects = [
            f"{by_id[object_id].name} appears as {by_id[object_id].description}"
            for object_id in beat.important_objects
        ]
        if master is not None:
            objects.append(
                f"Stable positions: {master.recurring_object_positions}"
            )
        return "; ".join(objects) or "No recurring story object is required."

    @staticmethod
    def _camera_content(beat: VisualBeat) -> str:
        return (
            f"{beat.camera_framing.value}. {beat.camera_view}. "
            f"Use {beat.camera_movement.value} movement because {beat.framing_reason}. "
            f"Clearly show {beat.information_added_beyond_narration}."
        )

    @staticmethod
    def _physical_state_content(beat: VisualBeat) -> str:
        parts = [beat.physical_state]
        if beat.safety_geography is not None:
            safety = beat.safety_geography
            parts.append(
                f"Place the subject at {safety.current_position}, the safe exit at "
                f"{safety.exit_or_safe_area}, show {safety.distance_or_scale} between "
                f"them, and place {safety.obstacle_between} in the way"
            )
        return ". ".join(parts)

    @staticmethod
    def _change_content(beat: VisualBeat) -> str:
        parts = [beat.change_from_previous_beat]
        if beat.progressive_change is not None:
            change = beat.progressive_change
            parts.append(
                f"Show {change.subject_id} moving from {change.previous_state} to "
                f"{change.current_state}, progressing as {change.progression}"
            )
        if beat.anticipated_consequence is not None:
            parts.append(f"Visually prepare {beat.anticipated_consequence}")
        return ". ".join(parts)


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
        len(f"\n\n{_PROVIDER_SECTION_LEADS[heading]} ")
        for heading, _ in entries
    )
    content_budget = maximum - fixed_length
    minimum_per_section = 24
    if content_budget < minimum_per_section * len(entries):
        raise ImagePromptBuildError(
            "Semantic image prompt is too long for the configured prompt budget"
        )

    lengths = [len(content) for _, content in entries]
    allocations = [min(length, minimum_per_section) for length in lengths]
    remaining = content_budget - sum(allocations)
    while remaining > 0:
        active = [
            index
            for index, length in enumerate(lengths)
            if allocations[index] < length
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
    "LOCATION CONTINUITY": "Draw the recurring setting with this stable layout.",
    "PROJECT STYLE DIRECTION": "Follow this project drawing direction.",
    "CHARACTER CONTINUITY": "Show these people with their established roles and appearance.",
    "OBJECT CONTINUITY": "Include these story objects in their established positions.",
    "CURRENT CAMERA / COMPOSITION": "Frame the scene this way.",
    "CURRENT PHYSICAL STATE": "Depict this physical situation.",
    "WHAT CHANGED": "Make this new physical change clearly visible.",
    "VISUAL FOCUS": "Guide attention to the story-critical action.",
    "DO NOT SHOW": "Exclude these story mistakes.",
    "SIMPLIFICATION RULE": "Keep the image visually simple and immediately readable.",
    "KEEP UNCHANGED": "Preserve the following established elements.",
    "CHANGE ONLY": "Change only this physical state.",
    "REQUIRED RESULT": "The edited result must communicate this fact.",
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
