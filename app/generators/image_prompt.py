"""Build concise provider-ready image prompts from semantic visual state."""

from __future__ import annotations

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
from app.style_contracts import DEFAULT_IMAGE_STYLE_ID, apply_image_style_contract


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

        sections = [
            self._reference_section(references),
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
                    f"Purpose: {beat.visual_purpose}. Viewer must understand: "
                    f"{beat.what_viewer_should_understand}. First notice: "
                    f"{beat.visual_focus or beat.what_viewer_should_understand}."
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
        semantic_prompt = "\n\n".join(
            f"{heading}:\n{content.strip()}"
            for heading, content in sections
            if content.strip()
        )
        operation_line = f"VISUAL OPERATION:\n{operation.value}"
        semantic_prompt = f"{operation_line}\n\n{semantic_prompt}"
        if len(semantic_prompt) > self.max_semantic_characters:
            raise ImagePromptBuildError(
                "Semantic image prompt is too long; simplify the visual plan"
            )
        return apply_image_style_contract(semantic_prompt, style_id)

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
                "REFERENCE INSTRUCTIONS",
                build_reference_role_instruction(references),
            ),
            ("KEEP UNCHANGED", "\n".join(f"- {item}" for item in keep)),
            ("CHANGE ONLY", f"- {change}"),
            (
                "REQUIRED RESULT",
                (
                    f"The viewer must quickly understand: "
                    f"{beat.what_viewer_should_understand}. "
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
        prompt = "EDIT THE PROVIDED SOURCE IMAGE.\n\n" + "\n\n".join(
            f"{heading}:\n{content}" for heading, content in sections
        )
        if len(prompt) > self.max_semantic_characters:
            raise ImagePromptBuildError("Semantic edit prompt is too long")
        return apply_image_style_contract(prompt, style_id)

    @staticmethod
    def _reference_section(
        references: tuple[ImageReference, ...],
    ) -> tuple[str, str]:
        if not references:
            return (
                "REFERENCE INSTRUCTIONS",
                "No image reference is attached; follow declared continuity exactly.",
            )
        return (
            "REFERENCE INSTRUCTIONS",
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
            f"Same {master.id} environment. "
            f"Geometry: {master.environment_geometry}. "
            f"Composition anchor: {master.basic_composition}. "
            f"Keep palette: {master.color_palette}. "
            "Do not redesign the environment."
        )

    @staticmethod
    def _character_content(plan: VisualPlan, beat: VisualBeat) -> str:
        by_id = {character.id: character for character in plan.characters}
        if not beat.characters_visible:
            return "No recurring characters visible; do not introduce extra people."
        return "; ".join(
            f"{by_id[character_id].name}: {by_id[character_id].description}"
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
            f"{by_id[object_id].name}: {by_id[object_id].description}"
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
            f"Movement: {beat.camera_movement.value}. Reason: {beat.framing_reason}. "
            f"Make clear: {beat.information_added_beyond_narration}."
        )

    @staticmethod
    def _physical_state_content(beat: VisualBeat) -> str:
        parts = [beat.physical_state]
        if beat.safety_geography is not None:
            safety = beat.safety_geography
            parts.append(
                "Safety geography — "
                f"current: {safety.current_position}; "
                f"exit: {safety.exit_or_safe_area}; "
                f"distance/scale: {safety.distance_or_scale}; "
                f"obstacle: {safety.obstacle_between}"
            )
        return ". ".join(parts)

    @staticmethod
    def _change_content(beat: VisualBeat) -> str:
        parts = [beat.change_from_previous_beat]
        if beat.progressive_change is not None:
            change = beat.progressive_change
            parts.append(
                f"{change.subject_id}: {change.previous_state} → "
                f"{change.current_state} ({change.progression})"
            )
        if beat.anticipated_consequence is not None:
            parts.append(f"Visually prepare: {beat.anticipated_consequence}")
        return ". ".join(parts)
