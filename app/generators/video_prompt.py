"""Motion-first prompts that preserve authority established by accepted stills."""

from __future__ import annotations

from dataclasses import dataclass

from app.generators.image_prompt import sanitize_provider_visual_text
from app.models.visual_plan import VisualBeat, VisualPlan


@dataclass(frozen=True, slots=True)
class VideoPromptBuilder:
    version: str = "video_prompt_v1"
    maximum_characters: int = 2_000

    def build_from_beat(
        self,
        plan: VisualPlan,
        beat: VisualBeat,
        *,
        manual_motion_prompt: str | None = None,
    ) -> str:
        characters = {item.id: item for item in plan.characters}
        location = next(item for item in plan.locations if item.id == beat.location_id)
        identities = (
            "; ".join(
                f"{characters[item].name}: {characters[item].description}"
                for item in beat.characters_visible
            )
            or "No new people"
        )
        action = manual_motion_prompt or beat.change_from_previous_beat
        motion = _motion_intensity(beat)
        prompt = (
            "Animate the accepted source frame; do not redesign it.\n\n"
            f"ACTION: {sanitize_provider_visual_text(action)}.\n"
            f"TEMPORAL CHANGE: {sanitize_provider_visual_text(beat.physical_state)}.\n"
            f"CAMERA: {sanitize_provider_visual_text(beat.camera_view)}; "
            f"{beat.camera_movement.value.lower().replace('_', ' ')}.\n"
            f"MOTION: {motion}.\n"
            f"KEEP STABLE: same {sanitize_provider_visual_text(location.name)}, "
            f"same geometry, same drawing style, same character identity and clothing; "
            f"{sanitize_provider_visual_text(identities)}. No new people, objects, text, "
            "dialogue, cuts, borders, or style change."
        )
        if beat.must_not_show:
            prompt += (
                " Avoid: "
                + "; ".join(
                    sanitize_provider_visual_text(item) for item in beat.must_not_show
                )
                + "."
            )
        if len(prompt) > self.maximum_characters:
            prompt = prompt[: self.maximum_characters].rsplit(" ", 1)[0]
        return prompt


def _motion_intensity(beat: VisualBeat) -> str:
    state = f"{beat.physical_state} {beat.change_from_previous_beat}".casefold()
    strong = ("collapse", "falling", "rising", "running", "rapid", "violent")
    if any(item in state for item in strong):
        return "clear controlled physical movement; preserve readable silhouettes"
    return "minimal controlled movement with stable shapes and continuity"
