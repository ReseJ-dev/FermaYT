"""Validated semantic plan produced before any visual assets are generated."""

from collections.abc import Sequence
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VisualOperation(str, Enum):
    """The visual operation preferred by the director for a beat."""

    NEW_IMAGE = "NEW_IMAGE"
    REFERENCE_GENERATION = "REFERENCE_GENERATION"
    EDIT_EXISTING = "EDIT_EXISTING"
    REUSE = "REUSE"
    TRANSFORM = "TRANSFORM"
    OVERLAY = "OVERLAY"


class ShotFraming(str, Enum):
    """Semantic framing selected for a clear storytelling reason."""

    WIDE = "WIDE"
    MEDIUM = "MEDIUM"
    CLOSE = "CLOSE"
    DETAIL = "DETAIL"
    CUTAWAY_DIAGRAM = "CUTAWAY_DIAGRAM"


class CameraMovement(str, Enum):
    """Movement applied on the timeline without requiring an unrelated image."""

    STATIC = "STATIC"
    SUBTLE_ZOOM = "SUBTLE_ZOOM"
    ZOOM = "ZOOM"
    PAN = "PAN"
    CROP = "CROP"
    HIGHLIGHT = "HIGHLIGHT"


class _VisualModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def trim_strings(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class _Identified(Protocol):
    id: str


class CharacterDefinition(_VisualModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class LocationDefinition(_VisualModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    spatial_layout: str = Field(
        min_length=1,
        description="Stable spatial relationships the viewer must understand.",
    )


class ImportantObjectDefinition(_VisualModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    story_function: str = Field(min_length=1)


class RecurringEnvironment(_VisualModel):
    id: str = Field(min_length=1)
    location_id: str = Field(min_length=1)
    continuity_requirements: str = Field(
        min_length=1,
        description="What must remain spatially and visually stable between beats.",
    )


class MasterScene(_VisualModel):
    id: str = Field(min_length=1)
    location_id: str = Field(min_length=1)
    description: str = Field(
        min_length=1,
        description="Reusable establishing composition and physical layout.",
    )
    characters_visible: list[str] = Field(default_factory=list)
    important_objects: list[str] = Field(default_factory=list)


class ProgressiveChange(_VisualModel):
    """A stable subject shown at a new stage of a developing condition."""

    subject_id: str = Field(min_length=1)
    previous_state: str = Field(min_length=1)
    current_state: str = Field(min_length=1)
    progression: str = Field(
        min_length=1,
        description="Direction of change, such as open to blocked or clear to opaque.",
    )


class SafetyGeography(_VisualModel):
    """Spatial facts needed to understand access to or distance from safety."""

    current_position: str = Field(min_length=1)
    exit_or_safe_area: str = Field(min_length=1)
    distance_or_scale: str = Field(min_length=1)
    obstacle_between: str = Field(min_length=1)


class VisualBeat(_VisualModel):
    id: str = Field(min_length=1)
    narration_segment: str = Field(min_length=1)
    visual_purpose: str = Field(min_length=1)
    what_viewer_should_understand: str = Field(min_length=1)
    location_id: str = Field(min_length=1)
    characters_visible: list[str] = Field(default_factory=list)
    important_objects: list[str] = Field(default_factory=list)
    camera_framing: ShotFraming
    camera_view: str = Field(min_length=1)
    framing_reason: str = Field(
        min_length=1,
        description="Why this framing improves understanding at this exact beat.",
    )
    camera_movement: CameraMovement
    geography_established_by: str | None = Field(
        default=None,
        description="Earlier wide/master visual that locates this closer view.",
    )
    physical_state: str = Field(
        min_length=1,
        description="Positions, routes, obstructions and object states in this beat.",
    )
    progressive_change: ProgressiveChange | None = None
    safety_geography: SafetyGeography | None = None
    anticipated_consequence: str | None = Field(
        default=None,
        description="Physical consequence visually prepared before narration confirms it.",
    )
    change_from_previous_beat: str = Field(min_length=1)
    information_added_beyond_narration: str = Field(
        min_length=1,
        description="Spatial, causal, scale or state information supplied by the visual.",
    )
    preferred_visual_operation: VisualOperation
    source_visual_id: str | None = Field(
        default=None,
        description="Earlier beat or master scene used as the visual source.",
    )
    overlay_description: str | None = Field(
        default=None,
        description="A restrained route, arrow or highlight; never a full slide.",
    )
    estimated_duration_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def require_source_for_dependent_operation(self) -> "VisualBeat":
        dependent_operations = {
            VisualOperation.REFERENCE_GENERATION,
            VisualOperation.EDIT_EXISTING,
            VisualOperation.REUSE,
            VisualOperation.TRANSFORM,
            VisualOperation.OVERLAY,
        }
        if (
            self.preferred_visual_operation in dependent_operations
            and self.source_visual_id is None
        ):
            raise ValueError("this visual operation requires source_visual_id")
        if (
            self.preferred_visual_operation is VisualOperation.OVERLAY
            and self.overlay_description is None
        ):
            raise ValueError("OVERLAY requires overlay_description")
        if (
            self.preferred_visual_operation is not VisualOperation.OVERLAY
            and self.overlay_description is not None
        ):
            raise ValueError("overlay_description is only valid for OVERLAY")
        return self


class VisualPlan(_VisualModel):
    """Whole-story semantic visual plan; it intentionally contains no prompts."""

    story_summary: str = Field(min_length=1)
    visual_strategy: str = Field(min_length=1)
    characters: list[CharacterDefinition]
    locations: list[LocationDefinition]
    important_objects: list[ImportantObjectDefinition]
    recurring_environments: list[RecurringEnvironment]
    possible_master_scenes: list[MasterScene]
    visual_beats: list[VisualBeat] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> "VisualPlan":
        character_ids = _unique_ids(self.characters, "character")
        location_ids = _unique_ids(self.locations, "location")
        object_ids = _unique_ids(self.important_objects, "important object")
        environment_ids = _unique_ids(
            self.recurring_environments,
            "recurring environment",
        )
        master_scene_ids = _unique_ids(self.possible_master_scenes, "master scene")
        beat_ids = _unique_ids(self.visual_beats, "visual beat")

        for environment in self.recurring_environments:
            _require_known(environment.location_id, location_ids, "location")

        for master_scene in self.possible_master_scenes:
            _require_known(master_scene.location_id, location_ids, "location")
            _require_all_known(
                master_scene.characters_visible,
                character_ids,
                "character",
            )
            _require_all_known(
                master_scene.important_objects,
                object_ids,
                "important object",
            )

        previous_beat_ids: set[str] = set()
        visual_locations = {
            master_scene.id: master_scene.location_id
            for master_scene in self.possible_master_scenes
        }
        for beat in self.visual_beats:
            _require_known(beat.location_id, location_ids, "location")
            _require_all_known(beat.characters_visible, character_ids, "character")
            _require_all_known(beat.important_objects, object_ids, "important object")
            if beat.source_visual_id is not None:
                valid_sources = master_scene_ids | previous_beat_ids
                _require_known(beat.source_visual_id, valid_sources, "source visual")
            if beat.camera_framing in {ShotFraming.CLOSE, ShotFraming.DETAIL}:
                if beat.geography_established_by is None:
                    raise ValueError(
                        "CLOSE and DETAIL beats require geography_established_by"
                    )
            if beat.geography_established_by is not None:
                valid_geography = master_scene_ids | previous_beat_ids
                _require_known(
                    beat.geography_established_by,
                    valid_geography,
                    "geography visual",
                )
                established_location = visual_locations[beat.geography_established_by]
                if established_location != beat.location_id:
                    raise ValueError(
                        "geography visual must establish the same location"
                    )
            if beat.progressive_change is not None:
                valid_subjects = character_ids | object_ids | location_ids
                _require_known(
                    beat.progressive_change.subject_id,
                    valid_subjects,
                    "progressive change subject",
                )
            previous_beat_ids.add(beat.id)
            visual_locations[beat.id] = beat.location_id

        # Evaluate these collections so duplicate checks also apply when empty.
        _ = environment_ids, beat_ids
        return self


def _unique_ids(items: Sequence[_Identified], label: str) -> set[str]:
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate {label} id")
    return set(ids)


def _require_known(value: str, known: set[str], label: str) -> None:
    if value not in known:
        raise ValueError(f"unknown {label} id: {value}")


def _require_all_known(values: list[str], known: set[str], label: str) -> None:
    for value in values:
        _require_known(value, known, label)
