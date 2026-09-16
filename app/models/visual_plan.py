"""Validated semantic plan produced before any visual assets are generated."""

from collections.abc import Sequence
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VisualPlanReferenceError(ValueError):
    """Structured cross-reference failure raised by the canonical validator."""

    def __init__(
        self,
        *,
        category: str,
        owner_type: str,
        owner_id: str,
        field: str,
        invalid_id: str,
        reference_type: str,
        available_ids: set[str],
    ) -> None:
        self.category = category
        self.owner_type = owner_type
        self.owner_id = owner_id
        self.field = field
        self.invalid_id = invalid_id
        self.reference_type = reference_type
        self.available_ids = tuple(sorted(available_ids))
        if category == "INVALID_REFERENCE_TYPE":
            message = f"invalid {reference_type} reference: {invalid_id}"
        else:
            message = f"unknown {reference_type} id: {invalid_id}"
        super().__init__(message)

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "category": self.category,
            "owner_type": self.owner_type,
            "owner_id": self.owner_id,
            "field": self.field,
            "invalid_id": self.invalid_id,
        }
        if len(self.available_ids) <= 20:
            result["available_ids"] = list(self.available_ids)
        return result


class VisualPlanDuplicateIdError(ValueError):
    """Duplicate registry identifier with a machine-readable category."""

    def __init__(self, label: str, duplicate_id: str) -> None:
        self.category = "DUPLICATE_ID"
        self.label = label
        self.duplicate_id = duplicate_id
        super().__init__(f"duplicate {label} id: {duplicate_id}")


class VisualPlanMasterSceneAssignmentError(ValueError):
    """A beat cannot be assigned to a mastered environment unambiguously."""

    category = "MASTER_SCENE_ASSIGNMENT_REQUIRED"

    def __init__(
        self,
        *,
        beat_id: str,
        recurring_environment_ids: set[str],
        provided_master_scene_id: str | None,
        allowed_master_scene_ids: set[str],
    ) -> None:
        self.beat_id = beat_id
        self.recurring_environment_ids = tuple(sorted(recurring_environment_ids))
        self.provided_master_scene_id = provided_master_scene_id
        self.allowed_master_scene_ids = tuple(sorted(allowed_master_scene_ids))
        environments = ", ".join(self.recurring_environment_ids)
        allowed = ", ".join(self.allowed_master_scene_ids)
        super().__init__(
            f"beat {beat_id} in recurring environment {environments} must reference "
            f"one of these master scenes: {allowed}"
        )

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "category": self.category,
            "beat_id": self.beat_id,
            "recurring_environment_ids": list(self.recurring_environment_ids),
            "provided_master_scene_id": self.provided_master_scene_id,
            "allowed_master_scene_ids": list(self.allowed_master_scene_ids),
            "invariant": (
                "A beat in a mastered recurring environment must reference a master "
                "scene belonging to that environment."
            ),
        }
        if len(self.recurring_environment_ids) == 1:
            result["recurring_environment_id"] = self.recurring_environment_ids[0]
        return result


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
    environment_geometry: str = Field(min_length=1)
    recurring_object_positions: str = Field(min_length=1)
    color_palette: str = Field(min_length=1)
    basic_composition: str = Field(min_length=1)
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
    master_scene_id: str | None = Field(
        default=None,
        description="Immutable master environment governing this beat's continuity.",
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
    visual_focus: str | None = Field(
        default=None,
        description="Single element or action that must attract attention first.",
    )
    must_not_show: list[str] = Field(
        default_factory=list,
        description="Story mistakes or distracting elements explicitly excluded.",
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
    estimated_duration_seconds: float = Field(gt=0, le=12)

    @model_validator(mode="before")
    @classmethod
    def discard_overlay_description_for_non_overlay(
        cls, value: object
    ) -> object:
        """Normalize a harmless structured-provider inconsistency."""
        if not isinstance(value, dict):
            return value
        if (
            value.get("preferred_visual_operation") != VisualOperation.OVERLAY.value
            and value.get("overlay_description") is not None
        ):
            value = dict(value)
            value["overlay_description"] = None
        return value

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


class VisualPlanPlanningScope(_VisualModel):
    scope_type: str = Field(
        pattern=r"^(FULL|FIRST_SECONDS|FIRST_BEATS|STYLE_PREVIEW)$"
    )
    requested_seconds: float | None = Field(default=None, gt=0)
    requested_beats: int | None = Field(default=None, gt=0)
    source_start_char: int = Field(ge=0)
    source_end_char: int = Field(gt=0)
    continuation_boundary_char: int | None = Field(default=None, ge=0)
    source_hash: str = Field(min_length=64, max_length=64)
    is_partial: bool
    expected_beat_count: int = Field(gt=0)
    version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source_range(self) -> "VisualPlanPlanningScope":
        if self.source_end_char <= self.source_start_char:
            raise ValueError("planning source range must be positive")
        if self.is_partial and self.continuation_boundary_char is None:
            raise ValueError("partial planning scope requires a continuation boundary")
        if (
            self.continuation_boundary_char is not None
            and not self.source_start_char
            <= self.continuation_boundary_char
            <= self.source_end_char
        ):
            raise ValueError("continuation boundary must be inside the source range")
        return self


class VisualPlan(_VisualModel):
    """Whole-story semantic visual plan; it intentionally contains no prompts."""

    story_summary: str = Field(min_length=1)
    visual_strategy: str = Field(min_length=1)
    planning_scope: VisualPlanPlanningScope | None = None
    characters: list[CharacterDefinition]
    locations: list[LocationDefinition]
    important_objects: list[ImportantObjectDefinition]
    recurring_environments: list[RecurringEnvironment]
    possible_master_scenes: list[MasterScene]
    visual_beats: list[VisualBeat] = Field(min_length=1)

    @property
    def is_partial(self) -> bool:
        return bool(self.planning_scope and self.planning_scope.is_partial)

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
        recurring_environments_by_location: dict[str, set[str]] = {}
        for environment in self.recurring_environments:
            recurring_environments_by_location.setdefault(
                environment.location_id, set()
            ).add(environment.id)
        master_scenes_by_location: dict[str, set[str]] = {}
        for master in self.possible_master_scenes:
            master_scenes_by_location.setdefault(master.location_id, set()).add(
                master.id
            )

        for environment in self.recurring_environments:
            _require_known(
                environment.location_id,
                location_ids,
                "location",
                owner_type="RecurringEnvironment",
                owner_id=environment.id,
                field="location_id",
            )

        for master_scene in self.possible_master_scenes:
            _require_known(
                master_scene.location_id,
                location_ids,
                "location",
                owner_type="MasterScene",
                owner_id=master_scene.id,
                field="location_id",
            )
            _require_all_known(
                master_scene.characters_visible,
                character_ids,
                "character",
                owner_type="MasterScene",
                owner_id=master_scene.id,
                field="characters_visible",
            )
            _require_all_known(
                master_scene.important_objects,
                object_ids,
                "important object",
                owner_type="MasterScene",
                owner_id=master_scene.id,
                field="important_objects",
            )

        previous_beat_ids: set[str] = set()
        visual_locations = {
            master_scene.id: master_scene.location_id
            for master_scene in self.possible_master_scenes
        }
        for beat in self.visual_beats:
            _require_known(
                beat.location_id,
                location_ids,
                "location",
                owner_type="VisualBeat",
                owner_id=beat.id,
                field="location_id",
            )
            _require_all_known(
                beat.characters_visible,
                character_ids,
                "character",
                owner_type="VisualBeat",
                owner_id=beat.id,
                field="characters_visible",
            )
            _require_all_known(
                beat.important_objects,
                object_ids,
                "important object",
                owner_type="VisualBeat",
                owner_id=beat.id,
                field="important_objects",
            )
            recurring_environment_ids = recurring_environments_by_location.get(
                beat.location_id, set()
            )
            allowed_master_scene_ids = master_scenes_by_location.get(
                beat.location_id, set()
            )
            if (
                recurring_environment_ids
                and allowed_master_scene_ids
                and beat.master_scene_id not in allowed_master_scene_ids
            ):
                raise VisualPlanMasterSceneAssignmentError(
                    beat_id=beat.id,
                    recurring_environment_ids=recurring_environment_ids,
                    provided_master_scene_id=beat.master_scene_id,
                    allowed_master_scene_ids=allowed_master_scene_ids,
                )
            if beat.source_visual_id is not None:
                valid_sources = master_scene_ids | previous_beat_ids
                _require_known(
                    beat.source_visual_id,
                    valid_sources,
                    "source visual",
                    owner_type="VisualBeat",
                    owner_id=beat.id,
                    field="source_visual_id",
                )
            if (
                beat.camera_framing in {ShotFraming.CLOSE, ShotFraming.DETAIL}
                and beat.geography_established_by is None
            ):
                raise ValueError(
                    "CLOSE and DETAIL beats require geography_established_by"
                )
            if beat.geography_established_by is not None:
                valid_geography = master_scene_ids | previous_beat_ids
                _require_known(
                    beat.geography_established_by,
                    valid_geography,
                    "geography visual",
                    owner_type="VisualBeat",
                    owner_id=beat.id,
                    field="geography_established_by",
                )
                established_location = visual_locations[beat.geography_established_by]
                if established_location != beat.location_id:
                    raise VisualPlanReferenceError(
                        category="INVALID_REFERENCE_TYPE",
                        owner_type="VisualBeat",
                        owner_id=beat.id,
                        field="geography_established_by",
                        invalid_id=beat.geography_established_by,
                        reference_type="geography visual for the same location",
                        available_ids={
                            visual_id
                            for visual_id in valid_geography
                            if visual_locations[visual_id] == beat.location_id
                        },
                    )
            if beat.master_scene_id is not None:
                _require_known(
                    beat.master_scene_id,
                    master_scene_ids,
                    "master scene",
                    owner_type="VisualBeat",
                    owner_id=beat.id,
                    field="master_scene_id",
                )
                master_location = visual_locations[beat.master_scene_id]
                if master_location != beat.location_id:
                    raise VisualPlanReferenceError(
                        category="INVALID_REFERENCE_TYPE",
                        owner_type="VisualBeat",
                        owner_id=beat.id,
                        field="master_scene_id",
                        invalid_id=beat.master_scene_id,
                        reference_type="master scene for the beat location",
                        available_ids={
                            master_id
                            for master_id in master_scene_ids
                            if visual_locations[master_id] == beat.location_id
                        },
                    )
                for reference_id in (
                    beat.source_visual_id,
                    beat.geography_established_by,
                ):
                    if reference_id in master_scene_ids and reference_id != beat.master_scene_id:
                        raise ValueError(
                            "beat cannot reference a different master scene"
                        )
            if beat.progressive_change is not None:
                valid_subjects = character_ids | object_ids | location_ids
                _require_known(
                    beat.progressive_change.subject_id,
                    valid_subjects,
                    "progressive change subject",
                    owner_type="VisualBeat",
                    owner_id=beat.id,
                    field="progressive_change.subject_id",
                )
            previous_beat_ids.add(beat.id)
            visual_locations[beat.id] = beat.location_id

        # Evaluate these collections so duplicate checks also apply when empty.
        _ = environment_ids, beat_ids
        return self


def _unique_ids(items: Sequence[_Identified], label: str) -> set[str]:
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        duplicate_id = next(item_id for item_id in ids if ids.count(item_id) > 1)
        raise VisualPlanDuplicateIdError(label, duplicate_id)
    return set(ids)


def _require_known(
    value: str,
    known: set[str],
    label: str,
    *,
    owner_type: str,
    owner_id: str,
    field: str,
) -> None:
    if value not in known:
        raise VisualPlanReferenceError(
            category="UNKNOWN_REFERENCE",
            owner_type=owner_type,
            owner_id=owner_id,
            field=field,
            invalid_id=value,
            reference_type=label,
            available_ids=known,
        )


def _require_all_known(
    values: list[str],
    known: set[str],
    label: str,
    *,
    owner_type: str,
    owner_id: str,
    field: str,
) -> None:
    for value in values:
        _require_known(
            value,
            known,
            label,
            owner_type=owner_type,
            owner_id=owner_id,
            field=field,
        )
