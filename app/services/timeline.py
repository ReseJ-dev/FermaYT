"""Build and inspect a persisted executable timeline without rendering video."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.asset_roles import RENDERABLE_VISUAL_ASSET_ROLES
from app.errors import ProjectVisualPlanError
from app.generation_scope import GenerationScope
from app.models.timeline import NormalizedOverlay, NormalizedTransform
from app.persistence import (
    BeatVisualResult,
    GeneratedVideoAsset,
    ProjectNarrationAsset,
    ProjectTimeline,
    VideoGenerationAttempt,
)
from app.repositories import (
    create_project_timeline,
    get_project,
    get_project_timeline_by_revision,
    get_project_visual_plan_record,
    get_visual_execution_plan,
    list_beat_visual_results,
)
from app.services.narration_alignment import align_project_visual_beats
from app.services.visual_planning import (
    hash_story_text,
    require_current_project_visual_plan,
)

TIMELINE_RHYTHM_VERSION = "timeline_rhythm_v3"


@dataclass(frozen=True, slots=True)
class TimelinePacingConfig:
    """Deterministic, non-billable visual pacing policy."""

    warning_static_seconds: float = 4.0
    strong_warning_static_seconds: float = 6.0
    automatic_motion_seconds: float = 4.0
    automatic_motion_enabled: bool = True
    subtle_motion_end_scale: float = 1.05

    def __post_init__(self) -> None:
        if self.warning_static_seconds <= 0:
            raise ValueError("warning_static_seconds must be positive")
        if self.strong_warning_static_seconds < self.warning_static_seconds:
            raise ValueError("strong static warning must not precede warning")
        if self.automatic_motion_seconds <= 0:
            raise ValueError("automatic_motion_seconds must be positive")
        if not 1.0 < self.subtle_motion_end_scale <= 1.15:
            raise ValueError("subtle motion scale must be between 1.0 and 1.15")

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VisualProgressionAnalysis:
    effective_screen_states: int
    longest_unchanged_hold: float
    average_screen_state_duration: float
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TimelineQualityReport:
    narration_duration: float
    visual_beats: int
    timeline_entries: int
    timestamp_sources: dict[str, int]
    alignment_methods: dict[str, int]
    average_visual_duration: float
    shortest_visual: float
    longest_visual: float
    operation_counts: dict[str, int]
    visual_gaps_repaired: int
    timing_warnings: int
    unique_generated_assets: int
    effective_screen_states: int
    longest_unchanged_hold: float
    average_screen_state_duration: float
    paid_generations_per_minute: float
    static_hold_warnings: int


def build_project_timeline(
    session: Session,
    project_id: str,
    execution_plan_id: str,
    narration_asset_id: str,
    generation_scope: GenerationScope | None = None,
    selected_beat_ids: tuple[str, ...] | None = None,
    pacing_config: TimelinePacingConfig | None = None,
) -> ProjectTimeline:
    """Create or reuse a complete gap-free timeline for current accepted assets."""
    scope = generation_scope or GenerationScope.full()
    pacing = pacing_config or TimelinePacingConfig()
    state = require_current_project_visual_plan(session, project_id)
    project = get_project(session, project_id)
    plan_record = get_project_visual_plan_record(session, project_id)
    execution = get_visual_execution_plan(session, execution_plan_id)
    narration = session.get(ProjectNarrationAsset, narration_asset_id)
    if project is None or plan_record is None or execution is None or narration is None:
        raise ValueError("Timeline inputs are incomplete")
    plan_revision = _stable_hash(state.plan.model_dump(mode="json"))
    if (
        execution.project_id != project_id
        or execution.visual_plan_id != plan_record.id
        or execution.visual_plan_revision != plan_revision
    ):
        raise ValueError("Execution plan is stale or belongs to another project")
    if narration.project_id != project_id:
        raise ValueError("Narration asset belongs to another project")
    if not _narration_matches_project(narration, project):
        raise ValueError("Narration asset is stale for current project TTS settings")
    if not _file_matches_sha256(narration.audio_path, narration.audio_sha256):
        raise ValueError("Narration audio file is missing or changed")
    alignment = align_project_visual_beats(
        session,
        project_id,
        narration,
    )
    accepted = _accepted_result_by_beat(
        list_beat_visual_results(
            session,
            project_id,
            execution_plan_id=execution.id,
            accepted_only=True,
        )
    )
    selected_ids = selected_beat_ids or scope.select_beat_ids(state.plan)
    selected_id_set = set(selected_ids)
    selected_beats = [
        beat for beat in state.plan.visual_beats if beat.id in selected_id_set
    ]
    if tuple(beat.id for beat in selected_beats) != selected_ids:
        raise ValueError("Generation scope must select an ordered plan prefix")
    missing = [beat.id for beat in selected_beats if beat.id not in accepted]
    if missing:
        raise ValueError(
            "Timeline requires accepted visual assets for beats: " + ", ".join(missing)
        )
    for beat in selected_beats:
        result = accepted[beat.id]
        _require_renderable_result(result, beat.id)
        if result.output_path is None or result.file_sha256 is None:
            raise ValueError(
                f"Accepted visual metadata is incomplete for beat {beat.id}"
            )
        if not _file_matches_sha256(result.output_path, result.file_sha256):
            raise ValueError(
                f"Accepted visual file is missing or changed for beat {beat.id}"
            )
    generated_videos = {
        beat.id: session.scalar(
            select(GeneratedVideoAsset)
            .join(VideoGenerationAttempt)
            .where(
                GeneratedVideoAsset.project_id == project_id,
                GeneratedVideoAsset.beat_id == beat.id,
                VideoGenerationAttempt.accepted_image_result_id == accepted[beat.id].id,
            )
            .order_by(GeneratedVideoAsset.created_at.desc())
            .limit(1)
        )
        for beat in selected_beats
    }
    for beat_id, video in generated_videos.items():
        if video is not None and not _file_matches_sha256(
            video.file_path, video.file_sha256
        ):
            raise ValueError(
                f"Generated video file is missing or changed for beat {beat_id}"
            )
    asset_snapshot = [
        {
            "beat_id": beat.id,
            "result_id": accepted[beat.id].id,
            "file_sha256": accepted[beat.id].file_sha256,
            "qa_revision": accepted[beat.id].qa_revision,
            "operation": accepted[beat.id].resolved_operation,
            "transform": accepted[beat.id].transform_metadata,
            "overlay": accepted[beat.id].overlay_metadata,
            "video_asset_id": (
                generated_videos[beat.id].id if generated_videos[beat.id] else None
            ),
            "video_sha256": (
                generated_videos[beat.id].file_sha256
                if generated_videos[beat.id]
                else None
            ),
        }
        for beat in selected_beats
    ]
    timeline_revision = _stable_hash(
        {
            "story_revision": hash_story_text(project.story_text),
            "visual_plan_revision": plan_revision,
            "execution_plan_revision": execution.resolution_revision,
            "accepted_visual_graph": asset_snapshot,
            "narration_revision": narration.generation_revision,
            "alignment_revision": alignment.alignment_revision,
            "rhythm_version": TIMELINE_RHYTHM_VERSION,
            "pacing": pacing.snapshot(),
            "generation_scope": scope.snapshot(),
        }
    )
    existing = get_project_timeline_by_revision(session, timeline_revision)
    if existing is not None:
        return existing

    timing_by_beat = {item.beat_id: item for item in alignment.beat_timings}
    selected_timings = [timing_by_beat[beat.id] for beat in selected_beats]
    timeline_duration = (
        narration.duration if scope.is_full else selected_timings[-1].audio_end
    )
    raw_intervals = [(item.audio_start, item.audio_end) for item in selected_timings]
    intervals, rhythm_warnings, repaired = plan_visual_rhythm(
        raw_intervals,
        timeline_duration,
    )
    entries: list[dict[str, Any]] = []
    intentional_static_beat_ids: set[str] = set()
    previous_asset_path: str | None = None
    previous_transform_metadata: dict[str, Any] | None = None
    for position, (beat, interval) in enumerate(
        zip(selected_beats, intervals, strict=True)
    ):
        result = accepted[beat.id]
        video_asset = generated_videos[beat.id]
        assert result.output_path is not None
        start_time, end_time = interval
        entry_duration = end_time - start_time
        transform_metadata = normalize_transform_metadata(result.transform_metadata)
        intentional_static = _is_intentional_static_hold(beat)
        if intentional_static:
            intentional_static_beat_ids.add(beat.id)
        if (
            transform_metadata is None
            and pacing.automatic_motion_enabled
            and entry_duration > pacing.automatic_motion_seconds
            and not intentional_static
        ):
            start_scale = 1.0
            if (
                result.output_path == previous_asset_path
                and previous_transform_metadata
            ):
                start_scale = float(
                    previous_transform_metadata.get("end_scale", start_scale)
                )
            transform_metadata = build_default_timeline_motion(
                beat,
                start_scale=start_scale,
                end_scale=pacing.subtle_motion_end_scale,
            )
        entries.append(
            {
                "position": position,
                "beat_id": beat.id,
                "beat_visual_result_id": result.id,
                "asset_type": "VIDEO" if video_asset is not None else "STILL",
                "video_asset_id": video_asset.id if video_asset is not None else None,
                "operation": (
                    video_asset.operation
                    if video_asset is not None
                    else result.resolved_operation
                ),
                "asset_path": (
                    video_asset.file_path
                    if video_asset is not None
                    else result.output_path
                ),
                "source_result_id": result.source_result_id,
                "master_scene_id": result.master_scene_id,
                "start_time": start_time,
                "end_time": end_time,
                "transform_metadata": (
                    None if video_asset is not None else transform_metadata
                ),
                "overlay_metadata": normalize_overlay_metadata(
                    result.overlay_metadata,
                    entry_duration,
                ),
                "transition_metadata": None,
                "clip_start": 0.0 if video_asset is not None else None,
                "clip_end": (
                    min(video_asset.duration, entry_duration)
                    if video_asset is not None
                    else None
                ),
                "mute_audio": True,
                "fit_metadata": {"mode": project.image_fit},
            }
        )
        previous_asset_path = (
            video_asset.file_path if video_asset is not None else result.output_path
        )
        previous_transform_metadata = (
            None if video_asset is not None else transform_metadata
        )
    progression = analyze_visual_progression(
        entries,
        timeline_duration,
        pacing,
        intentional_static_beat_ids=intentional_static_beat_ids,
    )
    warnings = (
        list(narration.timing_warnings)
        + list(alignment.warnings)
        + rhythm_warnings
        + list(progression.warnings)
    )
    if repaired:
        warnings.append(f"Repaired {repaired} narration gap/overlap boundaries")
    return create_project_timeline(
        session,
        project_id=project_id,
        visual_plan_id=plan_record.id,
        visual_plan_revision=plan_revision,
        execution_plan_id=execution.id,
        execution_plan_revision=execution.resolution_revision,
        narration_asset_id=narration.id,
        alignment_id=alignment.id,
        rhythm_version=TIMELINE_RHYTHM_VERSION,
        generation_scope_type=scope.type.value,
        generation_scope_value=(
            float(scope.value) if scope.value is not None else None
        ),
        generation_scope_version=scope.version,
        timeline_revision=timeline_revision,
        duration=timeline_duration,
        warnings=warnings,
        entries=entries,
    )


def plan_visual_rhythm(
    intervals: list[tuple[float, float]],
    duration: float,
    *,
    preferred_minimum: float = 1.5,
) -> tuple[list[tuple[float, float]], list[str], int]:
    """Repair boundaries conservatively while keeping narration synchronization primary."""
    if not intervals or duration <= 0:
        raise ValueError("Timeline needs positive narration duration and intervals")
    boundary_count = len(intervals) - 1
    boundaries = [0.0]
    repaired = 0
    for index in range(boundary_count):
        left_end = intervals[index][1]
        right_start = intervals[index + 1][0]
        if abs(left_end - right_start) > 1e-6:
            repaired += 1
        boundaries.append(max(0.0, min(duration, (left_end + right_start) / 2)))
    boundaries.append(duration)
    minimum = min(0.05, duration / (len(intervals) * 2))
    for index in range(1, len(boundaries)):
        boundaries[index] = max(boundaries[index], boundaries[index - 1] + minimum)
    boundaries[-1] = duration
    if boundaries[-2] >= duration:
        step = duration / len(intervals)
        boundaries = [step * index for index in range(len(intervals) + 1)]
        repaired += 1

    warnings: list[str] = []
    for index in range(len(intervals)):
        visual_duration = boundaries[index + 1] - boundaries[index]
        if visual_duration < preferred_minimum:
            warnings.append(
                f"Beat {index + 1} visual interval is short ({visual_duration:.2f}s); "
                "kept near its narration fact"
            )
            if index > 0:
                previous_duration = boundaries[index] - boundaries[index - 1]
                shift = min(0.25, max(0.0, previous_duration - preferred_minimum))
                boundaries[index] -= shift
    planned = [
        (round(boundaries[index], 6), round(boundaries[index + 1], 6))
        for index in range(len(intervals))
    ]
    _validate_timeline_intervals(planned, duration)
    return planned, warnings, repaired


def normalize_transform_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if metadata is None:
        return None
    transform_type = str(metadata.get("type", "")).strip()
    if not transform_type:
        raise ValueError("Transform type is missing")
    if transform_type == "PAN":
        normalized = NormalizedTransform(
            type="PAN",
            **{
                "from": metadata.get("from") or {"x": 0.45, "y": 0.5},
                "to": metadata.get("to") or {"x": 0.55, "y": 0.5},
            },
        )
    else:
        normalized = NormalizedTransform(
            type=transform_type,
            start_scale=float(metadata.get("start_scale") or 1.0),
            end_scale=float(metadata.get("end_scale") or 1.08),
            focus=metadata.get("focus") or {"x": 0.5, "y": 0.5},
            crop=metadata.get("crop"),
        )
    return normalized.model_dump(mode="json", by_alias=True, exclude_none=True)


def build_default_timeline_motion(
    beat: Any,
    *,
    start_scale: float = 1.0,
    end_scale: float = 1.05,
) -> dict[str, Any]:
    """Create a subtle deterministic push toward the semantic story focus."""
    focus_text = " ".join(
        str(value or "")
        for value in (
            beat.visual_focus,
            beat.camera_view,
            beat.what_viewer_should_understand,
        )
    ).casefold()
    x = 0.5
    y = 0.5
    if any(token in focus_text for token in (" left", "left ", "слева", "лев")):
        x = 0.32
    elif any(token in focus_text for token in (" right", "right ", "справа", "прав")):
        x = 0.68
    if any(
        token in focus_text for token in (" upper", "above", "top", "сверху", "верх")
    ):
        y = 0.34
    elif any(
        token in focus_text for token in (" lower", "below", "bottom", "снизу", "низ")
    ):
        y = 0.66
    target_scale = min(1.15, start_scale + (end_scale - 1.0))
    if target_scale - start_scale >= 0.01:
        transform = NormalizedTransform(
            type="FOCUS",
            start_scale=start_scale,
            end_scale=target_scale,
            focus={"x": x, "y": y},
        )
    else:
        from_x = max(0.2, min(0.8, x - 0.05))
        to_x = max(0.2, min(0.8, x + 0.05))
        transform = NormalizedTransform(
            type="PAN",
            **{
                "from": {"x": from_x, "y": y},
                "to": {"x": to_x, "y": y},
            },
        )
    return transform.model_dump(mode="json", by_alias=True, exclude_none=True)


def normalize_overlay_metadata(
    metadata: dict[str, Any] | None,
    entry_duration: float,
) -> dict[str, Any] | None:
    if metadata is None:
        return None
    overlay_type = str(metadata.get("type", "")).strip()
    instruction = str(metadata.get("instruction", "")).strip()
    anchor = str(metadata.get("semantic_anchor", "")).strip()
    if not overlay_type or not instruction or not anchor:
        raise ValueError("Overlay metadata is incomplete")
    directional = {"ARROW", "ROUTE", "MEASUREMENT", "DISTANCE", "DEPTH"}
    values: dict[str, Any] = {
        "type": overlay_type,
        "instruction": instruction,
        "semantic_anchor": anchor,
        "appear_offset": min(0.4, max(entry_duration * 0.2, 0.0)),
    }
    if overlay_type in directional:
        values.update(
            start={"x": 0.5, "y": 0.25},
            end={"x": 0.5, "y": 0.75},
        )
    else:
        values["center"] = {"x": 0.5, "y": 0.5}
    return NormalizedOverlay.model_validate(values).model_dump(
        mode="json",
        exclude_none=True,
    )


def timeline_is_current(session: Session, timeline: ProjectTimeline) -> bool:
    if timeline.rhythm_version != TIMELINE_RHYTHM_VERSION:
        return False
    project = get_project(session, timeline.project_id)
    if project is None:
        return False
    try:
        state = require_current_project_visual_plan(session, timeline.project_id)
    except ProjectVisualPlanError:
        return False
    if (
        _stable_hash(state.plan.model_dump(mode="json"))
        != timeline.visual_plan_revision
    ):
        return False
    narration = timeline.narration_asset
    if not _narration_matches_project(narration, project):
        return False
    if not _file_matches_sha256(narration.audio_path, narration.audio_sha256):
        return False
    accepted = _accepted_result_by_beat(
        list_beat_visual_results(
            session,
            timeline.project_id,
            execution_plan_id=timeline.execution_plan_id,
            accepted_only=True,
        )
    )
    return all(
        accepted.get(entry.beat_id) is not None
        and accepted[entry.beat_id].id == entry.beat_visual_result_id
        and accepted[entry.beat_id].output_path is not None
        and accepted[entry.beat_id].file_sha256 is not None
        and (
            _timeline_video_entry_is_current(session, entry)
            if entry.asset_type == "VIDEO"
            else _file_matches_sha256(
                accepted[entry.beat_id].output_path,
                accepted[entry.beat_id].file_sha256,
            )
        )
        for entry in timeline.entries
    )


def _timeline_video_entry_is_current(session: Session, entry: Any) -> bool:
    if not entry.video_asset_id or not entry.mute_audio:
        return False
    asset = session.get(GeneratedVideoAsset, entry.video_asset_id)
    return bool(
        asset
        and asset.beat_id == entry.beat_id
        and asset.file_path == entry.asset_path
        and _file_matches_sha256(asset.file_path, asset.file_sha256)
    )


def _require_renderable_result(result: BeatVisualResult, beat_id: str) -> None:
    if result.asset_role not in RENDERABLE_VISUAL_ASSET_ROLES:
        raise ValueError(
            f"Timeline beat {beat_id} uses non-renderable asset role "
            f"{result.asset_role}"
        )
    if result.generation_status != "SUCCEEDED" or not result.is_accepted:
        raise ValueError(f"Timeline beat {beat_id} has no accepted successful visual")
    if (
        result.resolved_operation
        in {"NEW_IMAGE", "REFERENCE_GENERATION", "EDIT_EXISTING"}
        and result.production_profile != "DRAFT"
        and (
            result.qa_status not in {"PASS", "PASS_WITH_WARNING"}
            or result.qa_warning is not None
        )
    ):
        raise ValueError(f"Timeline beat {beat_id} generated visual did not pass QA")


def format_timeline_debug(timeline: ProjectTimeline) -> str:
    lines = [
        f"Narration: {timeline.narration_asset.audio_path}",
        f"Duration: {timeline.duration:.3f}s",
        f"Timeline revision: {timeline.timeline_revision}",
        "",
    ]
    for entry in timeline.entries:
        lines.extend(
            [
                f"{_clock(entry.start_time)} – {_clock(entry.end_time)}",
                f"Beat {entry.position + 1:02d} · {entry.beat_id}",
                entry.operation,
                entry.asset_path,
                (
                    f"source: {entry.source_result_id}"
                    if entry.source_result_id is not None
                    else "source: root"
                ),
                json.dumps(
                    entry.transform_metadata or entry.overlay_metadata or {},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "",
            ]
        )
    if timeline.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in timeline.warnings)
    return "\n".join(lines)


def analyze_visual_progression(
    entries: list[Any],
    duration: float,
    config: TimelinePacingConfig | None = None,
    *,
    intentional_static_beat_ids: set[str] | frozenset[str] = frozenset(),
) -> VisualProgressionAnalysis:
    """Measure rendered screen states, including transforms and overlay appearance."""
    pacing = config or TimelinePacingConfig()
    effective_states = 0
    previous_state: str | None = None
    current_static_state: str | None = None
    current_static_duration = 0.0
    current_static_beats: list[str] = []
    longest_hold = 0.0
    warning_rows: list[tuple[float, tuple[str, ...]]] = []

    def finish_static_hold() -> None:
        nonlocal current_static_state, current_static_duration
        nonlocal current_static_beats, longest_hold
        if current_static_state is None:
            return
        longest_hold = max(longest_hold, current_static_duration)
        unique_beats = tuple(dict.fromkeys(current_static_beats))
        if current_static_duration > pacing.warning_static_seconds and not all(
            beat_id in intentional_static_beat_ids for beat_id in unique_beats
        ):
            warning_rows.append((current_static_duration, unique_beats))
        current_static_state = None
        current_static_duration = 0.0
        current_static_beats = []

    for entry in entries:
        start = float(_entry_value(entry, "start_time"))
        end = float(_entry_value(entry, "end_time"))
        entry_duration = end - start
        beat_id = str(_entry_value(entry, "beat_id"))
        asset_path = str(_entry_value(entry, "asset_path"))
        transform = _entry_value(entry, "transform_metadata")
        overlay = _entry_value(entry, "overlay_metadata")
        dynamic = _transform_has_motion(transform)
        overlay_offset = (
            min(entry_duration, float(overlay.get("appear_offset", 0.0)))
            if isinstance(overlay, dict)
            else 0.0
        )
        phases: list[tuple[float, str]] = []
        base_state = _screen_state_signature(asset_path, transform, None)
        if overlay_offset > 0:
            phases.append((overlay_offset, base_state))
        visible_duration = entry_duration - overlay_offset
        if visible_duration > 0:
            phases.append(
                (
                    visible_duration,
                    _screen_state_signature(asset_path, transform, overlay),
                )
            )

        for phase_duration, state in phases:
            if state != previous_state:
                effective_states += 1
                previous_state = state
            if dynamic:
                finish_static_hold()
                # A continuous transform has a materially different end state even
                # though it does not require a second asset or timeline cut.
                effective_states += 1
                previous_state = f"{state}:motion-end"
                continue
            if state == current_static_state:
                current_static_duration += phase_duration
                current_static_beats.append(beat_id)
            else:
                finish_static_hold()
                current_static_state = state
                current_static_duration = phase_duration
                current_static_beats = [beat_id]
    finish_static_hold()

    warnings: list[str] = []
    for hold_duration, beat_ids in warning_rows:
        severity = (
            "STRONG_WARNING"
            if hold_duration > pacing.strong_warning_static_seconds
            else "WARNING"
        )
        warnings.append(
            "STATIC_VISUAL_HOLD_TOO_LONG "
            f"severity={severity} duration={hold_duration:.2f}s "
            f"beats={','.join(beat_ids)}"
        )
    state_count = max(effective_states, 1)
    return VisualProgressionAnalysis(
        effective_screen_states=state_count,
        longest_unchanged_hold=round(longest_hold, 3),
        average_screen_state_duration=round(duration / state_count, 3),
        warnings=tuple(warnings),
    )


def build_timeline_quality_report(timeline: ProjectTimeline) -> TimelineQualityReport:
    durations = [item.end_time - item.start_time for item in timeline.entries]
    methods = Counter(item.alignment_method for item in timeline.alignment.beat_timings)
    operations = Counter(item.operation for item in timeline.entries)
    progression = analyze_visual_progression(timeline.entries, timeline.duration)
    unique_assets = {item.asset_path for item in timeline.entries}
    paid_operations = {"NEW_IMAGE", "REFERENCE_GENERATION", "EDIT_EXISTING"}
    paid_results = {
        item.beat_visual_result_id
        for item in timeline.entries
        if item.operation in paid_operations
    }
    repaired = sum(
        int(match.group(1))
        for warning in timeline.warnings
        if (match := re.search(r"Repaired (\d+)", warning)) is not None
    )
    return TimelineQualityReport(
        narration_duration=timeline.duration,
        visual_beats=len({item.beat_id for item in timeline.entries}),
        timeline_entries=len(timeline.entries),
        timestamp_sources={
            timeline.narration_asset.timestamp_source: len(timeline.entries)
        },
        alignment_methods=dict(sorted(methods.items())),
        average_visual_duration=round(sum(durations) / len(durations), 3),
        shortest_visual=round(min(durations), 3),
        longest_visual=round(max(durations), 3),
        operation_counts=dict(sorted(operations.items())),
        visual_gaps_repaired=repaired,
        timing_warnings=len(timeline.warnings),
        unique_generated_assets=len(unique_assets),
        effective_screen_states=progression.effective_screen_states,
        longest_unchanged_hold=progression.longest_unchanged_hold,
        average_screen_state_duration=progression.average_screen_state_duration,
        paid_generations_per_minute=round(
            len(paid_results) / (timeline.duration / 60),
            3,
        ),
        static_hold_warnings=sum(
            warning.startswith("STATIC_VISUAL_HOLD_TOO_LONG")
            for warning in timeline.warnings
        ),
    )


def _entry_value(entry: Any, name: str) -> Any:
    return entry.get(name) if isinstance(entry, dict) else getattr(entry, name)


def _screen_state_signature(
    asset_path: str,
    transform: object,
    overlay: object,
) -> str:
    return _stable_hash(
        {
            "asset": asset_path,
            "transform": transform,
            "overlay": overlay,
        }
    )


def _transform_has_motion(transform: object) -> bool:
    if not isinstance(transform, dict):
        return False
    transform_type = transform.get("type")
    if transform_type == "PAN":
        return transform.get("from") != transform.get("to")
    if transform_type in {"ZOOM_IN", "ZOOM_OUT", "FOCUS"}:
        return (
            abs(
                float(transform.get("end_scale", 1.0))
                - float(transform.get("start_scale", 1.0))
            )
            >= 0.01
        )
    return False


def _is_intentional_static_hold(beat: Any) -> bool:
    semantic_text = " ".join(
        str(value or "")
        for value in (
            beat.visual_purpose,
            beat.what_viewer_should_understand,
            beat.framing_reason,
            beat.physical_state,
            beat.change_from_previous_beat,
        )
    ).casefold()
    markers = (
        "intentional static hold",
        "complete stillness",
        "completely still",
        "hold completely still",
        "frozen frame",
        "remain motionless",
        "намеренно неподвиж",
        "полная неподвижность",
    )
    return any(marker in semantic_text for marker in markers)


def _accepted_result_by_beat(
    results: list[BeatVisualResult],
) -> dict[str, BeatVisualResult]:
    selected: dict[str, BeatVisualResult] = {}
    for result in results:
        current = selected.get(result.beat_id)
        if current is None or (
            result.accepted_at or result.updated_at,
            result.attempt,
            result.id,
        ) > (
            current.accepted_at or current.updated_at,
            current.attempt,
            current.id,
        ):
            selected[result.beat_id] = result
    return selected


def _validate_timeline_intervals(
    intervals: list[tuple[float, float]],
    duration: float,
) -> None:
    if abs(intervals[0][0]) > 1e-6 or abs(intervals[-1][1] - duration) > 1e-6:
        raise ValueError("Timeline does not cover the complete narration")
    for index, (start, end) in enumerate(intervals):
        if end <= start:
            raise ValueError("Timeline entry duration must be positive")
        if index and abs(start - intervals[index - 1][1]) > 1e-6:
            raise ValueError("Timeline contains a visual gap or overlap")


def _clock(value: float) -> str:
    milliseconds = round(value * 1000)
    minutes, remainder = divmod(milliseconds, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def _configured_tts_model(provider: str, model: str | None) -> str | None:
    if model is not None:
        return model
    if provider == "qwen":
        return "qwen3-tts-flash"
    if provider == "elevenlabs":
        return "eleven_multilingual_v2"
    return None


def _narration_matches_project(narration: ProjectNarrationAsset, project: Any) -> bool:
    return (
        narration.story_text_hash == hash_story_text(project.story_text)
        and narration.provider == project.tts_provider
        and narration.model
        == _configured_tts_model(project.tts_provider, project.tts_model)
        and narration.voice == project.tts_voice
        and narration.language
        == (project.tts_language if project.tts_provider == "qwen" else None)
    )


def _file_matches_sha256(path: str, expected: str) -> bool:
    file_path = Path(path)
    if not file_path.is_file():
        return False
    digest = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
