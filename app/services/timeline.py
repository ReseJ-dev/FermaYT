"""Build and inspect a persisted executable timeline without rendering video."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.errors import ProjectVisualPlanError
from app.models.timeline import NormalizedOverlay, NormalizedTransform
from app.persistence import (
    BeatVisualResult,
    ProjectNarrationAsset,
    ProjectTimeline,
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

TIMELINE_RHYTHM_VERSION = "timeline_rhythm_v1"


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


def build_project_timeline(
    session: Session,
    project_id: str,
    execution_plan_id: str,
    narration_asset_id: str,
) -> ProjectTimeline:
    """Create or reuse a complete gap-free timeline for current accepted assets."""
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
    missing = [beat.id for beat in state.plan.visual_beats if beat.id not in accepted]
    if missing:
        raise ValueError(
            "Timeline requires accepted visual assets for beats: " + ", ".join(missing)
        )
    for beat in state.plan.visual_beats:
        result = accepted[beat.id]
        if result.output_path is None or result.file_sha256 is None:
            raise ValueError(f"Accepted visual metadata is incomplete for beat {beat.id}")
        if not _file_matches_sha256(result.output_path, result.file_sha256):
            raise ValueError(f"Accepted visual file is missing or changed for beat {beat.id}")
    asset_snapshot = [
        {
            "beat_id": beat.id,
            "result_id": accepted[beat.id].id,
            "file_sha256": accepted[beat.id].file_sha256,
            "qa_revision": accepted[beat.id].qa_revision,
            "operation": accepted[beat.id].resolved_operation,
            "transform": accepted[beat.id].transform_metadata,
            "overlay": accepted[beat.id].overlay_metadata,
        }
        for beat in state.plan.visual_beats
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
        }
    )
    existing = get_project_timeline_by_revision(session, timeline_revision)
    if existing is not None:
        return existing

    raw_intervals = [
        (item.audio_start, item.audio_end) for item in alignment.beat_timings
    ]
    intervals, rhythm_warnings, repaired = plan_visual_rhythm(
        raw_intervals,
        narration.duration,
    )
    entries: list[dict[str, Any]] = []
    for position, (beat, interval) in enumerate(
        zip(state.plan.visual_beats, intervals, strict=True)
    ):
        result = accepted[beat.id]
        assert result.output_path is not None
        start_time, end_time = interval
        entries.append(
            {
                "position": position,
                "beat_id": beat.id,
                "beat_visual_result_id": result.id,
                "operation": result.resolved_operation,
                "asset_path": result.output_path,
                "source_result_id": result.source_result_id,
                "master_scene_id": result.master_scene_id,
                "start_time": start_time,
                "end_time": end_time,
                "transform_metadata": normalize_transform_metadata(
                    result.transform_metadata
                ),
                "overlay_metadata": normalize_overlay_metadata(
                    result.overlay_metadata,
                    end_time - start_time,
                ),
                "transition_metadata": None,
            }
        )
    warnings = list(narration.timing_warnings) + list(alignment.warnings) + rhythm_warnings
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
        timeline_revision=timeline_revision,
        duration=narration.duration,
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
                "from": {"x": 0.45, "y": 0.5},
                "to": {"x": 0.55, "y": 0.5},
            },
        )
    else:
        normalized = NormalizedTransform(
            type=transform_type,
            start_scale=float(metadata.get("start_scale") or 1.0),
            end_scale=float(metadata.get("end_scale") or 1.08),
            focus={"x": 0.5, "y": 0.5},
            crop=metadata.get("crop"),
        )
    return normalized.model_dump(mode="json", by_alias=True, exclude_none=True)


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
    project = get_project(session, timeline.project_id)
    if project is None:
        return False
    try:
        state = require_current_project_visual_plan(session, timeline.project_id)
    except ProjectVisualPlanError:
        return False
    if _stable_hash(state.plan.model_dump(mode="json")) != timeline.visual_plan_revision:
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
        and _file_matches_sha256(
            accepted[entry.beat_id].output_path,
            accepted[entry.beat_id].file_sha256,
        )
        for entry in timeline.entries
    )


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


def build_timeline_quality_report(timeline: ProjectTimeline) -> TimelineQualityReport:
    durations = [item.end_time - item.start_time for item in timeline.entries]
    methods = Counter(item.alignment_method for item in timeline.alignment.beat_timings)
    operations = Counter(item.operation for item in timeline.entries)
    repaired = sum(
        int(match.group(1))
        for warning in timeline.warnings
        if (match := re.search(r"Repaired (\d+)", warning)) is not None
    )
    return TimelineQualityReport(
        narration_duration=timeline.duration,
        visual_beats=len(timeline.alignment.beat_timings),
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
    )


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
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
