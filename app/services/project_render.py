"""Deterministically execute a persisted ProjectTimeline into a final MP4."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from app.errors import MediaProbeError, ProjectTimelineRenderError
from app.media.concat import concat_scene_videos
from app.media.probe import MediaProbeResult, probe_media
from app.media.timeline_renderer import mux_narration, render_timeline_entry
from app.models.render import ProjectRenderConfig, RenderTransition
from app.models.timeline import NormalizedOverlay, NormalizedTransform
from app.persistence import (
    BeatVisualResult,
    ProjectTimeline,
    ProjectVideoRender,
    TimelineEntry,
)
from app.repositories import (
    create_pending_video_render,
    fail_video_render,
    get_successful_video_render,
    succeed_video_render,
)
from app.services.timeline import timeline_is_current
from app.storage import ProjectMediaPaths

RENDERER_VERSION = "project_timeline_renderer_v1"


def render_project_video(
    session: Session,
    project_id: str,
    timeline_id: str,
    *,
    config: ProjectRenderConfig | None = None,
    projects_root: str | Path = "data/projects",
) -> ProjectVideoRender:
    """Render or reuse one validated timeline/config/source revision."""
    timeline = session.get(ProjectTimeline, timeline_id)
    if timeline is None or timeline.project_id != project_id:
        raise ValueError("Project timeline was not found")
    render_config = config or ProjectRenderConfig()
    revision = _render_revision(timeline, render_config)
    reusable = get_successful_video_render(session, revision)
    if reusable is not None and _successful_output_is_valid(reusable, render_config, timeline.duration):
        try:
            _validate_render_inputs(session, timeline, render_config)
        except Exception as exc:
            record = _create_render_attempt(
                session, timeline, render_config, revision
            )
            safe = _safe_error(exc, None, "VALIDATION")
            fail_video_render(
                session,
                record,
                safe_error=safe,
                failed_stage="VALIDATION",
                diagnostics={"exception_type": type(exc).__name__},
            )
            raise ProjectTimelineRenderError(safe) from exc
        return reusable
    record = _create_render_attempt(session, timeline, render_config, revision)
    current_entry: TimelineEntry | None = None
    stage = "VALIDATION"
    try:
        _validate_render_inputs(session, timeline, render_config)
        frame_ranges = _allocate_frames(timeline, render_config.fps)
        paths = ProjectMediaPaths(project_id, projects_root)
        cache_dir = paths.timeline_render_cache_dir(revision)
        clip_paths: list[Path] = []
        previous_asset: str | None = None
        carried_transform: NormalizedTransform | None = None
        reused_clips = 0
        for entry, (_, _, frame_count) in zip(timeline.entries, frame_ranges, strict=True):
            current_entry = entry
            stage = "ENTRY_RENDER"
            own_transform = _parse_transform(entry)
            if entry.asset_path != previous_asset:
                carried_transform = None
            effective_transform = own_transform or carried_transform
            overlay = _parse_overlay(entry)
            clip_path = cache_dir / f"{entry.position:04d}-{entry.id}-{frame_count}f.mp4"
            if _clip_is_valid(clip_path, frame_count, render_config):
                reused_clips += 1
            else:
                render_timeline_entry(
                    entry.asset_path,
                    clip_path,
                    frame_count=frame_count,
                    config=render_config,
                    transform=effective_transform,
                    overlay=overlay,
                )
                _require_clip(clip_path, frame_count, render_config)
            clip_paths.append(clip_path)
            previous_asset = entry.asset_path
            if own_transform is not None:
                carried_transform = _final_transform_state(own_transform)

        stage = "CONCAT"
        silent_path = cache_dir / "timeline-video.mp4"
        concat_scene_videos(clip_paths, silent_path)
        total_frames = frame_ranges[-1][1]
        stage = "AUDIO_MUX"
        destination = paths.timeline_render_path(revision)
        partial = destination.with_name(f".{destination.name}.{uuid4()}.partial.mp4")
        try:
            mux_narration(
                silent_path,
                timeline.narration_asset.audio_path,
                partial,
                frame_count=total_frames,
                config=render_config,
            )
            stage = "FINAL_VALIDATION"
            metadata = _require_final_output(partial, total_frames, render_config)
            os.replace(partial, destination)
        finally:
            partial.unlink(missing_ok=True)
        digest = _sha256(destination)
        return succeed_video_render(
            session,
            record,
            output_path=str(destination),
            output_sha256=digest,
            duration=metadata.duration,
            width=metadata.width or 0,
            height=metadata.height or 0,
            fps=metadata.fps or 0.0,
            diagnostics={
                "entry_count": len(timeline.entries),
                "total_frames": total_frames,
                "reused_intermediate_clips": reused_clips,
                "manifest": build_render_manifest(timeline, render_config, frame_ranges),
            },
        )
    except Exception as exc:
        safe = _safe_error(exc, current_entry, stage)
        fail_video_render(
            session,
            record,
            safe_error=safe,
            failed_stage=stage,
            failed_entry_id=current_entry.id if current_entry else None,
            failed_beat_id=current_entry.beat_id if current_entry else None,
            diagnostics={"exception_type": type(exc).__name__},
        )
        raise ProjectTimelineRenderError(safe) from exc


def build_render_manifest(
    timeline: ProjectTimeline,
    config: ProjectRenderConfig,
    frame_ranges: list[tuple[int, int, int]] | None = None,
) -> dict[str, Any]:
    ranges = frame_ranges or _allocate_frames(timeline, config.fps)
    operation_counts = Counter(entry.operation for entry in timeline.entries)
    static_count = sum(
        entry.transform_metadata is None and entry.overlay_metadata is None
        for entry in timeline.entries
    )
    return {
        "renderer_version": RENDERER_VERSION,
        "timeline_id": timeline.id,
        "timeline_revision": timeline.timeline_revision,
        "narration_asset_id": timeline.narration_asset_id,
        "narration_path": timeline.narration_asset.audio_path,
        "config": config.model_dump(mode="json"),
        "duration": timeline.duration,
        "total_frames": ranges[-1][1],
        "operation_counts": dict(sorted(operation_counts.items())),
        "static_entries": static_count,
        "transform_entries": sum(entry.transform_metadata is not None for entry in timeline.entries),
        "overlay_entries": sum(entry.overlay_metadata is not None for entry in timeline.entries),
        "generated_or_edited_entries": sum(
            entry.operation in {"NEW_IMAGE", "REFERENCE_GENERATION", "EDIT_EXISTING"}
            for entry in timeline.entries
        ),
        "transition_counts": {"CUT": max(0, len(timeline.entries) - 1)},
        "entries": [
            {
                "position": entry.position,
                "beat_id": entry.beat_id,
                "operation": entry.operation,
                "asset_path": entry.asset_path,
                "frames": ranges[index][2],
                "frame_start": ranges[index][0],
                "frame_end": ranges[index][1],
                "transform": entry.transform_metadata,
                "overlay": entry.overlay_metadata,
            }
            for index, entry in enumerate(timeline.entries)
        ],
    }


def format_render_manifest(manifest: dict[str, Any], output_path: str | None = None) -> str:
    """Format a compact human-readable account of deterministic render work."""
    config = manifest["config"]
    transitions = manifest["transition_counts"]
    lines = [
        "PROJECT RENDER",
        f"Duration: {manifest['duration']:.3f} sec",
        f"Resolution: {config['width']}x{config['height']}",
        f"FPS: {config['fps']}",
        f"Timeline entries: {len(manifest['entries'])}",
        f"Static: {manifest['static_entries']}",
        f"Transforms: {manifest['transform_entries']}",
        f"Overlays: {manifest['overlay_entries']}",
        f"Generated/edit visual entries: {manifest['generated_or_edited_entries']}",
        f"Transitions: {transitions.get('CUT', 0)} CUT, {transitions.get('CROSSFADE', 0)} CROSSFADE",
        f"Audio: {manifest['narration_path']}",
    ]
    if output_path is not None:
        lines.extend((f"Output: {output_path}", "Validation: PASS"))
    return "\n".join(lines)


def _validate_render_inputs(session: Session, timeline: ProjectTimeline, config: ProjectRenderConfig) -> None:
    if config.default_transition is not RenderTransition.CUT:
        raise ValueError("CROSSFADE rendering is not implemented; use CUT")
    if not timeline_is_current(session, timeline):
        raise ValueError("Timeline is stale or a source asset is missing or changed")
    if not timeline.entries:
        raise ValueError("Timeline contains no entries")
    tolerance = 1e-6
    previous_end = 0.0
    for position, entry in enumerate(timeline.entries):
        if entry.position != position:
            raise ValueError(f"Timeline entry order is invalid at beat {entry.beat_id}")
        if abs(entry.start_time - previous_end) > tolerance:
            raise ValueError(f"Timeline has a gap or overlap before beat {entry.beat_id}")
        if entry.end_time <= entry.start_time or entry.end_time > timeline.duration + tolerance:
            raise ValueError(f"Timeline bounds are invalid for beat {entry.beat_id}")
        result = session.get(BeatVisualResult, entry.beat_visual_result_id)
        if result is None or result.generation_status != "SUCCEEDED" or not result.is_accepted:
            raise ValueError(f"Beat {entry.beat_id} has no accepted successful visual result")
        _parse_transform(entry)
        _parse_overlay(entry)
        if entry.transition_metadata not in (None, {}, {"type": "CUT"}):
            raise ValueError(f"Unsupported transition metadata for beat {entry.beat_id}")
        previous_end = entry.end_time
    if abs(previous_end - timeline.duration) > tolerance:
        raise ValueError("Timeline does not end at narration duration")


def _create_render_attempt(
    session: Session,
    timeline: ProjectTimeline,
    config: ProjectRenderConfig,
    revision: str,
) -> ProjectVideoRender:
    return create_pending_video_render(
        session,
        project_id=timeline.project_id,
        timeline_id=timeline.id,
        timeline_revision=timeline.timeline_revision,
        render_config_version=config.version,
        render_config_snapshot=config.model_dump(mode="json"),
        renderer_version=RENDERER_VERSION,
        render_revision=revision,
    )


def _allocate_frames(timeline: ProjectTimeline, fps: int) -> list[tuple[int, int, int]]:
    total = max(1, round(timeline.duration * fps))
    boundaries = [0]
    for entry in timeline.entries[:-1]:
        boundaries.append(round(entry.end_time * fps))
    boundaries.append(total)
    ranges = []
    for index in range(len(timeline.entries)):
        start, end = boundaries[index], boundaries[index + 1]
        if end <= start:
            raise ValueError(f"Beat {timeline.entries[index].beat_id} is shorter than one output frame")
        ranges.append((start, end, end - start))
    return ranges


def _parse_transform(entry: TimelineEntry) -> NormalizedTransform | None:
    return NormalizedTransform.model_validate(entry.transform_metadata) if entry.transform_metadata else None


def _parse_overlay(entry: TimelineEntry) -> NormalizedOverlay | None:
    overlay = NormalizedOverlay.model_validate(entry.overlay_metadata) if entry.overlay_metadata else None
    if overlay is not None and overlay.appear_offset >= entry.end_time - entry.start_time:
        raise ValueError(f"Overlay appears after beat {entry.beat_id} ends")
    return overlay


def _final_transform_state(transform: NormalizedTransform) -> NormalizedTransform:
    values = transform.model_dump(mode="python", by_alias=True, exclude_none=True)
    if transform.type.value == "PAN" and transform.to is not None:
        values["from"] = transform.to.model_dump()
        values["to"] = transform.to.model_dump()
    elif transform.end_scale is not None:
        values["start_scale"] = transform.end_scale
        values["end_scale"] = transform.end_scale
    return NormalizedTransform.model_validate(values)


def _clip_is_valid(path: Path, frames: int, config: ProjectRenderConfig) -> bool:
    try:
        _require_clip(path, frames, config)
    except (MediaProbeError, OSError, ValueError):
        return False
    return True


def _require_clip(path: Path, frames: int, config: ProjectRenderConfig) -> MediaProbeResult:
    metadata = probe_media(path)
    expected = frames / config.fps
    if metadata.has_audio or not metadata.has_video:
        raise ValueError("Intermediate clip stream layout is invalid")
    if (metadata.width, metadata.height) != (config.width, config.height):
        raise ValueError("Intermediate clip resolution is invalid")
    if metadata.fps is None or abs(metadata.fps - config.fps) > 0.01:
        raise ValueError("Intermediate clip fps is invalid")
    if abs(metadata.duration - expected) > 1 / config.fps + 0.02:
        raise ValueError("Intermediate clip duration is invalid")
    return metadata


def _require_final_output(path: Path, frames: int, config: ProjectRenderConfig) -> MediaProbeResult:
    metadata = probe_media(path)
    expected = frames / config.fps
    if not metadata.has_video or not metadata.has_audio:
        raise ValueError("Final MP4 must contain video and audio streams")
    if (metadata.width, metadata.height) != (config.width, config.height):
        raise ValueError("Final MP4 resolution does not match render config")
    if metadata.fps is None or abs(metadata.fps - config.fps) > 0.01:
        raise ValueError("Final MP4 fps does not match render config")
    if abs(metadata.duration - expected) > 1 / config.fps + 0.05:
        raise ValueError("Final MP4 duration differs from frame-accurate timeline")
    return metadata


def _successful_output_is_valid(record: ProjectVideoRender, config: ProjectRenderConfig, duration: float) -> bool:
    if record.output_path is None or record.output_sha256 is None:
        return False
    path = Path(record.output_path)
    if not path.is_file() or _sha256(path) != record.output_sha256:
        return False
    try:
        _require_final_output(path, round(duration * config.fps), config)
    except (MediaProbeError, OSError, ValueError):
        return False
    return True


def _render_revision(timeline: ProjectTimeline, config: ProjectRenderConfig) -> str:
    payload = {
        "timeline_revision": timeline.timeline_revision,
        "narration_sha256": timeline.narration_asset.audio_sha256,
        "visual_results": [entry.beat_visual_result_id for entry in timeline.entries],
        "config": config.model_dump(mode="json"),
        "renderer": RENDERER_VERSION,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_error(exc: Exception, entry: TimelineEntry | None, stage: str) -> str:
    detail = re.sub(r"(?i)(bearer|api[_ -]?key|authorization)[^\s,;]*", r"\1 [redacted]", str(exc))
    prefix = f"Beat {entry.position + 1} ({entry.beat_id}), " if entry else ""
    return f"{prefix}{stage}: {detail}"[:1500]
