"""Pipeline for rendering prepared scenes."""

import asyncio
from pathlib import Path
from collections.abc import Sequence

from app.media.scene_renderer import ImageFit, render_scene
from app.models.job import VideoJob, VideoJobStatus
from app.models.scene import Scene
from app.storage import ProjectMediaPaths


def validate_scenes_for_render(scenes: Sequence[Scene]) -> None:
    """Ensure every scene has usable image, audio, and duration metadata."""
    if not scenes:
        raise ValueError("video job must contain at least one scene")
    for scene in scenes:
        missing: list[str] = []
        if not scene.image_path:
            missing.append("image_path")
        elif not Path(scene.image_path).is_file():
            missing.append("image file")
        if not scene.audio_path:
            missing.append("audio_path")
        elif not Path(scene.audio_path).is_file():
            missing.append("audio file")
        if scene.duration is None or scene.duration <= 0:
            missing.append("duration")
        if missing:
            raise ValueError(
                f"Scene {scene.id} is missing: {', '.join(missing)}"
            )


async def render_job_scenes(
    job: VideoJob,
    media_paths: ProjectMediaPaths | None = None,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    fit: ImageFit = "cover",
    only_missing_or_outdated: bool = False,
) -> list[str]:
    """Render job scenes sequentially from their prepared assets."""
    job.set_status(VideoJobStatus.RENDERING)
    paths = media_paths or ProjectMediaPaths(job.id)
    try:
        validate_scenes_for_render(job.scenes)
        rendered_paths: list[str] = []
        for scene in job.scenes:
            destination = paths.scene_video_path(scene.id)
            if only_missing_or_outdated and _is_current(scene, destination):
                rendered_paths.append(str(destination))
                continue
            video_path = await asyncio.to_thread(
                render_scene,
                scene.image_path,
                scene.audio_path,
                destination,
                width,
                height,
                fps,
                fit,
            )
            scene.video_path = video_path
            rendered_paths.append(video_path)
        return rendered_paths
    except Exception as exc:
        error_message = str(exc).strip() or type(exc).__name__
        job.fail(error_message)
        raise


def _is_current(scene: Scene, destination: Path) -> bool:
    if not scene.video_path or not destination.is_file():
        return False
    if Path(scene.video_path).resolve() != destination.resolve():
        return False
    source_mtime = max(
        Path(scene.image_path).stat().st_mtime,  # type: ignore[arg-type]
        Path(scene.audio_path).stat().st_mtime,  # type: ignore[arg-type]
    )
    return destination.stat().st_mtime >= source_mtime
