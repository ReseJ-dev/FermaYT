"""Final video rendering orchestration."""

import asyncio
from pathlib import Path

from app.media.concat import concat_scene_videos
from app.media.scene_renderer import ImageFit
from app.models.job import VideoJob
from app.pipeline.render_pipeline import render_job_scenes
from app.storage import ProjectMediaPaths


async def render_final_video(
    job: VideoJob,
    media_paths: ProjectMediaPaths | None = None,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    fit: ImageFit = "cover",
) -> str:
    """Render prepared scenes, concatenate them, and complete the job."""
    paths = media_paths or ProjectMediaPaths(job.id)
    try:
        scene_paths = await render_job_scenes(
            job,
            paths,
            width,
            height,
            fps,
            fit,
            only_missing_or_outdated=True,
        )
        final_path = await asyncio.to_thread(
            concat_scene_videos,
            [Path(path) for path in scene_paths],
            paths.final_video_path,
        )
        job.complete(final_path)
        return final_path
    except Exception as exc:
        error_message = str(exc).strip() or type(exc).__name__
        job.fail(error_message)
        raise
