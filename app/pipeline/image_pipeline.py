"""Pipeline step for generating a scene image."""

from pathlib import Path

from app.generators.image import generate_image
from app.models.job import VideoJob, VideoJobStatus
from app.models.scene import Scene


async def generate_scene_image(
    job: VideoJob,
    scene: Scene,
    output_dir: str,
) -> str:
    """Generate an image for one scene and store its local path."""
    job.set_status(VideoJobStatus.GENERATING_IMAGE)
    output_path = str(Path(output_dir) / job.id / f"{scene.id}.png")

    try:
        image_path = await generate_image(scene.image_prompt, output_path)
    except Exception as exc:
        job.fail(str(exc))
        raise

    scene.image_path = image_path
    return image_path


async def generate_job_images(
    job: VideoJob,
    output_dir: str,
) -> list[str]:
    """Generate images for all job scenes in sequence."""
    if not job.scenes:
        raise ValueError("video job must contain at least one scene")

    image_paths: list[str] = []
    for scene in job.scenes:
        image_path = await generate_scene_image(job, scene, output_dir)
        image_paths.append(image_path)

    return image_paths
