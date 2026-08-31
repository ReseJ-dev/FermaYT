"""Pipeline steps for generating scene audio."""

import asyncio
from pathlib import Path

from app.clients.tts_api import QwenTTSApiClient
from app.generators.voice import generate_voice
from app.media.probe import get_media_duration
from app.models.job import VideoJob, VideoJobStatus
from app.models.scene import Scene


async def generate_scene_audio(
    job: VideoJob,
    scene: Scene,
    output_dir: str,
    client: QwenTTSApiClient | None = None,
) -> str:
    """Generate audio for one scene and store its local path."""
    job.set_status(VideoJobStatus.GENERATING_AUDIO)
    output_path = str(
        Path(output_dir) / job.id / "audio" / f"{scene.id}.wav"
    )

    try:
        audio_path = await generate_voice(scene.text, output_path, client)
        scene.audio_path = audio_path
        duration = await asyncio.to_thread(get_media_duration, audio_path)
    except Exception as exc:
        error_message = str(exc).strip() or type(exc).__name__
        job.fail(error_message)
        raise

    scene.duration = duration
    return audio_path


async def generate_job_audio(
    job: VideoJob,
    output_dir: str,
    client: QwenTTSApiClient | None = None,
) -> list[str]:
    """Generate audio for all job scenes in sequence."""
    if not job.scenes:
        raise ValueError("video job must contain at least one scene")

    audio_paths: list[str] = []
    for scene in job.scenes:
        audio_path = await generate_scene_audio(
            job,
            scene,
            output_dir,
            client,
        )
        audio_paths.append(audio_path)
    return audio_paths
