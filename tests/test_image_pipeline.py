"""Tests for scene image pipeline orchestration."""

import asyncio
from pathlib import Path

import pytest

import app.pipeline.image_pipeline as image_pipeline
from app.clients.image_api import ImageGenerationError
from app.models.job import VideoJob, VideoJobStatus
from app.models.scene import Scene
from app.pipeline.image_pipeline import generate_scene_image


def make_job_and_scene() -> tuple[VideoJob, Scene]:
    scene = Scene(
        id="scene-1",
        text="Scene narration",
        image_prompt="A cinematic forest",
    )
    job = VideoJob(
        id="job-1",
        prompt="Create a video",
        text="Video narration",
        scenes=[scene],
    )
    return job, scene


def test_generate_scene_image_updates_scene_and_returns_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job, scene = make_job_and_scene()
    generate_call: tuple[str, str] | None = None

    async def fake_generate_image(prompt: str, output_path: str) -> str:
        nonlocal generate_call
        generate_call = (prompt, output_path)
        return output_path

    monkeypatch.setattr(
        image_pipeline,
        "generate_image",
        fake_generate_image,
    )

    result = asyncio.run(generate_scene_image(job, scene, str(tmp_path)))
    expected_path = str(tmp_path / "job-1" / "scene-1.png")

    assert job.status is VideoJobStatus.GENERATING_IMAGE
    assert generate_call == ("A cinematic forest", expected_path)
    assert scene.image_path == expected_path
    assert result == expected_path


def test_generate_scene_image_marks_job_failed_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job, scene = make_job_and_scene()
    error = ImageGenerationError("Image provider failed")

    async def failing_generate_image(prompt: str, output_path: str) -> str:
        raise error

    monkeypatch.setattr(
        image_pipeline,
        "generate_image",
        failing_generate_image,
    )

    with pytest.raises(ImageGenerationError) as exc_info:
        asyncio.run(generate_scene_image(job, scene, str(tmp_path)))

    assert exc_info.value is error
    assert job.status is VideoJobStatus.FAILED
    assert job.error == "Image provider failed"
    assert scene.image_path is None
