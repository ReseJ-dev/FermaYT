"""Tests for scene image pipeline orchestration."""

import asyncio
from pathlib import Path

import pytest

import app.pipeline.image_pipeline as image_pipeline
from app.clients.image_api import ImageGenerationError
from app.models.job import VideoJob, VideoJobStatus
from app.models.scene import Scene
from app.pipeline.image_pipeline import generate_job_images, generate_scene_image


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


def test_generate_job_images_processes_scenes_sequentially(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scenes = [
        Scene(
            id=f"scene-{index}",
            text=f"Narration {index}",
            image_prompt=f"Prompt {index}",
        )
        for index in range(1, 4)
    ]
    job = VideoJob(
        id="job-1",
        prompt="Create a video",
        text="Video narration",
        scenes=scenes,
    )
    calls: list[tuple[str, str]] = []

    async def fake_generate_image(prompt: str, output_path: str) -> str:
        calls.append((prompt, output_path))
        await asyncio.sleep(0)
        return output_path

    monkeypatch.setattr(
        image_pipeline,
        "generate_image",
        fake_generate_image,
    )

    result = asyncio.run(generate_job_images(job, str(tmp_path)))
    expected_paths = [
        str(tmp_path / "job-1" / f"scene-{index}.png")
        for index in range(1, 4)
    ]

    assert calls == list(
        zip(
            ["Prompt 1", "Prompt 2", "Prompt 3"],
            expected_paths,
            strict=True,
        )
    )
    assert result == expected_paths
    assert [scene.image_path for scene in scenes] == expected_paths
    assert len(set(result)) == len(scenes)


def test_generate_job_images_rejects_empty_scenes(tmp_path: Path) -> None:
    job = VideoJob(prompt="Create a video", text="Video narration")

    with pytest.raises(
        ValueError,
        match="video job must contain at least one scene",
    ):
        asyncio.run(generate_job_images(job, str(tmp_path)))
