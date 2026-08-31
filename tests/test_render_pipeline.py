"""Tests for scene and final render pipelines."""

import asyncio
from pathlib import Path

import pytest

import app.pipeline.final_render as final_render
import app.pipeline.render_pipeline as render_pipeline
from app.errors import VideoRenderError
from app.models.job import VideoJob, VideoJobStatus
from app.models.scene import Scene
from app.pipeline.final_render import render_final_video
from app.pipeline.render_pipeline import render_job_scenes
from app.storage import ProjectMediaPaths


def make_renderable_job(tmp_path: Path, scene_count: int = 2) -> VideoJob:
    scenes: list[Scene] = []
    for index in range(1, scene_count + 1):
        image_path = tmp_path / f"source-{index}.png"
        audio_path = tmp_path / f"source-{index}.wav"
        image_path.write_bytes(b"image")
        audio_path.write_bytes(b"audio")
        scenes.append(
            Scene(
                id=f"scene-{index}",
                text=f"Narration {index}",
                image_prompt=f"Image {index}",
                image_path=str(image_path),
                audio_path=str(audio_path),
                duration=2.0,
            )
        )
    return VideoJob(
        id="project-1",
        prompt="Create video",
        text="Narration",
        scenes=scenes,
    )


def test_render_job_scenes_processes_scenes_sequentially(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job = make_renderable_job(tmp_path)
    paths = ProjectMediaPaths("project-1", tmp_path / "projects")
    calls: list[str] = []

    def fake_render(
        image_path: str,
        audio_path: str,
        output_path: Path,
        width: int,
        height: int,
        fps: int,
        fit: str,
    ) -> str:
        calls.append(Path(output_path).stem)
        output_path.write_bytes(b"video")
        return str(output_path)

    monkeypatch.setattr(render_pipeline, "render_scene", fake_render)

    result = asyncio.run(render_job_scenes(job, paths))
    expected = [
        str(paths.scene_video_path("scene-1")),
        str(paths.scene_video_path("scene-2")),
    ]

    assert calls == ["scene-1", "scene-2"]
    assert result == expected
    assert [scene.video_path for scene in job.scenes] == expected
    assert job.status is VideoJobStatus.RENDERING


def test_render_job_scenes_fails_job_for_missing_asset(tmp_path: Path) -> None:
    job = make_renderable_job(tmp_path, scene_count=1)
    job.scenes[0].image_path = None
    paths = ProjectMediaPaths("project-1", tmp_path / "projects")

    with pytest.raises(ValueError, match="image_path"):
        asyncio.run(render_job_scenes(job, paths))

    assert job.status is VideoJobStatus.FAILED
    assert "image_path" in (job.error or "")


def test_render_final_video_skips_current_scenes_and_completes_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job = make_renderable_job(tmp_path, scene_count=1)
    paths = ProjectMediaPaths("project-1", tmp_path / "projects")
    render_calls: list[Path] = []

    def fake_render(
        image_path: str,
        audio_path: str,
        output_path: Path,
        width: int,
        height: int,
        fps: int,
        fit: str,
    ) -> str:
        render_calls.append(output_path)
        output_path.write_bytes(b"video")
        return str(output_path)

    def fake_concat(scene_paths: list[Path], output_path: Path) -> str:
        output_path.write_bytes(b"final")
        return str(output_path)

    monkeypatch.setattr(render_pipeline, "render_scene", fake_render)
    monkeypatch.setattr(final_render, "concat_scene_videos", fake_concat)

    first_result = asyncio.run(render_final_video(job, paths))
    second_result = asyncio.run(render_final_video(job, paths))

    assert first_result == str(paths.final_video_path)
    assert second_result == str(paths.final_video_path)
    assert render_calls == [paths.scene_video_path("scene-1")]
    assert job.status is VideoJobStatus.COMPLETED
    assert job.final_video_path == str(paths.final_video_path)


def test_render_final_video_marks_job_failed_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job = make_renderable_job(tmp_path, scene_count=1)
    paths = ProjectMediaPaths("project-1", tmp_path / "projects")

    async def fake_render_scenes(*args: object, **kwargs: object) -> list[str]:
        video_path = paths.scene_video_path("scene-1")
        video_path.write_bytes(b"video")
        return [str(video_path)]

    def failing_concat(scene_paths: list[Path], output_path: Path) -> str:
        raise VideoRenderError("Could not concatenate scenes")

    monkeypatch.setattr(final_render, "render_job_scenes", fake_render_scenes)
    monkeypatch.setattr(final_render, "concat_scene_videos", failing_concat)

    with pytest.raises(VideoRenderError, match="Could not concatenate"):
        asyncio.run(render_final_video(job, paths))

    assert job.status is VideoJobStatus.FAILED
    assert job.error == "Could not concatenate scenes"
