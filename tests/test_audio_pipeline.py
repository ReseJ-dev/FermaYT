"""Tests for scene audio pipeline orchestration."""

import asyncio
from pathlib import Path

import pytest

import app.pipeline.audio_pipeline as audio_pipeline
from app.errors import TTSGenerationError
from app.models.job import VideoJob, VideoJobStatus
from app.models.scene import Scene
from app.pipeline.audio_pipeline import (
    generate_job_audio,
    generate_scene_audio,
)


def make_job(scene_count: int = 1) -> VideoJob:
    scenes = [
        Scene(
            id=f"scene-{index}",
            text=f"Narration {index}",
            image_prompt=f"Image {index}",
        )
        for index in range(1, scene_count + 1)
    ]
    return VideoJob(
        id="job-1",
        prompt="Create a video",
        text="Video narration",
        scenes=scenes,
    )


def test_generate_scene_audio_updates_scene(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job = make_job()
    scene = job.scenes[0]
    generation_call: tuple[str, str, object] | None = None
    fake_client = object()
    probe_calls: list[str] = []

    async def fake_generate_voice(
        text: str,
        output_path: str,
        client: object,
    ) -> str:
        nonlocal generation_call
        generation_call = (text, output_path, client)
        return output_path

    monkeypatch.setattr(audio_pipeline, "generate_voice", fake_generate_voice)
    monkeypatch.setattr(
        audio_pipeline,
        "get_media_duration",
        lambda path: probe_calls.append(path) or 2.5,
    )

    result = asyncio.run(
        generate_scene_audio(job, scene, str(tmp_path), fake_client)  # type: ignore[arg-type]
    )
    expected_path = str(tmp_path / "job-1" / "audio" / "scene-1.wav")

    assert job.status is VideoJobStatus.GENERATING_AUDIO
    assert generation_call == ("Narration 1", expected_path, fake_client)
    assert probe_calls == [expected_path]
    assert scene.audio_path == expected_path
    assert scene.duration == 2.5
    assert result == expected_path


def test_generate_scene_audio_marks_job_failed_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job = make_job()
    scene = job.scenes[0]
    error = TTSGenerationError("TTS provider failed")

    async def failing_generate_voice(
        text: str,
        output_path: str,
        client: object,
    ) -> str:
        raise error

    monkeypatch.setattr(
        audio_pipeline,
        "generate_voice",
        failing_generate_voice,
    )

    with pytest.raises(TTSGenerationError) as exc_info:
        asyncio.run(generate_scene_audio(job, scene, str(tmp_path)))

    assert exc_info.value is error
    assert job.status is VideoJobStatus.FAILED
    assert job.error == "TTS provider failed"
    assert scene.audio_path is None


def test_generate_job_audio_processes_three_scenes_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job = make_job(scene_count=3)
    calls: list[str] = []

    async def fake_generate_voice(
        text: str,
        output_path: str,
        client: object,
    ) -> str:
        calls.append(text)
        await asyncio.sleep(0)
        return output_path

    monkeypatch.setattr(audio_pipeline, "generate_voice", fake_generate_voice)
    monkeypatch.setattr(audio_pipeline, "get_media_duration", lambda path: 3.0)

    result = asyncio.run(generate_job_audio(job, str(tmp_path)))
    expected_paths = [
        str(tmp_path / "job-1" / "audio" / f"scene-{index}.wav")
        for index in range(1, 4)
    ]

    assert calls == ["Narration 1", "Narration 2", "Narration 3"]
    assert result == expected_paths
    assert [scene.audio_path for scene in job.scenes] == expected_paths
    assert [scene.duration for scene in job.scenes] == [3.0, 3.0, 3.0]


def test_generate_scene_audio_fails_job_when_duration_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job = make_job()
    scene = job.scenes[0]

    async def fake_generate_voice(
        text: str,
        output_path: str,
        client: object,
    ) -> str:
        return output_path

    def failing_probe(path: str) -> float:
        raise RuntimeError("Could not probe audio")

    monkeypatch.setattr(audio_pipeline, "generate_voice", fake_generate_voice)
    monkeypatch.setattr(audio_pipeline, "get_media_duration", failing_probe)

    with pytest.raises(RuntimeError, match="Could not probe audio"):
        asyncio.run(generate_scene_audio(job, scene, str(tmp_path)))

    assert job.status is VideoJobStatus.FAILED
    assert job.error == "Could not probe audio"
    assert scene.audio_path is not None
    assert scene.duration is None


def test_generate_job_audio_rejects_empty_scenes(tmp_path: Path) -> None:
    job = VideoJob(prompt="Create a video", text="Video narration")

    with pytest.raises(ValueError, match="at least one scene"):
        asyncio.run(generate_job_audio(job, str(tmp_path)))
