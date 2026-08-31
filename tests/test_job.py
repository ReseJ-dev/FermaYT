"""Tests for the VideoJob model."""

import json
from datetime import UTC
from uuid import UUID

from app.models.job import VideoJob, VideoJobStatus
from app.models.scene import Scene


def make_job() -> VideoJob:
    return VideoJob(prompt="Create a nature video", text="A forest story")


def test_create_video_job_with_defaults() -> None:
    job = make_job()

    assert str(UUID(job.id)) == job.id
    assert job.status is VideoJobStatus.CREATED
    assert job.scenes == []
    assert job.final_video_path is None
    assert job.error is None
    assert job.created_at.tzinfo is UTC
    assert job.updated_at.tzinfo is UTC


def test_scenes_are_independent_between_jobs() -> None:
    first_job = make_job()
    second_job = make_job()

    first_job.add_scene(Scene(text="First", image_prompt="First image"))

    assert len(first_job.scenes) == 1
    assert second_job.scenes == []


def test_add_scene_updates_job() -> None:
    job = make_job()
    scene = Scene(text="First", image_prompt="First image")
    previous_updated_at = job.updated_at

    job.add_scene(scene)

    assert job.scenes == [scene]
    assert job.updated_at >= previous_updated_at


def test_set_status_updates_job() -> None:
    job = make_job()
    previous_updated_at = job.updated_at

    job.set_status(VideoJobStatus.GENERATING_IMAGE)

    assert job.status is VideoJobStatus.GENERATING_IMAGE
    assert job.updated_at >= previous_updated_at


def test_fail_updates_job() -> None:
    job = make_job()
    previous_updated_at = job.updated_at

    job.fail("Image generation failed")

    assert job.status is VideoJobStatus.FAILED
    assert job.error == "Image generation failed"
    assert job.updated_at >= previous_updated_at


def test_complete_updates_job() -> None:
    job = make_job()
    job.fail("Temporary error")
    previous_updated_at = job.updated_at

    job.complete("videos/final.mp4")

    assert job.status is VideoJobStatus.COMPLETED
    assert job.final_video_path == "videos/final.mp4"
    assert job.error is None
    assert job.updated_at >= previous_updated_at


def test_json_serialization() -> None:
    job = make_job()
    job.add_scene(
        Scene(text="First", image_prompt="First image", duration=3.0)
    )

    payload = json.loads(job.model_dump_json())

    assert payload["status"] == "created"
    assert payload["scenes"][0]["duration"] == 3.0
    assert payload["created_at"].endswith("Z")
