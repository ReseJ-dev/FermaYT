"""Tests for safe project media paths."""

from pathlib import Path

import pytest

from app.storage import ProjectMediaPaths


def test_project_media_paths_create_expected_directories(
    tmp_path: Path,
) -> None:
    paths = ProjectMediaPaths("project-1", tmp_path / "projects")

    assert paths.project_dir == (tmp_path / "projects" / "project-1").resolve()
    assert paths.image_path("scene-1") == paths.images_dir / "scene-1.png"
    assert paths.audio_path("scene-1") == paths.audio_dir / "scene-1.wav"
    assert paths.scene_video_path("scene-1") == (
        paths.scenes_dir / "scene-1.mp4"
    )
    assert paths.final_video_path == paths.output_dir / "final.mp4"
    assert all(
        directory.is_dir()
        for directory in (
            paths.images_dir,
            paths.audio_dir,
            paths.scenes_dir,
            paths.uploads_dir,
            paths.output_dir,
        )
    )


def test_uploaded_image_path_uses_server_generated_name(
    tmp_path: Path,
) -> None:
    paths = ProjectMediaPaths("project-1", tmp_path)

    first = paths.uploaded_image_path(".JPEG")
    second = paths.uploaded_image_path(".JPEG")

    assert first.parent == paths.uploads_dir
    assert first.suffix == ".jpeg"
    assert first != second


@pytest.mark.parametrize(
    "project_id",
    ["../outside", "..", "/absolute", "nested/project", "nested\\project"],
)
def test_project_id_rejects_path_traversal(
    tmp_path: Path,
    project_id: str,
) -> None:
    with pytest.raises(ValueError, match="project_id"):
        ProjectMediaPaths(project_id, tmp_path)


@pytest.mark.parametrize(
    "scene_id",
    ["../outside", "..", "/absolute", "nested/scene", "nested\\scene"],
)
def test_scene_id_rejects_path_traversal(
    tmp_path: Path,
    scene_id: str,
) -> None:
    paths = ProjectMediaPaths("project-1", tmp_path)

    with pytest.raises(ValueError, match="scene_id"):
        paths.image_path(scene_id)


@pytest.mark.parametrize("extension", ["png", ".tar.gz", "../.png", ".p/ng"])
def test_uploaded_image_path_rejects_unsafe_extension(
    tmp_path: Path,
    extension: str,
) -> None:
    paths = ProjectMediaPaths("project-1", tmp_path)

    with pytest.raises(ValueError, match="extension"):
        paths.uploaded_image_path(extension)
