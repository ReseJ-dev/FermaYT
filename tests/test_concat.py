"""Tests for concatenating rendered scenes."""

import subprocess
from pathlib import Path

import pytest

from app.errors import VideoRenderError
from app.media.concat import concat_scene_videos


def test_concat_scene_videos_builds_safe_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scene_paths = [tmp_path / "scene 1.mp4", tmp_path / "scene 2.mp4"]
    for path in scene_paths:
        path.write_bytes(b"video")
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        list_path = Path(command[command.index("-i") + 1])
        captured["list_content"] = list_path.read_text(encoding="utf-8")
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    output_path = tmp_path / "output" / "final.mp4"

    result = concat_scene_videos(scene_paths, output_path)
    command = captured["command"]

    assert result == str(output_path)
    assert isinstance(command, list)
    assert command[:3] == ["ffmpeg", "-y", "-f"]
    assert command[-3:] == ["copy", str(output_path)] or command[-1] == str(output_path)
    assert "file://" in str(captured["list_content"])
    assert "%20" in str(captured["list_content"])
    assert "shell" not in captured["kwargs"]
    assert output_path.parent.is_dir()


def test_concat_scene_videos_supports_one_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.mp4"
    scene_path.write_bytes(b"video")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="",
            stderr="",
        ),
    )

    result = concat_scene_videos([scene_path], tmp_path / "final.mp4")

    assert result == str(tmp_path / "final.mp4")


def test_concat_scene_videos_rejects_empty_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        concat_scene_videos([], tmp_path / "final.mp4")


def test_concat_scene_videos_handles_ffmpeg_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.mp4"
    scene_path.write_bytes(b"video")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="concat failed",
        ),
    )

    with pytest.raises(VideoRenderError, match="concat failed"):
        concat_scene_videos([scene_path], tmp_path / "final.mp4")
