"""Tests for FFmpeg scene rendering."""

import subprocess
from pathlib import Path

import pytest

from app.errors import VideoRenderError
from app.media.scene_renderer import render_scene


def test_render_scene_builds_cover_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    output_path = tmp_path / "nested" / "scene.mp4"

    result = render_scene("image.png", "audio.wav", output_path)
    command = captured["command"]

    assert result == str(output_path)
    assert output_path.parent.is_dir()
    assert isinstance(command, list)
    assert command[0] == "ffmpeg"
    video_filter = command[command.index("-vf") + 1]
    assert "scale=1080:1920:force_original_aspect_ratio=increase" in video_filter
    assert "crop=1080:1920" in video_filter
    assert "libx264" in command
    assert "aac" in command
    assert "yuv420p" in command
    assert "-shortest" in command
    assert "shell" not in captured["kwargs"]


def test_render_scene_builds_contain_filter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: (
            commands.append(command)
            or subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        ),
    )

    render_scene(
        "image.png",
        "audio.wav",
        tmp_path / "scene.mp4",
        width=1920,
        height=1080,
        fps=24,
        fit="contain",
    )

    command = commands[0]
    video_filter = command[command.index("-vf") + 1]
    assert "scale=1920:1080:force_original_aspect_ratio=decrease" in video_filter
    assert "pad=1920:1080:(ow-iw)/2:(oh-ih)/2" in video_filter
    assert command[command.index("-r") + 1] == "24"


def test_render_scene_rejects_unknown_fit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported image fit"):
        render_scene(
            "image.png",
            "audio.wav",
            tmp_path / "scene.mp4",
            fit="stretch",  # type: ignore[arg-type]
        )


def test_render_scene_handles_ffmpeg_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="encoder failed",
        ),
    )

    with pytest.raises(VideoRenderError, match="encoder failed"):
        render_scene("image.png", "audio.wav", tmp_path / "scene.mp4")


def test_render_scene_handles_missing_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def missing_ffmpeg(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", missing_ffmpeg)

    with pytest.raises(VideoRenderError, match="not installed"):
        render_scene("image.png", "audio.wav", tmp_path / "scene.mp4")
