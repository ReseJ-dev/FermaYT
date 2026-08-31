"""Tests for ffprobe media duration detection."""

import subprocess
from pathlib import Path

import pytest

from app.errors import MediaProbeError
from app.media.probe import get_media_duration


def test_get_media_duration_returns_positive_seconds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "audio.wav"
    media_path.write_bytes(b"audio")

    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        assert command[0] == "ffprobe"
        assert command[-1] == str(media_path)
        assert kwargs["check"] is False
        assert "shell" not in kwargs
        return subprocess.CompletedProcess(command, 0, stdout="2.75\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert get_media_duration(media_path) == 2.75


def test_get_media_duration_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(MediaProbeError, match="does not exist"):
        get_media_duration(tmp_path / "missing.wav")


def test_get_media_duration_handles_missing_ffprobe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "audio.wav"
    media_path.write_bytes(b"audio")

    def fake_run(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(MediaProbeError, match="not installed"):
        get_media_duration(media_path)


def test_get_media_duration_handles_ffprobe_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "audio.wav"
    media_path.write_bytes(b"audio")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            1,
            stdout="",
            stderr="invalid media",
        ),
    )

    with pytest.raises(MediaProbeError, match="invalid media"):
        get_media_duration(media_path)


@pytest.mark.parametrize("raw_duration", ["invalid", "0", "-1", "nan", "inf"])
def test_get_media_duration_rejects_invalid_duration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_duration: str,
) -> None:
    media_path = tmp_path / "audio.wav"
    media_path.write_bytes(b"audio")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=raw_duration,
            stderr="",
        ),
    )

    with pytest.raises(MediaProbeError, match="duration"):
        get_media_duration(media_path)
