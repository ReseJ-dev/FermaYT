"""Tests for the local application launcher."""

from pathlib import Path

import pytest

import run


def test_prepare_data_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(run, "DATA_DIR", tmp_path / "data")

    run.prepare_data_directories()

    assert (tmp_path / "data" / "projects").is_dir()


def test_main_starts_local_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fake_run_server() -> None:
        calls.append("run_server")

    monkeypatch.setattr(run, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(run, "run_server", fake_run_server)

    run.main()

    assert (tmp_path / "data" / "projects").is_dir()
    assert calls == ["run_server"]
