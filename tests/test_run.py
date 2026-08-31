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
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_run(application: str, **kwargs: object) -> None:
        calls.append((application, kwargs))

    monkeypatch.setattr(run, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(run.uvicorn, "run", fake_run)

    run.main()

    assert (tmp_path / "data" / "projects").is_dir()
    assert calls == [
        (
            "app.main:app",
            {"host": "127.0.0.1", "port": 8000, "reload": False},
        )
    ]
