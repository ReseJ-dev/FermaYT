"""Tests for the double-click desktop launcher."""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_shell_launcher_has_valid_syntax() -> None:
    launcher = PROJECT_ROOT / "start_fermayt.sh"

    result = subprocess.run(
        ["bash", "-n", str(launcher)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert launcher.stat().st_mode & 0o111


def test_stop_launcher_has_valid_syntax() -> None:
    launcher = PROJECT_ROOT / "stop_fermayt.sh"

    result = subprocess.run(
        ["bash", "-n", str(launcher)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert launcher.stat().st_mode & 0o111


def test_shell_launcher_handles_existing_server_and_runtime_failure() -> None:
    content = (PROJECT_ROOT / "start_fermayt.sh").read_text(encoding="utf-8")

    assert 'http://127.0.0.1:8000/health' in content
    assert "already running" in content
    assert "server stopped unexpectedly" in content
    assert '"$PYTHON_BIN" "$APP_DIR/run.py"' in content
    assert '"$PYTHON_BIN" "$APP_DIR/run.py" &' not in content
    assert 'exec "$PYTHON_BIN" "$APP_DIR/run.py"' not in content


def test_python_entrypoint_owns_and_cleans_up_pid_file(
    monkeypatch, tmp_path: Path
) -> None:
    import run

    pid_file = tmp_path / "fermayt.pid"
    observed_pid = None

    def fake_uvicorn_run(*args, **kwargs) -> None:
        nonlocal observed_pid
        observed_pid = pid_file.read_text(encoding="utf-8").strip()

    monkeypatch.setattr(run, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run, "PID_FILE", pid_file)
    monkeypatch.setattr(run.uvicorn, "run", fake_uvicorn_run)

    run.main()

    assert observed_pid == str(os.getpid())
    assert not pid_file.exists()


def test_cli_exit_does_not_wait_for_abandoned_worker_thread(tmp_path: Path) -> None:
    pid_file = tmp_path / "fermayt.pid"
    source = f"""
import threading
import time
import run

run.DATA_DIR = run.Path({str(tmp_path)!r})
run.PID_FILE = run.Path({str(pid_file)!r})

def fake_uvicorn_run(*args, **kwargs):
    del args, kwargs
    threading.Thread(target=time.sleep, args=(30,)).start()

run.uvicorn.run = fake_uvicorn_run
run.cli_main()
"""

    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert not pid_file.exists()


def test_desktop_launcher_is_local_and_uses_terminal() -> None:
    desktop_file = PROJECT_ROOT / "FermaYT.desktop"
    content = desktop_file.read_text(encoding="utf-8")

    assert "Exec=/home/grindyun/coding/FermaYT/start_fermayt.sh" in content
    assert "Path=/home/grindyun/coding/FermaYT" in content
    assert "Terminal=true" in content
    assert "0.0.0.0" not in content


def test_native_launcher_is_an_executable_file() -> None:
    launcher = PROJECT_ROOT / "FermaYT"

    assert launcher.read_bytes()[:4] == b"\x7fELF"
    assert launcher.stat().st_mode & 0o111
