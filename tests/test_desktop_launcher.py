"""Tests for the double-click desktop launcher."""

import subprocess
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
