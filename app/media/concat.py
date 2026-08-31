"""Concatenate rendered scenes with FFmpeg."""

import subprocess
import tempfile
from pathlib import Path

from app.errors import VideoRenderError


def concat_scene_videos(
    paths: list[Path],
    output_path: Path,
) -> str:
    """Concatenate compatible scene videos without invoking a shell."""
    if not paths:
        raise ValueError("at least one scene video is required")
    missing_paths = [path for path in paths if not path.is_file()]
    if missing_paths:
        raise VideoRenderError(f"Scene video does not exist: {missing_paths[0]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".txt",
        prefix="concat-",
        dir=output_path.parent,
        delete=False,
    ) as file_list:
        for path in paths:
            file_list.write(f"file '{path.resolve().as_uri()}'\n")
        list_path = Path(file_list.name)

    command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        str(output_path),
    ]
    try:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise VideoRenderError("ffmpeg is not installed") from exc
        if result.returncode != 0:
            detail = result.stderr.strip()[:500] or "unknown ffmpeg error"
            raise VideoRenderError(f"ffmpeg concat failed: {detail}")
    finally:
        list_path.unlink(missing_ok=True)

    return str(output_path)
