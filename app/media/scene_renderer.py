"""Render one video scene with FFmpeg."""

import subprocess
from pathlib import Path
from typing import Literal

from app.errors import VideoRenderError


ImageFit = Literal["cover", "contain"]


def render_scene(
    image_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    fit: ImageFit = "cover",
) -> str:
    """Render a still image and audio track into an MP4 scene."""
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("width, height, and fps must be positive")
    if fit == "cover":
        video_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1"
        )
    elif fit == "contain":
        video_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        )
    else:
        raise ValueError(f"unsupported image fit: {fit}")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-i",
        str(audio_path),
        "-vf",
        video_filter,
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        "-movflags",
        "+faststart",
        str(destination),
    ]
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
        raise VideoRenderError(f"ffmpeg scene render failed: {detail}")
    return str(destination)
