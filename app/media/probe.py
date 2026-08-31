"""Media metadata probing with ffprobe."""

import math
import subprocess
from pathlib import Path

from app.errors import MediaProbeError


def get_media_duration(path: str | Path) -> float:
    """Return a positive media duration in seconds."""
    media_path = Path(path)
    if not media_path.is_file():
        raise MediaProbeError(f"Media file does not exist: {media_path}")

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MediaProbeError("ffprobe is not installed") from exc

    if result.returncode != 0:
        detail = result.stderr.strip()[:500] or "unknown ffprobe error"
        raise MediaProbeError(f"ffprobe failed: {detail}")

    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise MediaProbeError("ffprobe returned an invalid duration") from exc

    if not math.isfinite(duration) or duration <= 0:
        raise MediaProbeError("ffprobe returned a non-positive duration")
    return duration
