"""Media metadata probing with ffprobe."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.errors import MediaProbeError


@dataclass(frozen=True, slots=True)
class MediaProbeResult:
    duration: float
    has_video: bool
    has_audio: bool
    width: int | None
    height: int | None
    fps: float | None


def probe_media(path: str | Path) -> MediaProbeResult:
    """Return validated stream and container metadata."""
    media_path = Path(path)
    if not media_path.is_file() or media_path.stat().st_size <= 0:
        raise MediaProbeError(f"Media file does not exist or is empty: {media_path}")
    command = [
        "ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(media_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise MediaProbeError("ffprobe is not installed") from exc
    if result.returncode != 0:
        detail = result.stderr.strip()[-500:] or "unknown ffprobe error"
        raise MediaProbeError(f"ffprobe failed: {detail}")
    try:
        payload: dict[str, Any] = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise MediaProbeError("ffprobe returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise MediaProbeError("ffprobe returned invalid JSON metadata")
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise MediaProbeError("ffprobe response has no streams")
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    duration = _positive_float((payload.get("format") or {}).get("duration"))
    if duration is None:
        duration = max(
            (_positive_float(item.get("duration")) or 0.0 for item in streams),
            default=0.0,
        )
    if duration <= 0:
        raise MediaProbeError("ffprobe returned a non-positive duration")
    fps = _parse_rate(video.get("avg_frame_rate") or video.get("r_frame_rate")) if video else None
    return MediaProbeResult(
        duration=duration,
        has_video=video is not None,
        has_audio=audio is not None,
        width=int(video["width"]) if video and video.get("width") else None,
        height=int(video["height"]) if video and video.get("height") else None,
        fps=fps,
    )


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
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise MediaProbeError("ffprobe is not installed") from exc
    if result.returncode != 0:
        detail = result.stderr.strip()[:500] or "unknown ffprobe error"
        raise MediaProbeError(f"ffprobe failed: {detail}")
    duration = _positive_float(result.stdout.strip())
    if duration is None:
        raise MediaProbeError("ffprobe returned an invalid or non-positive duration")
    return duration


def _positive_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _parse_rate(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        numerator, denominator = value.split("/", 1)
        rate = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        return None
    return rate if math.isfinite(rate) and rate > 0 else None
