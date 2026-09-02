"""Deterministic FFmpeg execution primitives for persisted timeline entries."""

from __future__ import annotations

import html
import math
import subprocess
from pathlib import Path

from app.errors import VideoRenderError
from app.models.render import OverlayStyleConfig, ProjectRenderConfig, RenderImageFit
from app.models.timeline import NormalizedOverlay, NormalizedTransform


def render_timeline_entry(
    image_path: str | Path,
    output_path: str | Path,
    *,
    frame_count: int,
    config: ProjectRenderConfig,
    transform: NormalizedTransform | None = None,
    overlay: NormalizedOverlay | None = None,
    overlay_style: OverlayStyleConfig | None = None,
) -> str:
    """Render one exact-frame-count, silent, codec-compatible timeline segment."""
    source = Path(image_path)
    if not source.is_file():
        raise VideoRenderError(f"Timeline image does not exist: {source}")
    if frame_count < 1:
        raise VideoRenderError("Timeline entry must contain at least one frame")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    filter_graph = _entry_filter(config, frame_count, transform)
    command = [
        "ffmpeg", "-y", "-loop", "1", "-framerate", str(config.fps),
        "-i", str(source),
    ]
    overlay_path: Path | None = None
    if overlay is not None:
        overlay_path = destination.with_suffix(".overlay.svg")
        overlay_path.write_text(
            build_overlay_svg(overlay, config, overlay_style or OverlayStyleConfig()),
            encoding="utf-8",
        )
        command.extend(["-loop", "1", "-framerate", str(config.fps), "-i", str(overlay_path)])
        filter_graph += (
            f"[base];[1:v]format=rgba[annotation];[base][annotation]overlay=0:0:"
            f"enable='gte(t,{overlay.appear_offset:.6f})'[out]"
        )
        map_label = "[out]"
    else:
        filter_graph += "[out]"
        map_label = "[out]"
    command.extend(
        [
            "-filter_complex", filter_graph,
            "-map", map_label,
            "-frames:v", str(frame_count),
            "-r", str(config.fps),
            "-an", "-c:v", config.video_codec,
            "-pix_fmt", config.pixel_format,
            "-movflags", "+faststart",
            str(destination),
        ]
    )
    try:
        _run_ffmpeg(command, "timeline entry render")
    finally:
        if overlay_path is not None:
            overlay_path.unlink(missing_ok=True)
    return str(destination)


def mux_narration(
    video_path: str | Path,
    narration_path: str | Path,
    output_path: str | Path,
    *,
    frame_count: int,
    config: ProjectRenderConfig,
) -> str:
    """Mux the canonical narration once, without altering timeline video frames."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    duration = frame_count / config.fps
    command = [
        "ffmpeg", "-y", "-i", str(video_path), "-i", str(narration_path),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
        "-c:a", config.audio_codec, "-t", f"{duration:.9f}",
        "-movflags", "+faststart", str(destination),
    ]
    _run_ffmpeg(command, "narration mux")
    return str(destination)


def build_overlay_svg(
    overlay: NormalizedOverlay,
    config: ProjectRenderConfig,
    style: OverlayStyleConfig,
) -> str:
    """Build a simple centralized programmatic annotation at normalized coordinates."""
    width, height = config.width, config.height
    margin = min(width, height) * style.safe_margin
    start = _point(overlay.start, width, height, margin, (0.3, 0.35))
    end = _point(overlay.end, width, height, margin, (0.7, 0.65))
    center = _point(overlay.center, width, height, margin, (0.5, 0.5))
    stroke = style.stroke_color
    common = f'stroke="{stroke}" stroke-width="{style.stroke_width}" fill="none" stroke-linecap="round"'
    kind = overlay.type.value
    if kind in {"ARROW", "ROUTE"}:
        dash = ' stroke-dasharray="18 12"' if kind == "ROUTE" else ""
        if kind == "ROUTE" and overlay.points:
            coordinates = [
                _point(point, width, height, margin, (0.5, 0.5))
                for point in overlay.points
            ]
            path_data = " ".join(
                f"{'M' if index == 0 else 'L'}{x},{y}"
                for index, (x, y) in enumerate(coordinates)
            )
            shape = f'<path d="{path_data}" {common}{dash} marker-end="url(#arrow)"/>'
        else:
            shape = f'<line x1="{start[0]}" y1="{start[1]}" x2="{end[0]}" y2="{end[1]}" {common}{dash} marker-end="url(#arrow)"/>'
    elif kind == "X_MARK":
        x, y = center
        size = min(width, height) * 0.08
        shape = f'<path d="M{x-size},{y-size} L{x+size},{y+size} M{x+size},{y-size} L{x-size},{y+size}" {common}/>'
    elif kind == "CIRCLE":
        shape = f'<ellipse cx="{center[0]}" cy="{center[1]}" rx="{width*0.12}" ry="{height*0.1}" {common}/>'
    elif kind == "HIGHLIGHT":
        shape = f'<rect x="{center[0]-width*0.13}" y="{center[1]-height*0.09}" width="{width*0.26}" height="{height*0.18}" rx="12" fill="{style.fill_color}" fill-opacity="{style.opacity}" stroke="{stroke}" stroke-width="{style.stroke_width}"/>'
    else:
        label = html.escape(overlay.semantic_anchor[:60])
        x, y = center
        box_width = min(width * 0.64, max(width * 0.18, len(label) * style.font_size * 0.58))
        shape = (
            f'<rect x="{x-box_width/2}" y="{y-style.font_size}" width="{box_width}" height="{style.font_size*1.6}" rx="10" fill="#111111" fill-opacity="0.78"/>'
            f'<text x="{x}" y="{y+style.font_size*0.18}" text-anchor="middle" fill="{style.text_color}" font-family="sans-serif" font-size="{style.font_size}">{label}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="{stroke}"/></marker></defs>'
        f'{shape}</svg>'
    )


def _entry_filter(
    config: ProjectRenderConfig,
    frame_count: int,
    transform: NormalizedTransform | None,
) -> str:
    width, height, fps = config.width, config.height, config.fps
    if config.image_fit_mode is RenderImageFit.COVER:
        fit = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
    else:
        color = config.background_color.lstrip("#")
        fit = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x{color}"
    chain = f"[0:v]{fit},setsar=1"
    if transform is not None:
        denominator = max(frame_count - 1, 1)
        if transform.type.value == "CROP" and transform.crop is not None:
            left, top, right, bottom = transform.crop
            chain += f",crop=iw*{right-left:.8f}:ih*{bottom-top:.8f}:iw*{left:.8f}:ih*{top:.8f},scale={width}:{height}"
        elif transform.type.value == "PAN":
            assert transform.from_point is not None and transform.to is not None
            scale = min(config.max_motion_scale, 1.08)
            fw, fh = math.ceil(width * scale), math.ceil(height * scale)
            dx = transform.to.x - transform.from_point.x
            dy = transform.to.y - transform.from_point.y
            x = f"(iw-{width})*({transform.from_point.x:.8f}+({dx:.8f})*n/{denominator})"
            y = f"(ih-{height})*({transform.from_point.y:.8f}+({dy:.8f})*n/{denominator})"
            chain += f",scale={fw}:{fh},crop={width}:{height}:x='{x}':y='{y}'"
        else:
            start = min(transform.start_scale or 1.0, config.max_motion_scale)
            end = min(transform.end_scale or start, config.max_motion_scale)
            focus = transform.focus
            fx, fy = (focus.x, focus.y) if focus else (0.5, 0.5)
            zoom = f"{start:.8f}+({end-start:.8f})*on/{denominator}"
            x = f"(iw-iw/zoom)*{fx:.8f}"
            y = f"(ih-ih/zoom)*{fy:.8f}"
            chain += f",zoompan=z='{zoom}':x='{x}':y='{y}':d=1:s={width}x{height}:fps={fps}"
    return f"{chain},trim=end_frame={frame_count},setpts=N/({fps}*TB)"


def _point(point: object, width: int, height: int, margin: float, fallback: tuple[float, float]) -> tuple[float, float]:
    x = getattr(point, "x", fallback[0]) if point is not None else fallback[0]
    y = getattr(point, "y", fallback[1]) if point is not None else fallback[1]
    return (max(margin, min(width - margin, x * width)), max(margin, min(height - margin, y * height)))


def _run_ffmpeg(command: list[str], stage: str) -> None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise VideoRenderError("ffmpeg is not installed") from exc
    if result.returncode != 0:
        detail = result.stderr.strip()[-1000:] or "unknown FFmpeg error"
        raise VideoRenderError(f"FFmpeg {stage} failed: {detail}")
