"""Real tiny-media tests for deterministic Stage 6 FFmpeg primitives."""

import shutil
import subprocess
from pathlib import Path

import pytest

from app.media import timeline_renderer
from app.media.probe import probe_media
from app.media.timeline_renderer import (
    mux_narration,
    render_timeline_entry,
    render_video_timeline_entry,
)
from app.models.render import ProjectRenderConfig, RenderImageFit
from app.models.timeline import NormalizedOverlay, NormalizedTransform


@pytest.fixture
def source_image(tmp_path: Path) -> Path:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required")
    path = tmp_path / "source.png"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x335577:s=240x180",
            "-frames:v",
            "1",
            str(path),
        ],
        check=True,
    )
    return path


@pytest.mark.parametrize(
    "transform",
    [
        None,
        NormalizedTransform(
            type="ZOOM_IN", start_scale=1.0, end_scale=1.1, focus={"x": 0.5, "y": 0.5}
        ),
        NormalizedTransform(
            type="ZOOM_OUT", start_scale=1.1, end_scale=1.0, focus={"x": 0.5, "y": 0.5}
        ),
        NormalizedTransform(
            type="PAN", **{"from": {"x": 0.2, "y": 0.5}, "to": {"x": 0.8, "y": 0.5}}
        ),
        NormalizedTransform(type="CROP", crop=(0.1, 0.1, 0.9, 0.9)),
        NormalizedTransform(
            type="FOCUS", start_scale=1.0, end_scale=1.08, focus={"x": 0.7, "y": 0.4}
        ),
    ],
    ids=["static", "zoom-in", "zoom-out", "pan", "crop", "focus"],
)
def test_static_and_transform_segments_are_legal_exact_media(
    source_image: Path,
    tmp_path: Path,
    transform: NormalizedTransform | None,
) -> None:
    config = ProjectRenderConfig(
        width=320,
        height=180,
        fps=10,
        image_fit_mode=RenderImageFit.CONTAIN,
    )
    output = tmp_path / f"{transform.type.value if transform else 'static'}.mp4"
    render_timeline_entry(
        source_image, output, frame_count=4, config=config, transform=transform
    )
    metadata = probe_media(output)
    assert metadata.has_video is True
    assert metadata.has_audio is False
    assert (metadata.width, metadata.height, metadata.fps) == (320, 180, 10)
    assert metadata.duration == pytest.approx(0.4, abs=0.03)


@pytest.mark.parametrize(
    "overlay_type",
    [
        "ARROW",
        "ROUTE",
        "X_MARK",
        "CIRCLE",
        "HIGHLIGHT",
        "TEXT_LABEL",
        "MEASUREMENT",
        "DISTANCE",
        "DEPTH",
    ],
)
def test_every_overlay_type_produces_legal_video(
    source_image: Path,
    tmp_path: Path,
    overlay_type: str,
) -> None:
    directional = overlay_type in {"ARROW", "ROUTE", "MEASUREMENT", "DISTANCE", "DEPTH"}
    overlay = NormalizedOverlay(
        type=overlay_type,
        instruction="Show the safe route clearly",
        semantic_anchor="side exit",
        start={"x": 0.25, "y": 0.3} if directional else None,
        end={"x": 0.75, "y": 0.7} if directional else None,
        center=None if directional else {"x": 0.5, "y": 0.5},
        appear_offset=0.1,
    )
    output = tmp_path / f"overlay-{overlay_type}.mp4"
    render_timeline_entry(
        source_image,
        output,
        frame_count=4,
        config=ProjectRenderConfig(width=320, height=180, fps=10),
        overlay=overlay,
    )
    assert probe_media(output).duration == pytest.approx(0.4, abs=0.03)


def test_render_config_is_versioned_json_safe_and_rejects_odd_dimensions() -> None:
    config = ProjectRenderConfig()
    payload = config.model_dump(mode="json")
    assert payload["version"] == "project_render_config_v2"
    assert payload["image_fit_mode"] == "COVER"
    assert payload["default_transition"] == "CUT"
    with pytest.raises(ValueError, match="even"):
        ProjectRenderConfig(width=319)


def test_final_mux_applies_loudness_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(
        timeline_renderer,
        "_run_ffmpeg",
        lambda command, stage: commands.append(command),
    )

    mux_narration(
        "video.mp4",
        "narration.wav",
        "final.mp4",
        frame_count=240,
        config=ProjectRenderConfig(fps=24),
    )

    command = commands[0]
    assert command[command.index("-af") + 1] == "loudnorm=I=-15:TP=-1:LRA=11"
    assert command[command.index("-b:a") + 1] == "192k"


def test_timeline_segments_use_crf_encoding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    source = tmp_path / "source.png"
    source.write_bytes(b"png")
    monkeypatch.setattr(
        timeline_renderer,
        "_run_ffmpeg",
        lambda command, stage: commands.append(command),
    )

    render_timeline_entry(
        source,
        tmp_path / "segment.mp4",
        frame_count=24,
        config=ProjectRenderConfig(fps=24),
    )

    command = commands[0]
    assert command[command.index("-crf") + 1] == "20"
    assert command[command.index("-preset") + 1] == "medium"


def test_subtle_focus_transform_changes_rendered_pixels_perceptibly(
    tmp_path: Path,
) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is required")
    source = tmp_path / "pattern.png"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x180",
            "-frames:v",
            "1",
            str(source),
        ],
        check=True,
    )
    output = tmp_path / "subtle-focus.mp4"
    render_timeline_entry(
        source,
        output,
        frame_count=30,
        config=ProjectRenderConfig(width=320, height=180, fps=10),
        transform=NormalizedTransform(
            type="FOCUS",
            start_scale=1.0,
            end_scale=1.05,
            focus={"x": 0.68, "y": 0.34},
        ),
    )

    frame_hashes = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(output),
            "-vf",
            r"select=eq(n\,0)+eq(n\,29)",
            "-f",
            "framemd5",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    hashes = [
        line.rsplit(",", 1)[-1].strip()
        for line in frame_hashes.splitlines()
        if line and not line.startswith("#")
    ]

    assert len(hashes) == 2
    assert hashes[0] != hashes[1]


def test_ai_video_is_normalized_and_provider_audio_is_removed(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required")
    source = tmp_path / "provider-video.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=240x320:r=12:d=0.4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.4",
            "-shortest",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    output = tmp_path / "normalized.mp4"
    overlaid = tmp_path / "normalized-overlay.mp4"
    config = ProjectRenderConfig(
        width=320,
        height=180,
        fps=10,
        image_fit_mode=RenderImageFit.CONTAIN,
    )

    render_video_timeline_entry(
        source,
        output,
        frame_count=10,
        config=config,
    )
    render_video_timeline_entry(
        source,
        overlaid,
        frame_count=10,
        config=config,
        overlay=NormalizedOverlay(
            type="HIGHLIGHT",
            instruction="Highlight the center",
            semantic_anchor="center",
            center={"x": 0.5, "y": 0.5},
            appear_offset=0.1,
        ),
    )

    metadata = probe_media(output)
    overlay_metadata = probe_media(overlaid)
    assert metadata.has_video is True
    assert metadata.has_audio is False
    assert (metadata.width, metadata.height, metadata.fps) == (320, 180, 10)
    assert metadata.duration == pytest.approx(1.0, abs=0.04)
    assert overlay_metadata.has_audio is False
    assert (
        overlay_metadata.width,
        overlay_metadata.height,
        overlay_metadata.fps,
    ) == (320, 180, 10)
    assert overlay_metadata.duration == pytest.approx(1.0, abs=0.04)
    assert overlaid.read_bytes() != output.read_bytes()
