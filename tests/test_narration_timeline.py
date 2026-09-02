"""Tests for story narration timing, monotonic alignment, and Stage 5 timeline."""

import asyncio
import hashlib
import json
import shutil
import subprocess
from collections.abc import Iterator, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session

import app.services.project_render as render_service
from app.database import create_session_factory, create_sqlite_engine, init_database
from app.errors import ProjectTimelineRenderError, StaleProjectVisualPlanError
from app.media.probe import probe_media
from app.models.render import ProjectRenderConfig
from app.models.timeline import (
    NarrationTimingGranularity,
    NarrationTimingItem,
    NarrationTimingTrack,
    TimestampedNarrationOutput,
    TimestampSource,
)
from app.models.visual_plan import VisualPlan
from app.repositories import (
    create_beat_visual_result,
    create_project,
    delete_project,
    list_project_narration_assets,
    list_project_timelines,
    list_project_video_renders,
    save_project_visual_plan_record,
    save_visual_execution_plan,
    update_project,
)
from app.services.narration import (
    build_proportional_timing,
    generate_project_narration,
    normalize_provider_timing,
)
from app.services.narration_alignment import (
    align_project_visual_beats,
    align_visual_beats,
)
from app.services.project_render import build_render_manifest, render_project_video
from app.services.timeline import (
    build_project_timeline,
    build_timeline_quality_report,
    format_timeline_debug,
    timeline_is_current,
)
from app.services.visual_planning import hash_story_text
from app.tts_capabilities import TTSProviderCapabilities


@pytest.fixture
def session(tmp_path: Path) -> Iterator[Session]:
    engine = create_sqlite_engine(tmp_path / "app.db")
    init_database(engine)
    factory = create_session_factory(engine)
    with factory() as database_session:
        yield database_session
    engine.dispose()


class FakeTTSProvider:
    capabilities = TTSProviderCapabilities()

    def __init__(self, model: str, voice: str, language: str = "Russian") -> None:
        self.model = model
        self.voice = voice
        self.language = language
        self.calls: list[str] = []

    async def generate(self, text: str) -> bytes:
        self.calls.append(text)
        return f"audio:{text}".encode()


class FakeWordTimestampTTS(FakeTTSProvider):
    capabilities = TTSProviderCapabilities(supports_word_timestamps=True)

    async def generate_with_timestamps(self, text: str) -> TimestampedNarrationOutput:
        self.calls.append(text)
        words = text.split()
        items = [
            NarrationTimingItem(
                text=word,
                start=index * 0.5,
                end=(index + 1) * 0.5,
            )
            for index, word in enumerate(words)
        ]
        return TimestampedNarrationOutput(
            audio=b"native-timestamp-audio",
            timing=NarrationTimingTrack(
                source=TimestampSource.NATIVE_WORD_TIMESTAMPS,
                granularity=NarrationTimingGranularity.WORD,
                confidence=0.98,
                items=items,
            ),
        )


class FailingTTSProvider(FakeTTSProvider):
    async def generate(self, text: str) -> bytes:
        del text
        raise RuntimeError("fake TTS failure")


def _resolver_with_calls(
    providers: list[FakeTTSProvider],
) -> Any:
    def resolver(name: str, config: Mapping[str, Any] | None) -> FakeTTSProvider:
        options = dict(config or {})
        provider_class = (
            FakeWordTimestampTTS if name == "native" else FakeTTSProvider
        )
        provider = provider_class(
            str(options.get("model") or "qwen3-tts-flash"),
            str(options.get("voice") or "Cherry"),
            str(options.get("language") or "Russian"),
        )
        providers.append(provider)
        return provider

    return resolver


def _story_segments() -> list[str]:
    return [
        "First the gate was open.",
        "The water continued rising.",
        "He moved farther inside.",
        "The water continued rising.",
        "The gate became blocked.",
        "Rescuers marked the route.",
        "He moved farther inside.",
        "Everyone reached the side exit.",
    ]


def _plan() -> VisualPlan:
    operations = [
        "NEW_IMAGE",
        "TRANSFORM",
        "EDIT_EXISTING",
        "EDIT_EXISTING",
        "EDIT_EXISTING",
        "OVERLAY",
        "REUSE",
        "REFERENCE_GENERATION",
    ]
    beats = []
    for index, (segment, operation) in enumerate(
        zip(_story_segments(), operations, strict=True),
        start=1,
    ):
        beat_id = f"beat_{index}"
        beats.append(
            {
                "id": beat_id,
                "narration_segment": segment,
                "visual_purpose": f"Show story state {index}",
                "what_viewer_should_understand": f"Physical state {index}",
                "location_id": "tunnel",
                "characters_visible": ["person"],
                "important_objects": ["gate"],
                "camera_framing": "WIDE",
                "camera_view": f"Tunnel view {index}",
                "framing_reason": "Keep geography readable",
                "camera_movement": "PAN" if operation == "TRANSFORM" else "STATIC",
                "geography_established_by": None,
                "master_scene_id": None,
                "physical_state": f"State {index}",
                "progressive_change": None,
                "safety_geography": None,
                "anticipated_consequence": None,
                "visual_focus": "gate",
                "must_not_show": [],
                "change_from_previous_beat": f"Change {index}",
                "information_added_beyond_narration": "Shows route geometry",
                "preferred_visual_operation": operation,
                "source_visual_id": f"beat_{index - 1}" if index > 1 else None,
                "overlay_description": (
                    "Draw a route arrow toward the side exit"
                    if operation == "OVERLAY"
                    else None
                ),
                "estimated_duration_seconds": 4,
            }
        )
    return VisualPlan.model_validate(
        {
            "story_summary": "A person escapes a flooding tunnel.",
            "visual_strategy": "Evolve one stable tunnel environment.",
            "characters": [
                {"id": "person", "name": "Person", "description": "Simple explorer"}
            ],
            "locations": [
                {
                    "id": "tunnel",
                    "name": "Tunnel",
                    "description": "One rectangular tunnel",
                    "spatial_layout": "Gate left, side exit right",
                }
            ],
            "important_objects": [
                {
                    "id": "gate",
                    "name": "Gate",
                    "description": "Simple rectangular gate",
                    "story_function": "Controls the route",
                }
            ],
            "recurring_environments": [],
            "possible_master_scenes": [],
            "visual_beats": beats,
        }
    )


def _setup_graph(session: Session, tmp_path: Path) -> tuple[Any, Any, list[Any]]:
    story = " ".join(_story_segments())
    project = create_project(
        session,
        name="Timeline story",
        story_text=story,
        image_provider="seedream",
        tts_provider="qwen",
        tts_model="qwen3-tts-flash",
        tts_voice="Cherry",
        tts_language="Russian",
    )
    plan = _plan()
    plan_record = save_project_visual_plan_record(
        session,
        project_id=project.id,
        schema_version="visual_plan_v1",
        visual_director_version="visual_director_v1",
        story_text_hash=hash_story_text(story),
        plan_json=plan.model_dump(mode="json"),
    )
    plan_revision = _hash(plan.model_dump(mode="json"))
    execution = save_visual_execution_plan(
        session,
        project_id=project.id,
        visual_plan=plan_record,
        visual_plan_revision=plan_revision,
        provider="seedream",
        model="fake-image",
        capability_snapshot={},
        decision_input_snapshot={},
        resolution_revision=_hash({"project": project.id}),
        decisions=[
            {
                "position": position,
                "beat_id": beat.id,
                "preferred_operation": beat.preferred_visual_operation.value,
                "resolved_operation": beat.preferred_visual_operation.value,
                "fallback_used": False,
                "fallback_from": None,
                "reason": ["fixture"],
                "source_visual_ids": [beat.source_visual_id]
                if beat.source_visual_id
                else [],
                "source_image_paths": [],
            }
            for position, beat in enumerate(plan.visual_beats)
        ],
    )
    results = []
    for position, beat in enumerate(plan.visual_beats, start=1):
        source = results[-1] if results else None
        path = (
            Path(source.output_path)
            if source is not None
            and beat.preferred_visual_operation.value in {"REUSE", "TRANSFORM", "OVERLAY"}
            else tmp_path / f"beat-{position}.png"
        )
        if not path.exists():
            path.write_bytes(f"visual-{position}".encode())
        result = create_beat_visual_result(
            session,
            project_id=project.id,
            visual_plan_id=plan_record.id,
            visual_plan_revision=plan_revision,
            execution_plan_id=execution.id,
            beat_id=beat.id,
            resolved_operation=beat.preferred_visual_operation.value,
            source_result_id=source.id if source else None,
            source_master_asset_id=None,
            source_path=source.output_path if source else None,
            output_path=str(path),
            file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            master_scene_id=None,
            prompt_used="prompt" if beat.preferred_visual_operation.value not in {"REUSE", "TRANSFORM", "OVERLAY"} else None,
            provider="seedream",
            model="fake-image",
            style_version="rough_explainer_v1",
            reference_snapshot=[],
            generation_status="SUCCEEDED",
            qa_status="PASS" if beat.preferred_visual_operation.value not in {"REUSE", "TRANSFORM", "OVERLAY"} else "NOT_RUN",
            is_accepted=True,
            error=None,
            transform_metadata=(
                {"type": "PAN", "target": "gate"}
                if beat.preferred_visual_operation.value == "TRANSFORM"
                else None
            ),
            overlay_metadata=(
                {
                    "type": "ROUTE",
                    "instruction": "Draw a route arrow toward the side exit",
                    "semantic_anchor": "gate",
                }
                if beat.preferred_visual_operation.value == "OVERLAY"
                else None
            ),
            semantic_state_snapshot={"position": position},
            generation_revision=_hash({"beat": beat.id}),
            attempt=1,
        )
        results.append(result)
    return project, execution, results


def _make_real_render_graph(session: Session, tmp_path: Path) -> tuple[Any, Any]:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required for the real render test")
    project, execution, results = _setup_graph(session, tmp_path)
    source = tmp_path / "source.png"
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
            "color=c=0x284466:s=320x180", "-frames:v", "1", str(source),
        ],
        check=True,
    )
    for result in results:
        assert result.output_path is not None
        Path(result.output_path).write_bytes(source.read_bytes())
        result.file_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    session.commit()
    narration = asyncio.run(
        generate_project_narration(
            session,
            project.id,
            provider_resolver=_resolver_with_calls([]),
            duration_probe=lambda path: 4.0,
            projects_root=tmp_path / "projects",
        )
    )
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
            "sine=frequency=440:duration=4", "-c:a", "pcm_s16le",
            narration.audio_path,
        ],
        check=True,
    )
    narration.audio_sha256 = hashlib.sha256(Path(narration.audio_path).read_bytes()).hexdigest()
    session.commit()
    return project, build_project_timeline(session, project.id, execution.id, narration.id)


def test_story_narration_persists_and_unchanged_request_is_idempotent(
    session: Session,
    tmp_path: Path,
) -> None:
    project, _, _ = _setup_graph(session, tmp_path)
    providers: list[FakeTTSProvider] = []
    resolver = _resolver_with_calls(providers)

    first = asyncio.run(
        generate_project_narration(
            session,
            project.id,
            provider_resolver=resolver,
            duration_probe=lambda path: 32.0,
            projects_root=tmp_path / "projects",
        )
    )
    second = asyncio.run(
        generate_project_narration(
            session,
            project.id,
            provider_resolver=resolver,
            duration_probe=lambda path: 32.0,
            projects_root=tmp_path / "projects",
        )
    )

    assert first.id == second.id
    assert Path(first.audio_path).is_file()
    assert first.duration == 32.0
    assert first.timestamp_source == "PROPORTIONAL_FALLBACK"
    assert len(list_project_narration_assets(session, project.id)) == 1
    assert sum(len(item.calls) for item in providers) == 1


def test_timestamp_capable_provider_persists_native_word_timing(
    session: Session,
    tmp_path: Path,
) -> None:
    project, _, _ = _setup_graph(session, tmp_path)

    def resolver(name: str, config: Mapping[str, Any] | None) -> FakeWordTimestampTTS:
        del name
        options = dict(config or {})
        return FakeWordTimestampTTS(
            str(options.get("model")),
            str(options.get("voice")),
            str(options.get("language")),
        )

    narration = asyncio.run(
        generate_project_narration(
            session,
            project.id,
            provider_resolver=resolver,
            duration_probe=lambda path: 32.0,
            projects_root=tmp_path / "projects",
        )
    )

    assert narration.timestamp_source == "NATIVE_WORD_TIMESTAMPS"
    assert narration.timing_granularity == "WORD"
    assert narration.timing_confidence == 0.98
    assert narration.timing_data[0]["source_start_char"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("story_text", "A completely changed narration."),
        ("tts_model", "another-model"),
        ("tts_voice", "AnotherVoice"),
        ("tts_provider", "elevenlabs"),
    ],
)
def test_narration_inputs_create_a_new_revision(
    session: Session,
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    project, _, _ = _setup_graph(session, tmp_path)
    providers: list[FakeTTSProvider] = []
    first = asyncio.run(
        generate_project_narration(
            session,
            project.id,
            provider_resolver=_resolver_with_calls(providers),
            duration_probe=lambda path: 32.0,
            projects_root=tmp_path / "projects",
        )
    )
    update_project(session, project.id, **{field: value})
    second = asyncio.run(
        generate_project_narration(
            session,
            project.id,
            provider_resolver=_resolver_with_calls(providers),
            duration_probe=lambda path: 30.0,
            projects_root=tmp_path / "projects",
        )
    )

    assert first.id != second.id
    assert first.audio_path != second.audio_path
    assert Path(first.audio_path).is_file()


def test_failed_narration_replacement_preserves_previous_success(
    session: Session,
    tmp_path: Path,
) -> None:
    project, _, _ = _setup_graph(session, tmp_path)
    first = asyncio.run(
        generate_project_narration(
            session,
            project.id,
            provider_resolver=_resolver_with_calls([]),
            duration_probe=lambda path: 32.0,
            projects_root=tmp_path / "projects",
        )
    )
    update_project(session, project.id, tts_voice="ReplacementVoice")

    def failing_resolver(
        name: str,
        config: Mapping[str, Any] | None,
    ) -> FailingTTSProvider:
        del name, config
        return FailingTTSProvider("qwen3-tts-flash", "ReplacementVoice")

    with pytest.raises(RuntimeError, match="fake TTS failure"):
        asyncio.run(
            generate_project_narration(
                session,
                project.id,
                provider_resolver=failing_resolver,
                duration_probe=lambda path: 32.0,
                projects_root=tmp_path / "projects",
            )
        )

    assert list_project_narration_assets(session, project.id) == [first]
    assert Path(first.audio_path).is_file()


def test_native_word_and_sentence_timestamps_are_normalized() -> None:
    word_track = NarrationTimingTrack(
        source="NATIVE_WORD_TIMESTAMPS",
        granularity="WORD",
        confidence=0.98,
        items=[
            {"text": "One", "start": 0, "end": 0.5},
            {"text": "two", "start": 0.5, "end": 1.0},
        ],
    )
    normalized = normalize_provider_timing("One, two.", 1.0, word_track)
    assert [(item.source_start_char, item.source_end_char) for item in normalized.items] == [(0, 3), (5, 8)]

    sentence_track = NarrationTimingTrack(
        source="NATIVE_SENTENCE_TIMESTAMPS",
        granularity="SENTENCE",
        confidence=0.9,
        items=[
            {"text": "First sentence.", "start": 0, "end": 1.2},
            {"text": "Second sentence.", "start": 1.2, "end": 2.5},
        ],
    )
    sentences = normalize_provider_timing(
        "First sentence. Second sentence.",
        2.5,
        sentence_track,
    )
    assert sentences.granularity is NarrationTimingGranularity.SENTENCE
    assert sentences.items[1].source_start_char > sentences.items[0].source_end_char


def test_proportional_timing_is_deterministic_and_covers_audio() -> None:
    first = build_proportional_timing("Short much-longer final", 9.0)
    second = build_proportional_timing("Short much-longer final", 9.0)
    assert first == second
    assert first.items[0].start == 0
    assert first.items[-1].end == 9.0
    assert first.items[1].end - first.items[1].start > first.items[0].end


def test_repeated_phrases_align_monotonically() -> None:
    plan = _plan()
    story = " ".join(_story_segments())
    timing = build_proportional_timing(story, 32.0)
    drafts = align_visual_beats(story, plan.visual_beats, timing, 32.0)
    repeated = [drafts[1], drafts[3]]
    assert repeated[1].source_start_char > repeated[0].source_end_char
    assert all(
        current.audio_start >= previous.audio_start
        for previous, current in pairwise(drafts)
    )


def test_ambiguous_alignment_uses_warned_low_confidence_fallback() -> None:
    plan = _plan()
    story = "Different narration with no matching visual segment words."
    timing = build_proportional_timing(story, 12.0)
    drafts = align_visual_beats(story, plan.visual_beats, timing, 12.0)
    assert all(item.audio_end > item.audio_start for item in drafts)
    assert any(item.warning for item in drafts)
    assert any(item.alignment_confidence == 0.35 for item in drafts)


def test_complete_eight_beat_timeline_is_gap_free_and_executable(
    session: Session,
    tmp_path: Path,
) -> None:
    project, execution, results = _setup_graph(session, tmp_path)
    narration = asyncio.run(
        generate_project_narration(
            session,
            project.id,
            provider_resolver=_resolver_with_calls([]),
            duration_probe=lambda path: 32.0,
            projects_root=tmp_path / "projects",
        )
    )
    timeline = build_project_timeline(
        session,
        project.id,
        execution.id,
        narration.id,
    )

    assert len(timeline.entries) == 8
    assert timeline.entries[0].start_time == 0
    assert timeline.entries[-1].end_time == 32.0
    assert all(
        left.end_time == right.start_time
        for left, right in pairwise(timeline.entries)
    )
    assert timeline.entries[1].asset_path == results[0].output_path
    assert timeline.entries[1].transform_metadata == {
        "type": "PAN",
        "from": {"x": 0.45, "y": 0.5},
        "to": {"x": 0.55, "y": 0.5},
    }
    assert timeline.entries[5].asset_path == results[4].output_path
    assert timeline.entries[5].overlay_metadata["type"] == "ROUTE"
    assert timeline.entries[5].overlay_metadata["start"]["y"] == 0.25
    assert timeline.entries[6].asset_path == results[5].output_path
    assert timeline_is_current(session, timeline) is True
    assert build_project_timeline(session, project.id, execution.id, narration.id).id == timeline.id
    assert len(list_project_timelines(session, project.id)) == 1
    debug = format_timeline_debug(timeline)
    assert "Beat 01" in debug and "TRANSFORM" in debug and "OVERLAY" in debug
    report = build_timeline_quality_report(timeline)
    assert report.timeline_entries == 8
    assert report.operation_counts["EDIT_EXISTING"] == 3


def test_new_accepted_visual_and_tts_change_make_timeline_stale(
    session: Session,
    tmp_path: Path,
) -> None:
    project, execution, results = _setup_graph(session, tmp_path)
    narration = asyncio.run(
        generate_project_narration(
            session,
            project.id,
            provider_resolver=_resolver_with_calls([]),
            duration_probe=lambda path: 32.0,
            projects_root=tmp_path / "projects",
        )
    )
    timeline = build_project_timeline(session, project.id, execution.id, narration.id)
    replacement_path = tmp_path / "replacement.png"
    replacement_path.write_bytes(b"replacement")
    original = results[0]
    replacement = create_beat_visual_result(
        session,
        project_id=project.id,
        visual_plan_id=original.visual_plan_id,
        visual_plan_revision=original.visual_plan_revision,
        execution_plan_id=execution.id,
        beat_id=original.beat_id,
        resolved_operation=original.resolved_operation,
        source_result_id=None,
        source_master_asset_id=None,
        source_path=None,
        output_path=str(replacement_path),
        file_sha256=hashlib.sha256(b"replacement").hexdigest(),
        master_scene_id=None,
        prompt_used="replacement prompt",
        provider="seedream",
        model="fake-image",
        style_version="rough_explainer_v1",
        reference_snapshot=[],
        generation_status="SUCCEEDED",
        qa_status="PASS",
        is_accepted=True,
        error=None,
        transform_metadata=None,
        overlay_metadata=None,
        semantic_state_snapshot={"position": 1},
        generation_revision=_hash({"replacement": True}),
        attempt=2,
    )
    assert replacement.id != original.id
    assert timeline_is_current(session, timeline) is False
    revised = build_project_timeline(session, project.id, execution.id, narration.id)
    assert revised.id != timeline.id
    assert revised.entries[0].beat_visual_result_id == replacement.id

    update_project(session, project.id, tts_voice="NewVoice")
    assert timeline_is_current(session, revised) is False
    with pytest.raises(ValueError, match="Narration asset is stale"):
        build_project_timeline(session, project.id, execution.id, narration.id)


def test_stale_visual_plan_cannot_build_timeline(
    session: Session,
    tmp_path: Path,
) -> None:
    project, execution, _ = _setup_graph(session, tmp_path)
    narration = asyncio.run(
        generate_project_narration(
            session,
            project.id,
            provider_resolver=_resolver_with_calls([]),
            duration_probe=lambda path: 32.0,
            projects_root=tmp_path / "projects",
        )
    )
    update_project(session, project.id, story_text="Changed after visual planning")
    with pytest.raises(StaleProjectVisualPlanError, match="story changed"):
        build_project_timeline(session, project.id, execution.id, narration.id)


def test_alignment_and_timeline_survive_restart(tmp_path: Path) -> None:
    engine = create_sqlite_engine(tmp_path / "restart.db")
    init_database(engine)
    factory = create_session_factory(engine)
    with factory() as first_session:
        project, execution, _ = _setup_graph(first_session, tmp_path)
        narration = asyncio.run(
            generate_project_narration(
                first_session,
                project.id,
                provider_resolver=_resolver_with_calls([]),
                duration_probe=lambda path: 32.0,
                projects_root=tmp_path / "projects",
            )
        )
        alignment = align_project_visual_beats(first_session, project.id, narration)
        timeline = build_project_timeline(
            first_session,
            project.id,
            execution.id,
            narration.id,
        )
        ids = narration.id, alignment.id, timeline.id
    init_database(engine)
    with factory() as restarted:
        assert restarted.get(type(narration), ids[0]) is not None
        persisted_alignment = restarted.get(type(alignment), ids[1])
        persisted_timeline = restarted.get(type(timeline), ids[2])
        assert persisted_alignment is not None
        assert len(persisted_alignment.beat_timings) == 8
        assert persisted_timeline is not None
        assert len(persisted_timeline.entries) == 8
        assert delete_project(restarted, project.id) is True
        assert restarted.get(type(narration), ids[0]) is None
        assert restarted.get(type(alignment), ids[1]) is None
        assert restarted.get(type(timeline), ids[2]) is None
    engine.dispose()


def test_real_project_timeline_render_is_valid_and_idempotent(
    session: Session,
    tmp_path: Path,
) -> None:
    project, timeline = _make_real_render_graph(session, tmp_path)
    config = ProjectRenderConfig(width=320, height=180, fps=10)

    first = render_project_video(
        session, project.id, timeline.id, config=config,
        projects_root=tmp_path / "render-projects",
    )
    output = Path(first.output_path or "")
    mtime = output.stat().st_mtime_ns
    second = render_project_video(
        session, project.id, timeline.id, config=config,
        projects_root=tmp_path / "render-projects",
    )
    metadata = probe_media(output)

    assert first.id == second.id
    assert output.stat().st_mtime_ns == mtime
    assert first.status == "SUCCEEDED"
    assert metadata.has_video and metadata.has_audio
    assert (metadata.width, metadata.height) == (320, 180)
    assert metadata.fps == 10
    assert metadata.duration == pytest.approx(4.0, abs=0.08)
    assert first.diagnostics["total_frames"] == 40
    assert project.final_video_path == str(output)
    assert project.rendered_at is not None
    assert len(list_project_video_renders(session, project.id)) == 1
    manifest = build_render_manifest(timeline, config)
    assert manifest["transition_counts"] == {"CUT": 7}
    assert sum(entry["frames"] for entry in manifest["entries"]) == 40

    changed_config = render_project_video(
        session,
        project.id,
        timeline.id,
        config=ProjectRenderConfig(width=322, height=180, fps=10),
        projects_root=tmp_path / "render-projects",
    )
    assert changed_config.id != first.id
    assert changed_config.render_revision != first.render_revision
    assert probe_media(changed_config.output_path or "").width == 322


def test_render_failure_is_persisted_and_preserves_previous_success(
    session: Session,
    tmp_path: Path,
) -> None:
    project, timeline = _make_real_render_graph(session, tmp_path)
    root = tmp_path / "render-projects"
    successful = render_project_video(
        session, project.id, timeline.id,
        config=ProjectRenderConfig(width=320, height=180, fps=10),
        projects_root=root,
    )
    Path(timeline.entries[0].asset_path).write_bytes(b"changed")

    with pytest.raises(Exception, match="Timeline is stale"):
        render_project_video(
            session, project.id, timeline.id,
            config=ProjectRenderConfig(width=320, height=180, fps=10),
            projects_root=root,
        )

    records = list_project_video_renders(session, project.id)
    assert [record.status for record in records] == ["SUCCEEDED", "FAILED"]
    assert records[-1].failed_stage == "VALIDATION"
    assert records[-1].attempt == 2
    assert Path(successful.output_path or "").is_file()
    assert project.final_video_path == successful.output_path


def test_failed_attempt_resumes_from_valid_intermediate_clips(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, timeline = _make_real_render_graph(session, tmp_path)
    config = ProjectRenderConfig(width=320, height=180, fps=10)
    root = tmp_path / "render-projects"
    original_renderer = render_service.render_timeline_entry
    call_count = 0

    def fail_on_second_entry(*args: Any, **kwargs: Any) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("intentional segment failure")
        return original_renderer(*args, **kwargs)

    monkeypatch.setattr(render_service, "render_timeline_entry", fail_on_second_entry)
    with pytest.raises(ProjectTimelineRenderError, match="Beat 2"):
        render_project_video(
            session, project.id, timeline.id, config=config, projects_root=root
        )

    monkeypatch.setattr(render_service, "render_timeline_entry", original_renderer)
    successful = render_project_video(
        session, project.id, timeline.id, config=config, projects_root=root
    )
    records = list_project_video_renders(session, project.id)
    assert [record.status for record in records] == ["FAILED", "SUCCEEDED"]
    assert successful.attempt == 2
    assert successful.diagnostics["reused_intermediate_clips"] >= 1


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
