"""Full one-click Project pipeline tests with fake paid providers and real FFmpeg."""

import asyncio
import json
import shutil
import subprocess
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_master_scene import _plan

from app.budgets import BUDGET_ESTIMATE_EXCEEDS_LIMIT, GenerationBudgetError
from app.costs import PricingUnit, configure_provider_pricing
from app.database import create_session_factory, create_sqlite_engine, init_database
from app.generation_scope import GenerationScope, GenerationScopeType
from app.jobs import GenerationJobManager, GenerationJobType
from app.models.visual_plan import VisualPlan
from app.models.visual_qa import VisualQADecision
from app.persistence import ProviderUsageRecord
from app.pipeline.visual_qa import VisualQAService
from app.production_profiles import ProductionProfile
from app.provider_capabilities import ImageProviderCapabilities
from app.repositories import (
    create_project,
    list_beat_visual_results,
    list_master_scene_assets,
    list_project_narration_assets,
    list_project_timelines,
    list_project_video_renders,
    update_project,
)
from app.services.project_pipeline import (
    ProjectPipelineDependencies,
    run_project_video_pipeline,
)
from app.tts_capabilities import TTSProviderCapabilities


@pytest.fixture
def session(tmp_path: Path) -> Iterator[Session]:
    engine = create_sqlite_engine(tmp_path / "app.db")
    init_database(engine)
    factory = create_session_factory(engine)
    with factory() as value:
        yield value
    engine.dispose()


class FakePlanningClient:
    def __init__(self, plan: VisualPlan | None = None) -> None:
        self.calls = 0
        self.plan = plan or _plan()

    async def generate(self, prompt: str) -> str:
        self.calls += 1
        assert "COMPLETE NARRATION" in prompt
        return json.dumps(self.plan.model_dump(mode="json"))


class FakeImageProvider:
    capabilities = ImageProviderCapabilities(
        reference_generation=True,
        multiple_references=True,
        max_reference_images=3,
        image_editing=True,
    )
    model = "fake-image"

    def __init__(self, *, fail_call: int | None = None) -> None:
        self.calls = 0
        self.fail_call = fail_call

    async def generate(self, prompt: str) -> str:
        return await self._call(prompt)

    async def generate_with_references(self, prompt: str, references: tuple) -> str:
        assert references
        return await self._call(prompt)

    async def edit(self, prompt: str, references: tuple) -> str:
        assert references
        return await self._call(prompt)

    async def _call(self, prompt: str) -> str:
        self.calls += 1
        assert "STYLE CONTRACT" in prompt
        if self.calls == self.fail_call:
            raise RuntimeError("fake image timeout")
        return f"fake://image/{self.calls}"


class TextOnlyImageProvider:
    capabilities = ImageProviderCapabilities()
    model = "text-only-image"

    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return f"fake://image/{self.calls}"


class FakeTTSProvider:
    capabilities = TTSProviderCapabilities()
    model = "fake-tts"
    language = "Russian"

    def __init__(self, audio: bytes, voice: str = "TestVoice") -> None:
        self.audio = audio
        self.voice = voice
        self.calls = 0

    async def generate(self, text: str) -> bytes:
        self.calls += 1
        assert text
        return self.audio


class PassingQAClient:
    provider = "fake-vision"
    model = "fake-vision"

    def __init__(self) -> None:
        self.calls = 0

    async def evaluate(self, prompt: str, image_paths: tuple[str, ...]) -> str:
        self.calls += 1
        assert image_paths and "VIDEO FRAME" in prompt
        return VisualQADecision(
            result="PASS",
            problem_categories=[],
            reasons=[],
            correction_instruction=None,
        ).model_dump_json()


def _assets(tmp_path: Path) -> tuple[bytes, bytes]:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg is required")
    image = tmp_path / "source.png"
    audio = tmp_path / "source.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=navy:s=320x180",
            "-frames:v",
            "1",
            str(image),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:a",
            "pcm_s16le",
            str(audio),
        ],
        check=True,
    )
    return image.read_bytes(), audio.read_bytes()


def _dependencies(
    tmp_path: Path,
    provider: FakeImageProvider,
    image_bytes: bytes,
    audio_bytes: bytes,
    planning: FakePlanningClient,
    qa: PassingQAClient,
) -> ProjectPipelineDependencies:
    async def downloader(url: str, output_path: str) -> str:
        assert url.startswith("fake://image/")
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(image_bytes)
        return output_path

    def image_resolver(
        name: str, config: Mapping[str, Any] | None
    ) -> FakeImageProvider:
        del name, config
        return provider

    def tts_resolver(name: str, config: Mapping[str, Any] | None) -> FakeTTSProvider:
        del name
        options = dict(config or {})
        return FakeTTSProvider(audio_bytes, str(options.get("voice") or "TestVoice"))

    return ProjectPipelineDependencies(
        planning_client=planning,
        image_provider_resolver=image_resolver,
        tts_provider_resolver=tts_resolver,
        visual_qa_service=VisualQAService(qa),
        projects_root=tmp_path / "projects",
        downloader=downloader,
    )


def _project(session: Session):
    return create_project(
        session,
        name="One-click mine story",
        story_text="The miners worked far below.",
        image_provider="seedream",
        image_model="fake-image",
        tts_provider="qwen",
        tts_model="fake-tts",
        tts_voice="TestVoice",
        width=320,
        height=180,
        fps=10,
        image_fit="cover",
    )


def _two_beat_plan() -> VisualPlan:
    payload = _plan().model_dump(mode="json")
    second = dict(payload["visual_beats"][0])
    second.update(
        {
            "id": "beat_2",
            "narration_segment": "Then the route began to fail.",
            "visual_purpose": "Show the first dangerous change",
            "what_viewer_should_understand": "The route is becoming unsafe",
            "camera_framing": "MEDIUM",
            "camera_view": "Workers and the lower ladder",
            "framing_reason": "Keep geography while making the change readable",
            "camera_movement": "STATIC",
            "change_from_previous_beat": "The lower ladder starts to break",
            "physical_state": "The lower ladder has one broken section",
            "progressive_change": {
                "subject_id": "ladder",
                "previous_state": "intact",
                "current_state": "one lower section is broken",
                "progression": "safe route to damaged route",
            },
            "information_added_beyond_narration": "Shows where the failure begins",
            "preferred_visual_operation": "EDIT_EXISTING",
            "source_visual_id": "beat_1",
            "estimated_duration_seconds": 4,
        }
    )
    payload["visual_beats"].append(second)
    return VisualPlan.model_validate(payload)


def _four_beat_plan() -> VisualPlan:
    payload = _two_beat_plan().model_dump(mode="json")
    for index in (3, 4):
        beat = dict(payload["visual_beats"][-1])
        beat.update(
            id=f"beat_{index}",
            narration_segment=f"The danger advances, step {index}.",
            source_visual_id=f"beat_{index - 1}",
            change_from_previous_beat=f"Damage advances to step {index}",
            physical_state=f"The ladder damage is now at step {index}",
        )
        beat["progressive_change"] = {
            "subject_id": "ladder",
            "previous_state": f"damage step {index - 1}",
            "current_state": f"damage step {index}",
            "progression": "safe route to unusable route",
        }
        payload["visual_beats"].append(beat)
    return VisualPlan.model_validate(payload)


def test_style_preview_stops_after_three_images_without_tts_or_video(
    session: Session,
    tmp_path: Path,
) -> None:
    project = _project(session)
    planning = FakePlanningClient(_four_beat_plan())
    provider = FakeImageProvider()
    qa = PassingQAClient()

    async def downloader(url: str, output_path: str) -> str:
        assert url.startswith("fake://image/")
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"preview-image")
        return output_path

    def image_resolver(name: str, config: Mapping[str, Any] | None) -> FakeImageProvider:
        del name, config
        return provider

    def forbidden_tts_resolver(name: str, config: Mapping[str, Any] | None) -> FakeTTSProvider:
        del name, config
        raise AssertionError("Style preview must not resolve or call TTS")

    dependencies = ProjectPipelineDependencies(
        planning_client=planning,
        image_provider_resolver=image_resolver,
        tts_provider_resolver=forbidden_tts_resolver,
        visual_qa_service=VisualQAService(qa),
        projects_root=tmp_path / "projects",
        downloader=downloader,
    )

    report = asyncio.run(
        run_project_video_pipeline(
            session,
            project.id,
            dependencies,
            job_id="style-preview-run",
            generation_scope=GenerationScope(GenerationScopeType.STYLE_PREVIEW),
        )
    )

    assert report.generation_scope["type"] == "STYLE_PREVIEW"
    assert report.visual_beats == 3
    assert report.semantic_visual_beats == 4
    assert len(report.preview_result_ids) == 3
    assert len(list_project_narration_assets(session, project.id)) == 0
    assert len(list_project_timelines(session, project.id)) == 0
    assert len(list_project_video_renders(session, project.id)) == 0
    assert project.final_video_path is None


def _configure_budget_prices(session: Session) -> None:
    effective = datetime(2026, 1, 1, tzinfo=UTC)
    for provider, model, operation, unit, price in [
        ("dashscope", "qwen-plus", "PLANNING", PricingUnit.PER_REQUEST, 0.01),
        ("seedream", "fake-image", "NEW_IMAGE", PricingUnit.PER_IMAGE, 0.06),
        ("seedream", "fake-image", "REFERENCE_GENERATION", PricingUnit.PER_IMAGE, 0.06),
        ("seedream", "fake-image", "EDIT", PricingUnit.PER_IMAGE, 0.06),
        ("dashscope", "qwen-vl-max", "VISUAL_QA", PricingUnit.PER_REQUEST, 0.01),
        ("qwen", "fake-tts", "TTS", PricingUnit.PER_CHARACTER, 0.001),
    ]:
        configure_provider_pricing(
            session,
            provider=provider,
            model=model,
            operation=operation,
            pricing_unit=unit,
            price=price,
            currency="USD",
            version="v1",
            effective_from=effective,
        )


def test_full_pipeline_uses_semantic_master_fallback_for_text_only_provider(
    session: Session,
    tmp_path: Path,
) -> None:
    image, audio = _assets(tmp_path)
    project = _project(session)
    project = update_project(session, project.id, visual_qa_enabled=False)
    assert project is not None
    provider = TextOnlyImageProvider()

    def image_resolver(name: str, config: Mapping[str, Any] | None) -> TextOnlyImageProvider:
        del name, config
        return provider

    async def downloader(url: str, output_path: str) -> str:
        assert url.startswith("fake://image/")
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(image)
        return output_path

    dependencies = ProjectPipelineDependencies(
        planning_client=FakePlanningClient(),
        image_provider_resolver=image_resolver,
        tts_provider_resolver=lambda name, config: FakeTTSProvider(audio),
        visual_qa_service=None,
        projects_root=tmp_path / "projects",
        downloader=downloader,
    )

    report = asyncio.run(
        run_project_video_pipeline(
            session,
            project.id,
            dependencies,
            job_id="text-only-master-run",
        )
    )

    assert provider.calls == 1
    assert "Use the same recurring environment" in provider.prompts[0]
    assert "Vertical shaft, surface above, side tunnel below" in provider.prompts[0]
    assert list_master_scene_assets(session, project.id) == []
    assert Path(report.final_mp4).is_file()


def test_full_pipeline_from_story_only_completes_background_job(
    session: Session,
    tmp_path: Path,
) -> None:
    image, audio = _assets(tmp_path)
    project = _project(session)
    planning = FakePlanningClient()
    provider = FakeImageProvider()
    qa = PassingQAClient()
    dependencies = _dependencies(tmp_path, provider, image, audio, planning, qa)
    manager = GenerationJobManager(tmp_path / "jobs.db")

    async def scenario() -> None:
        async def operation(job_id: str) -> None:
            report = await run_project_video_pipeline(session, project.id, dependencies)
            await manager.set_pipeline_result(
                job_id,
                final_render_id=report.final_render_id,
                report=report.as_dict(),
            )

        job = await manager.enqueue(
            project.id, GenerationJobType.GENERATE_VIDEO, operation
        )
        completed = await manager.wait(job.id)
        assert completed is not None
        assert completed.status.value == "completed"
        assert completed.final_render_id is not None
        assert completed.report["visual_beats"] == 1

    asyncio.run(scenario())
    assert planning.calls == 1
    assert len(list_master_scene_assets(session, project.id)) == 1
    assert len(list_beat_visual_results(session, project.id, accepted_only=True)) == 1
    assert len(list_project_narration_assets(session, project.id)) == 1
    assert len(list_project_timelines(session, project.id)) == 1
    render = list_project_video_renders(session, project.id)[0]
    assert render.status == "SUCCEEDED"
    assert Path(render.output_path or "").is_file()
    assert qa.calls >= 2  # master and generated beat


def test_pipeline_preflight_budget_blocks_images_before_provider_call(
    session: Session,
    tmp_path: Path,
) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg is required by pipeline preflight")
    project = _project(session)
    update_project(
        session,
        project.id,
        generation_budget_enabled=True,
        generation_budget_amount=0.05,
        generation_budget_currency="USD",
    )
    _configure_budget_prices(session)
    planning = FakePlanningClient()
    provider = FakeImageProvider()
    qa = PassingQAClient()
    dependencies = _dependencies(tmp_path, provider, b"unused", b"unused", planning, qa)

    with pytest.raises(GenerationBudgetError) as error:
        asyncio.run(
            run_project_video_pipeline(
                session, project.id, dependencies, job_id="budget-job"
            )
        )

    assert error.value.code == BUDGET_ESTIMATE_EXCEEDS_LIMIT
    assert planning.calls == 1
    assert provider.calls == 0


def test_pipeline_failure_resumes_without_regenerating_completed_master(
    session: Session,
    tmp_path: Path,
) -> None:
    image, audio = _assets(tmp_path)
    project = _project(session)
    project = update_project(
        session,
        project.id,
        story_text="The miners worked far below. Then the route began to fail.",
    )
    assert project is not None
    planning = FakePlanningClient(_two_beat_plan())
    provider = FakeImageProvider(fail_call=3)
    qa = PassingQAClient()
    dependencies = _dependencies(tmp_path, provider, image, audio, planning, qa)

    with pytest.raises(Exception, match="fake image timeout"):
        asyncio.run(run_project_video_pipeline(session, project.id, dependencies))
    assert provider.calls == 3
    assert len(list_master_scene_assets(session, project.id)) == 1
    assert (
        list_beat_visual_results(session, project.id)[-1].generation_status == "FAILED"
    )
    accepted_before_resume = list_beat_visual_results(
        session, project.id, accepted_only=True
    )
    assert [item.beat_id for item in accepted_before_resume] == ["beat_1"]
    first_beat_result_id = accepted_before_resume[0].id

    provider.fail_call = None
    report = asyncio.run(run_project_video_pipeline(session, project.id, dependencies))
    assert Path(report.final_mp4).is_file()
    assert planning.calls == 1
    assert provider.calls == 4  # master and beat 1 were reused; only beat 2 retried
    assert report.reused["visual_plan"] is True
    assert report.reused["master_assets"] == 1
    assert report.reused["accepted_visual_assets"] == 1
    accepted_after_resume = list_beat_visual_results(
        session, project.id, accepted_only=True
    )
    assert accepted_after_resume[0].id == first_beat_result_id


def test_pipeline_revisions_invalidate_only_required_downstream_work(
    session: Session,
    tmp_path: Path,
) -> None:
    image, audio = _assets(tmp_path)
    project = _project(session)
    planning = FakePlanningClient()
    provider = FakeImageProvider()
    qa = PassingQAClient()
    dependencies = _dependencies(tmp_path, provider, image, audio, planning, qa)

    first = asyncio.run(run_project_video_pipeline(session, project.id, dependencies))
    initial_image_calls = provider.calls
    assert first.reused["visual_plan"] is False

    update_project(session, project.id, tts_voice="AnotherVoice")
    voice = asyncio.run(run_project_video_pipeline(session, project.id, dependencies))
    assert provider.calls == initial_image_calls
    assert planning.calls == 1
    assert len(list_project_narration_assets(session, project.id)) == 2
    assert voice.final_render_id != first.final_render_id

    update_project(session, project.id, width=322)
    resized = asyncio.run(run_project_video_pipeline(session, project.id, dependencies))
    assert provider.calls == initial_image_calls
    assert len(list_project_narration_assets(session, project.id)) == 2
    assert resized.final_render_id != voice.final_render_id

    provider.model = "fake-image-v2"
    update_project(session, project.id, image_model="fake-image-v2")
    changed_image_provider = asyncio.run(
        run_project_video_pipeline(session, project.id, dependencies)
    )
    assert provider.calls == initial_image_calls + 1
    assert planning.calls == 1
    assert changed_image_provider.reused["visual_plan"] is True

    update_project(session, project.id, story_text="A changed finished narration.")
    changed_story = asyncio.run(
        run_project_video_pipeline(session, project.id, dependencies)
    )
    assert planning.calls == 2
    assert provider.calls == initial_image_calls + 2
    assert changed_story.story_revision != first.story_revision


def test_draft_renders_preview_and_final_promotes_only_qa_passed_assets(
    session: Session,
    tmp_path: Path,
) -> None:
    image, audio = _assets(tmp_path)
    project = _project(session)
    update_project(
        session,
        project.id,
        story_text="The miners worked below. Then the route began to fail.",
        draft_width=320,
        draft_height=180,
    )
    planning = FakePlanningClient(_two_beat_plan())
    provider = FakeImageProvider()
    qa = PassingQAClient()
    dependencies = _dependencies(tmp_path, provider, image, audio, planning, qa)

    draft = asyncio.run(
        run_project_video_pipeline(
            session,
            project.id,
            dependencies,
            production_profile=ProductionProfile.DRAFT,
        )
    )
    calls_after_draft = provider.calls

    assert draft.production_profile == "DRAFT"
    assert Path(draft.final_mp4).is_file()
    assert draft.paid_visual_beats == 2
    assert (
        list_project_video_renders(session, project.id)[-1].production_profile
        == "DRAFT"
    )

    final = asyncio.run(
        run_project_video_pipeline(
            session,
            project.id,
            dependencies,
            production_profile=ProductionProfile.FINAL,
        )
    )

    assert final.production_profile == "FINAL"
    assert final.production_profile_version != draft.production_profile_version
    assert Path(final.final_mp4).is_file()
    assert provider.calls == calls_after_draft
    results = list_beat_visual_results(session, project.id, accepted_only=True)
    draft_results = [item for item in results if item.production_profile == "DRAFT"]
    final_results = [item for item in results if item.production_profile == "FINAL"]
    assert len(draft_results) == len(final_results) == 2
    assert all(item.source_result_id is not None for item in final_results)
    assert all(item.resolved_operation == "REUSE" for item in final_results)


def test_final_does_not_promote_draft_assets_without_visual_qa(
    session: Session,
    tmp_path: Path,
) -> None:
    image, audio = _assets(tmp_path)
    project = _project(session)
    update_project(
        session,
        project.id,
        draft_width=320,
        draft_height=180,
        visual_qa_enabled=False,
    )
    planning = FakePlanningClient()
    provider = FakeImageProvider()
    qa = PassingQAClient()
    draft_dependencies = _dependencies(tmp_path, provider, image, audio, planning, qa)
    draft_dependencies = ProjectPipelineDependencies(
        planning_client=draft_dependencies.planning_client,
        image_provider_resolver=draft_dependencies.image_provider_resolver,
        tts_provider_resolver=draft_dependencies.tts_provider_resolver,
        visual_qa_service=None,
        projects_root=draft_dependencies.projects_root,
        downloader=draft_dependencies.downloader,
    )

    asyncio.run(
        run_project_video_pipeline(
            session,
            project.id,
            draft_dependencies,
            production_profile=ProductionProfile.DRAFT,
        )
    )
    calls_after_draft = provider.calls
    update_project(session, project.id, visual_qa_enabled=True)
    final_dependencies = _dependencies(tmp_path, provider, image, audio, planning, qa)
    asyncio.run(
        run_project_video_pipeline(
            session,
            project.id,
            final_dependencies,
            production_profile=ProductionProfile.FINAL,
        )
    )

    assert provider.calls == calls_after_draft + 1
    final_result = next(
        item
        for item in list_beat_visual_results(session, project.id, accepted_only=True)
        if item.production_profile == "FINAL"
    )
    assert final_result.resolved_operation == "NEW_IMAGE"
    assert final_result.source_result_id is None


def test_pilot_renders_prefix_then_full_reuses_pilot_assets(
    session: Session,
    tmp_path: Path,
) -> None:
    image, audio = _assets(tmp_path)
    project = _project(session)
    project = update_project(
        session,
        project.id,
        story_text="The miners worked below. Then the route began to fail.",
    )
    assert project is not None
    scoped_plan_payload = _two_beat_plan().model_dump(mode="json")
    second_master = dict(scoped_plan_payload["possible_master_scenes"][0])
    second_master.update(
        id="side_tunnel_master",
        description="Separate side tunnel master",
        basic_composition="Side tunnel opening and exit route",
    )
    scoped_plan_payload["possible_master_scenes"].append(second_master)
    scoped_plan_payload["visual_beats"][1]["master_scene_id"] = "side_tunnel_master"
    scoped_plan_payload["visual_beats"][1]["source_visual_id"] = None
    scoped_plan_payload["visual_beats"][1]["geography_established_by"] = None
    scoped_plan_payload["visual_beats"][1]["preferred_visual_operation"] = "NEW_IMAGE"
    planning = FakePlanningClient(VisualPlan.model_validate(scoped_plan_payload))
    provider = FakeImageProvider()
    qa = PassingQAClient()
    dependencies = _dependencies(tmp_path, provider, image, audio, planning, qa)
    original_story = project.story_text

    pilot = asyncio.run(
        run_project_video_pipeline(
            session,
            project.id,
            dependencies,
            job_id="pilot-run",
            generation_scope=GenerationScope(
                GenerationScopeType.FIRST_BEATS,
                1,
            ),
        )
    )
    calls_after_pilot = provider.calls
    pilot_render = list_project_video_renders(session, project.id)[-1]

    assert pilot.visual_beats == 1
    assert pilot.semantic_visual_beats == 2
    assert pilot.generation_scope["type"] == "FIRST_BEATS"
    assert Path(pilot.final_mp4).is_file()
    assert pilot_render.generation_scope_type == "FIRST_BEATS"
    assert project.pilot_video_path == pilot.final_mp4
    assert {
        item.beat_id
        for item in list_beat_visual_results(session, project.id, accepted_only=True)
    } == {"beat_1"}
    assert len(list_master_scene_assets(session, project.id)) == 1

    full = asyncio.run(
        run_project_video_pipeline(
            session,
            project.id,
            dependencies,
            job_id="full-run",
        )
    )
    full_render = list_project_video_renders(session, project.id)[-1]

    assert full.generation_scope["type"] == "FULL"
    assert full.visual_beats == full.semantic_visual_beats == 2
    assert provider.calls == calls_after_pilot + 2
    assert full.reused["accepted_visual_assets"] >= 1
    assert full.final_render_id != pilot.final_render_id
    assert full_render.render_revision != pilot_render.render_revision
    assert full_render.generation_scope_type == "FULL"
    assert project.story_text == original_story
    assert len(list_project_narration_assets(session, project.id)) == 1
    pilot_usage = list(
        session.scalars(
            select(ProviderUsageRecord).where(ProviderUsageRecord.job_id == "pilot-run")
        )
    )
    full_usage = list(
        session.scalars(
            select(ProviderUsageRecord).where(ProviderUsageRecord.job_id == "full-run")
        )
    )
    assert pilot_usage and full_usage
    assert any(item.status == "CACHED" for item in full_usage)
