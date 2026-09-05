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
from sqlalchemy.orm import Session
from test_master_scene import _plan

from app.budgets import BUDGET_ESTIMATE_EXCEEDS_LIMIT, GenerationBudgetError
from app.costs import PricingUnit, configure_provider_pricing
from app.database import create_session_factory, create_sqlite_engine, init_database
from app.jobs import GenerationJobManager, GenerationJobType
from app.models.visual_plan import VisualPlan
from app.models.visual_qa import VisualQADecision
from app.pipeline.visual_qa import VisualQAService
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
            session, provider=provider, model=model, operation=operation,
            pricing_unit=unit, price=price, currency="USD", version="v1",
            effective_from=effective,
        )


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
    dependencies = _dependencies(
        tmp_path, provider, b"unused", b"unused", planning, qa
    )

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
