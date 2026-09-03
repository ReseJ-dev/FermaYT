"""Tests for immutable master generation and descendant continuity."""

import asyncio
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

import app.generators.master_scene as master_scene_module
from app.clients.image_api import ImageGenerationError
from app.database import create_session_factory, create_sqlite_engine, init_database
from app.errors import ImagePromptBuildError, MasterSceneError
from app.generators.image_prompt import ImagePromptBuilder
from app.generators.master_scene import (
    build_continuity_generation_request,
    build_structured_continuity_generation_request,
    build_style_version,
    generate_continuity_image,
    generate_continuity_image_with_qa,
    generate_required_master_scenes,
)
from app.generators.style_reference import register_approved_style_reference
from app.models.visual_plan import VisualOperation, VisualPlan
from app.models.visual_qa import VisualQADecision
from app.pipeline.visual_operation_engine import VisualProviderCapabilities
from app.pipeline.visual_qa import VisualQAContext
from app.provider_diagnostics import ImageProviderDiagnostic
from app.providers import ImageReference, ImageReferenceRole
from app.repositories import (
    create_master_scene_asset,
    create_project,
    get_master_scene_asset,
    list_master_scene_assets,
)


@pytest.fixture
def session(tmp_path: Path) -> Iterator[Session]:
    engine = create_sqlite_engine(tmp_path / "app.db")
    init_database(engine)
    session_factory = create_session_factory(engine)
    with session_factory() as database_session:
        yield database_session
    engine.dispose()


def _plan() -> VisualPlan:
    master = {
        "id": "shaft_master",
        "location_id": "shaft",
        "description": "Full cutaway of the recurring mine shaft",
        "environment_geometry": "Vertical shaft, surface above, side tunnel below",
        "recurring_object_positions": "Ladder right, lift cable center, tunnel left",
        "color_palette": "Charcoal rock, amber lights, muted blue equipment",
        "basic_composition": "Surface at top and workers small at lower left",
        "characters_visible": ["miners"],
        "important_objects": ["ladder"],
    }
    unused_master = {
        **master,
        "id": "unused_master",
        "description": "An environment not used by any beat",
    }
    return VisualPlan.model_validate(
        {
            "story_summary": "A mine escape route fails",
            "visual_strategy": "Preserve the shaft while its state changes",
            "characters": [
                {
                    "id": "miners",
                    "name": "Miners",
                    "description": "Workers trapped below",
                }
            ],
            "locations": [
                {
                    "id": "shaft",
                    "name": "Main shaft",
                    "description": "Deep vertical mine shaft",
                    "spatial_layout": "Surface above and tunnel below",
                }
            ],
            "important_objects": [
                {
                    "id": "ladder",
                    "name": "Ladder",
                    "description": "Main escape ladder",
                    "story_function": "Connects workers to safety",
                }
            ],
            "recurring_environments": [
                {
                    "id": "mine_environment",
                    "location_id": "shaft",
                    "continuity_requirements": "Keep routes and anchors fixed",
                }
            ],
            "possible_master_scenes": [master, unused_master],
            "visual_beats": [
                {
                    "id": "beat_1",
                    "narration_segment": "The miners worked far below.",
                    "visual_purpose": "Establish geography",
                    "what_viewer_should_understand": "The exit is far above",
                    "location_id": "shaft",
                    "characters_visible": ["miners"],
                    "important_objects": ["ladder"],
                    "camera_framing": "WIDE",
                    "camera_view": "Full shaft cutaway",
                    "framing_reason": "Show the complete route",
                    "camera_movement": "SUBTLE_ZOOM",
                    "geography_established_by": "shaft_master",
                    "master_scene_id": "shaft_master",
                    "physical_state": "The ladder is intact",
                    "progressive_change": None,
                    "safety_geography": None,
                    "anticipated_consequence": None,
                    "change_from_previous_beat": "Initial state",
                    "information_added_beyond_narration": "Shows distance to exit",
                    "preferred_visual_operation": "NEW_IMAGE",
                    "source_visual_id": None,
                    "overlay_description": None,
                    "estimated_duration_seconds": 6,
                }
            ],
        }
    )


def _project(session: Session):
    return create_project(
        session,
        name="Mine story",
        story_text="A mine escape route fails",
        global_image_style_prompt="Layered cutaway illustration",
        image_provider="qwen",
        image_model="qwen-image-3.0",
    )


def _install_fake_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    async def fake_generate_image(
        prompt: str,
        output_path: str,
        client: object,
        *,
        style_id: str,
    ) -> str:
        del client
        assert style_id == "rough_explainer_v1"
        calls.append((prompt, output_path))
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"immutable-master-image")
        return output_path

    monkeypatch.setattr(master_scene_module, "generate_image", fake_generate_image)
    return calls


def test_generates_only_referenced_masters_and_persists_metadata(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(session)
    calls = _install_fake_generation(monkeypatch)

    assets = asyncio.run(
        generate_required_master_scenes(
            session,
            project,
            _plan(),
            client=object(),  # type: ignore[arg-type]
            projects_root=tmp_path / "projects",
        )
    )

    assert len(assets) == 1
    asset = assets[0]
    assert asset.master_scene_id == "shaft_master"
    assert Path(asset.file_path).is_file()
    assert len(asset.file_sha256) == 64
    assert asset.style_version == build_style_version()
    assert asset.provider == "qwen"
    assert asset.model == "qwen-image-3.0"
    assert asset.seed is None
    assert asset.reference_hashes == []
    assert "Stable geometry:" in asset.generation_prompt
    assert "Overall color palette:" in asset.generation_prompt
    assert "STYLE CONTRACT [rough_explainer_v1]" in asset.generation_prompt
    assert len(calls) == 1
    assert [item.master_scene_id for item in list_master_scene_assets(session, project.id)] == [
        "shaft_master"
    ]


def test_master_provider_failure_adds_safe_master_context_and_logs_it(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    project = _project(session)

    async def fail_generation(*args: object, **kwargs: object) -> str:
        del args, kwargs
        raise ImageGenerationError(
            "Qwen Image API returned HTTP 400",
            diagnostic=ImageProviderDiagnostic(
                provider="qwen",
                model="qwen-image-3.0",
                operation="generate",
                error_type="http",
                request_stage="provider_response",
                http_status=400,
                provider_error='{"code":"InvalidParameter","message":"Invalid size"}',
            ),
        )

    monkeypatch.setattr(master_scene_module, "generate_image", fail_generation)

    with caplog.at_level(logging.ERROR), pytest.raises(MasterSceneError) as raised:
        asyncio.run(
            generate_required_master_scenes(
                session,
                project,
                _plan(),
                client=object(),  # type: ignore[arg-type]
                projects_root=tmp_path / "projects",
            )
        )

    assert str(raised.value) == "Failed to generate master scene shaft_master"
    diagnostic = raised.value.safe_diagnostic
    assert isinstance(diagnostic, ImageProviderDiagnostic)
    assert diagnostic.master_scene_id == "shaft_master"
    assert diagnostic.request_stage == "master_scene_generation"
    assert diagnostic.http_status == 400
    assert "master_scene_id=\"shaft_master\"" in caplog.text
    assert "Invalid size" in caplog.text


def test_existing_master_is_reused_and_never_regenerated(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(session)
    calls = _install_fake_generation(monkeypatch)
    first = asyncio.run(
        generate_required_master_scenes(
            session,
            project,
            _plan(),
            client=object(),  # type: ignore[arg-type]
            projects_root=tmp_path / "projects",
        )
    )
    second = asyncio.run(
        generate_required_master_scenes(
            session,
            project,
            _plan(),
            client=object(),  # type: ignore[arg-type]
            projects_root=tmp_path / "projects",
        )
    )

    assert first[0].id == second[0].id
    assert len(calls) == 1


def test_style_drift_does_not_replace_existing_master(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(session)
    calls = _install_fake_generation(monkeypatch)
    asyncio.run(
        generate_required_master_scenes(
            session,
            project,
            _plan(),
            client=object(),  # type: ignore[arg-type]
            projects_root=tmp_path / "projects",
        )
    )
    project.global_image_style_prompt = "A completely different visual style"

    with pytest.raises(MasterSceneError, match="explicit new master version"):
        asyncio.run(
            generate_required_master_scenes(
                session,
                project,
                _plan(),
                client=object(),  # type: ignore[arg-type]
                projects_root=tmp_path / "projects",
            )
        )

    assert len(calls) == 1


def test_tampered_master_fails_integrity_check(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(session)
    _install_fake_generation(monkeypatch)
    asset = asyncio.run(
        generate_required_master_scenes(
            session,
            project,
            _plan(),
            client=object(),  # type: ignore[arg-type]
            projects_root=tmp_path / "projects",
        )
    )[0]
    Path(asset.file_path).write_bytes(b"changed-frame")

    with pytest.raises(MasterSceneError, match="integrity"):
        asyncio.run(
            generate_required_master_scenes(
                session,
                project,
                _plan(),
                client=object(),  # type: ignore[arg-type]
                projects_root=tmp_path / "projects",
            )
        )


def test_master_database_record_cannot_be_updated(
    session: Session,
    tmp_path: Path,
) -> None:
    project = _project(session)
    image_path = tmp_path / "master.png"
    image_path.write_bytes(b"master")
    asset = create_master_scene_asset(
        session,
        project_id=project.id,
        master_scene_id="shaft_master",
        file_path=str(image_path),
        file_sha256=master_scene_module._sha256_file(image_path),
        style_version="v1",
        generation_prompt="prompt",
        provider="qwen",
    )
    asset.provider = "replacement-provider"

    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()
    stored = get_master_scene_asset(session, project.id, "shaft_master")
    assert stored is not None
    assert stored.provider == "qwen"


def test_text_only_provider_receives_structured_master_description(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(session)
    _install_fake_generation(monkeypatch)
    plan = _plan()
    asset = asyncio.run(
        generate_required_master_scenes(
            session,
            project,
            plan,
            client=object(),  # type: ignore[arg-type]
            projects_root=tmp_path / "projects",
        )
    )[0]

    request = build_continuity_generation_request(
        plan,
        plan.visual_beats[0],
        VisualOperation.NEW_IMAGE,
        "Miners look toward the damaged ladder",
        {asset.master_scene_id: asset},
        VisualProviderCapabilities(),
    )

    assert request.reference_image_paths == ()
    assert "IMMUTABLE MASTER SCENE CONTINUITY" in request.prompt
    assert "environment_geometry:" in request.prompt
    assert "Miners look toward the damaged ladder" in request.prompt


def test_reference_provider_receives_verified_master_image(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(session)
    _install_fake_generation(monkeypatch)
    plan = _plan()
    asset = asyncio.run(
        generate_required_master_scenes(
            session,
            project,
            plan,
            client=object(),  # type: ignore[arg-type]
            projects_root=tmp_path / "projects",
        )
    )[0]

    request = build_continuity_generation_request(
        plan,
        plan.visual_beats[0],
        VisualOperation.REFERENCE_GENERATION,
        "New view inside the same shaft",
        {asset.master_scene_id: asset},
        VisualProviderCapabilities(reference_generation=True),
    )

    assert request.reference_image_paths == (asset.file_path,)
    assert request.reference_hashes == (asset.file_sha256,)
    assert request.master_scene_id == "shaft_master"
    assert "IMMUTABLE MASTER SCENE CONTINUITY" not in request.prompt
    assert "STYLE CONTRACT [rough_explainer_v1]" in request.prompt


def test_continuity_executor_passes_references_to_capable_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReferenceClient:
        received: tuple[str, tuple[ImageReference, ...]] | None = None

        async def generate(self, prompt: str) -> str:
            raise AssertionError("text-only generation must not be used")

        async def generate_with_references(
            self,
            prompt: str,
            references: tuple[ImageReference, ...],
        ) -> str:
            self.received = (prompt, references)
            return "https://example.com/frame.png"

    async def fake_download(url: str, output_path: str) -> str:
        assert url == "https://example.com/frame.png"
        return output_path

    monkeypatch.setattr(master_scene_module, "download_file", fake_download)
    client = ReferenceClient()
    request = master_scene_module.ContinuityGenerationRequest(
        operation=VisualOperation.REFERENCE_GENERATION,
        prompt="Same environment, new composition",
        master_scene_id="shaft_master",
        master_image_path="master.png",
        references=(
            ImageReference(
                reference_id="shaft_master",
                file_path="master.png",
                sha256="0" * 64,
                role=ImageReferenceRole.CONTENT_CONTINUITY,
            ),
        ),
    )

    result = asyncio.run(generate_continuity_image(request, "frame.png", client))

    assert result == "frame.png"
    assert client.received is not None
    assert client.received[0].startswith("Same environment, new composition")
    assert "STYLE CONTRACT [rough_explainer_v1]" in client.received[0]
    assert tuple(reference.file_path for reference in client.received[1]) == (
        "master.png",
    )


def test_style_reference_precedes_master_continuity_reference(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(session)
    _install_fake_generation(monkeypatch)
    plan = _plan()
    master_asset = asyncio.run(
        generate_required_master_scenes(
            session,
            project,
            plan,
            client=object(),  # type: ignore[arg-type]
            projects_root=tmp_path / "projects",
        )
    )[0]
    style_source = tmp_path / "STYLE_REFERENCE.png"
    style_source.write_bytes(b"\x89PNG\r\n\x1a\napproved-style")
    style_asset = register_approved_style_reference(
        session,
        project.id,
        style_source,
        projects_root=tmp_path / "projects",
    )

    request = build_continuity_generation_request(
        plan,
        plan.visual_beats[0],
        VisualOperation.REFERENCE_GENERATION,
        "A new composition in the same shaft",
        {master_asset.master_scene_id: master_asset},
        VisualProviderCapabilities(reference_generation=True),
        style_reference=style_asset,
    )

    assert [reference.role for reference in request.references] == [
        ImageReferenceRole.STYLE,
        ImageReferenceRole.CONTENT_CONTINUITY,
    ]
    assert request.references[0].reference_id == "rough_explainer_v1"
    assert request.references[1].reference_id == "shaft_master"
    assert "REFERENCE 1 [STYLE]" in request.prompt
    assert "REFERENCE 2 [CONTENT_CONTINUITY]" in request.prompt


def test_master_generation_attaches_approved_style_reference(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(session)
    style_source = tmp_path / "STYLE_REFERENCE.png"
    style_source.write_bytes(b"\x89PNG\r\n\x1a\napproved-style")
    style_asset = register_approved_style_reference(
        session,
        project.id,
        style_source,
        projects_root=tmp_path / "projects",
    )

    class ReferenceClient:
        references: tuple[ImageReference, ...] = ()
        prompt: str | None = None

        async def generate(self, prompt: str) -> str:
            raise AssertionError("text-only generation must not be used")

        async def generate_with_references(
            self,
            prompt: str,
            references: tuple[ImageReference, ...],
        ) -> str:
            self.prompt = prompt
            self.references = references
            return "https://example.com/master.png"

    async def fake_download(url: str, output_path: str) -> str:
        assert url == "https://example.com/master.png"
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"generated-master")
        return output_path

    monkeypatch.setattr(master_scene_module, "download_file", fake_download)
    client = ReferenceClient()

    asset = asyncio.run(
        generate_required_master_scenes(
            session,
            project,
            _plan(),
            client,
            projects_root=tmp_path / "projects",
            style_reference=style_asset,
            capabilities=VisualProviderCapabilities(reference_generation=True),
        )
    )[0]

    assert [reference.role for reference in client.references] == [
        ImageReferenceRole.STYLE
    ]
    assert client.prompt is not None
    assert "REFERENCE 1 [STYLE]" in client.prompt
    assert asset.reference_hashes == [style_asset.file_sha256]


def test_continuity_generation_feeds_qa_correction_into_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Client:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return f"https://example.com/candidate-{len(self.prompts)}.png"

    async def fake_download(url: str, output_path: str) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(url.encode())
        return output_path

    class QA:
        calls = 0

        async def evaluate(
            self,
            image_path: str,
            context: VisualQAContext,
        ) -> VisualQADecision:
            del image_path, context
            self.calls += 1
            if self.calls == 1:
                return VisualQADecision(
                    result="REGENERATE",
                    problem_categories=["COMPOSITION"],
                    reasons=["The broken ladder is too small"],
                    correction_instruction="Crop closer to the broken ladder",
                )
            return VisualQADecision(
                result="PASS",
                problem_categories=[],
                reasons=[],
                correction_instruction=None,
            )

    monkeypatch.setattr(master_scene_module, "download_file", fake_download)
    client = Client()
    request = master_scene_module.ContinuityGenerationRequest(
        operation=VisualOperation.NEW_IMAGE,
        prompt="A blocked mine shaft",
        style_version="rough_explainer_v1",
    )
    context = VisualQAContext(
        visual_purpose="Show the blocked route",
        what_viewer_should_understand="The ladder is unusable",
        required_objects=("ladder",),
        important_physical_action="The ladder broke",
        location_id="shaft",
        expected_physical_state="Broken ladder",
    )

    outcome = asyncio.run(
        generate_continuity_image_with_qa(
            request,
            str(tmp_path / "final.png"),
            client,
            context,
            QA(),  # type: ignore[arg-type]
            max_retries=2,
        )
    )

    assert outcome.attempts == 2
    assert len(client.prompts) == 2
    assert "VISUAL QA CORRECTION FOR REGENERATION" in client.prompts[1]
    assert "Crop closer to the broken ladder" in client.prompts[1]


def test_master_generation_is_qa_checked_and_keeps_correction_provenance(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(session)
    generation_calls = _install_fake_generation(monkeypatch)

    class QA:
        def __init__(self) -> None:
            self.calls = 0

        async def evaluate(
            self,
            image_path: str,
            context: VisualQAContext,
        ) -> VisualQADecision:
            del image_path, context
            self.calls += 1
            if self.calls == 1:
                return VisualQADecision(
                    result="REGENERATE",
                    problem_categories=["STYLE_DRIFT"],
                    reasons=["Mine walls contain realistic texture"],
                    correction_instruction="Remove rock texture and simplify the walls",
                )
            return VisualQADecision(
                result="PASS",
                problem_categories=[],
                reasons=[],
                correction_instruction=None,
            )

    qa = QA()
    asset = asyncio.run(
        generate_required_master_scenes(
            session,
            project,
            _plan(),
            client=object(),  # type: ignore[arg-type]
            projects_root=tmp_path / "projects",
            qa_service=qa,  # type: ignore[arg-type]
            max_qa_retries=1,
        )
    )[0]

    assert qa.calls == 2
    assert len(generation_calls) == 2
    assert "VISUAL QA CORRECTION FOR REGENERATION" in asset.generation_prompt
    assert "Remove rock texture and simplify the walls" in asset.generation_prompt

    unused_qa = QA()
    reused = asyncio.run(
        generate_required_master_scenes(
            session,
            project,
            _plan(),
            client=object(),  # type: ignore[arg-type]
            projects_root=tmp_path / "projects",
            qa_service=unused_qa,  # type: ignore[arg-type]
            max_qa_retries=1,
        )
    )[0]

    assert reused.id == asset.id
    assert unused_qa.calls == 0
    assert len(generation_calls) == 2


def test_image_prompt_builder_uses_semantics_and_never_narration() -> None:
    plan = _plan()
    beat = plan.visual_beats[0].model_copy(
        update={
            "narration_segment": "UNIQUE NARRATION MUST NEVER REACH IMAGE API",
            "visual_focus": "The blocked vertical escape route",
            "must_not_show": ["injured people", "a different mine layout"],
        }
    )

    prompt = ImagePromptBuilder().build(
        plan,
        beat,
        VisualOperation.EDIT_EXISTING,
    )

    assert "UNIQUE NARRATION MUST NEVER REACH IMAGE API" not in prompt
    assert "Same shaft_master environment" in prompt
    assert "Vertical shaft, surface above, side tunnel below" in prompt
    assert "Ladder: Main escape ladder" in prompt
    assert "VISUAL FOCUS:" in prompt
    assert "First notice: The blocked vertical escape route" in prompt
    assert "DO NOT SHOW:\ninjured people; a different mine layout" in prompt
    assert "clearly advances the story" in prompt
    assert "only decorates the narration" in prompt
    assert prompt.rstrip().endswith(
        "This contract overrides any conflicting style instruction elsewhere in the request."
    )


def test_image_prompt_sections_have_concise_semantic_order() -> None:
    prompt = ImagePromptBuilder().build(
        _plan(),
        _plan().visual_beats[0],
        VisualOperation.NEW_IMAGE,
    )
    headings = [
        "REFERENCE INSTRUCTIONS:",
        "LOCATION CONTINUITY:",
        "CHARACTER CONTINUITY:",
        "OBJECT CONTINUITY:",
        "CURRENT CAMERA / COMPOSITION:",
        "CURRENT PHYSICAL STATE:",
        "WHAT CHANGED:",
        "SIMPLIFICATION RULE:",
        "STYLE CONTRACT [rough_explainer_v1]",
    ]

    positions = [prompt.index(heading) for heading in headings]
    assert positions == sorted(positions)


def test_prompt_builder_rejects_semantic_instruction_overload() -> None:
    with pytest.raises(ImagePromptBuildError, match="too long"):
        ImagePromptBuilder(max_semantic_characters=50).build(
            _plan(),
            _plan().visual_beats[0],
            VisualOperation.NEW_IMAGE,
        )


def test_structured_continuity_request_does_not_need_manual_beat_prompt(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(session)
    _install_fake_generation(monkeypatch)
    plan = _plan()
    master_asset = asyncio.run(
        generate_required_master_scenes(
            session,
            project,
            plan,
            client=object(),  # type: ignore[arg-type]
            projects_root=tmp_path / "projects",
        )
    )[0]

    request = build_structured_continuity_generation_request(
        plan,
        plan.visual_beats[0],
        VisualOperation.NEW_IMAGE,
        {master_asset.master_scene_id: master_asset},
        VisualProviderCapabilities(),
    )

    assert plan.visual_beats[0].narration_segment not in request.prompt
    assert "LOCATION CONTINUITY:" in request.prompt
    assert "CURRENT PHYSICAL STATE:" in request.prompt
    assert request.prompt.count("STYLE CONTRACT [rough_explainer_v1]") == 1
