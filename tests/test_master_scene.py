"""Tests for immutable master generation and descendant continuity."""

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

import app.generators.master_scene as master_scene_module
from app.database import create_session_factory, create_sqlite_engine, init_database
from app.errors import MasterSceneError
from app.generators.master_scene import (
    build_continuity_generation_request,
    build_style_version,
    generate_continuity_image,
    generate_required_master_scenes,
)
from app.models.visual_plan import VisualOperation, VisualPlan
from app.pipeline.visual_operation_engine import VisualProviderCapabilities
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
        received: tuple[str, tuple[str, ...]] | None = None

        async def generate(self, prompt: str) -> str:
            raise AssertionError("text-only generation must not be used")

        async def generate_with_references(
            self,
            prompt: str,
            reference_image_paths: tuple[str, ...],
        ) -> str:
            self.received = (prompt, reference_image_paths)
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
        reference_image_paths=("master.png",),
        reference_hashes=("hash",),
    )

    result = asyncio.run(generate_continuity_image(request, "frame.png", client))

    assert result == "frame.png"
    assert client.received is not None
    assert client.received[0].startswith("Same environment, new composition")
    assert "STYLE CONTRACT [rough_explainer_v1]" in client.received[0]
    assert client.received[1] == ("master.png",)
