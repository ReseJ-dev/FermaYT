"""Tests for production provider selection and credential routing."""

from pathlib import Path

import pytest

from app.clients.image_api import KieZImageApiClient
from app.clients.kimi_ai import KimiVisualPlanningClient
from app.database import create_session_factory, create_sqlite_engine, init_database
from app.repositories import create_project, update_project
from app.secret_store import (
    BYTEPLUS_API_KEY,
    ELEVENLABS_API_KEY,
    KIE_API_KEY,
    KIMI_API_KEY,
)
from app.services.pipeline_production import build_production_pipeline_dependencies


class FakeSecretStore:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    def get_secret(self, name: str) -> str | None:
        return self.values.get(name)


def test_kimi_planning_does_not_require_dashscope_when_qa_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VISUAL_PLANNING_TIMEOUT_SECONDS", "720")
    monkeypatch.setenv("VISUAL_PLANNING_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("VISUAL_PLANNING_RETRY_BASE_SECONDS", "0.25")
    monkeypatch.setenv("VISUAL_PLANNING_MAX_OUTPUT_TOKENS", "20000")
    monkeypatch.setenv("VISUAL_PLANNING_REASONING_EFFORT", "high")
    engine = create_sqlite_engine(tmp_path / "production.db")
    init_database(engine)
    factory = create_session_factory(engine)
    store = FakeSecretStore(
        {
            KIMI_API_KEY: "kimi-key",
            BYTEPLUS_API_KEY: "byteplus-key",
            ELEVENLABS_API_KEY: "elevenlabs-key",
        }
    )
    with factory() as session:
        project = create_project(
            session,
            name="Kimi project",
            story_text="Story",
            image_provider="seedream",
            image_model="seedream-5-0-260128",
            tts_provider="elevenlabs",
            tts_model="eleven_multilingual_v2",
        )
        update_project(
            session,
            project.id,
            planning_provider="kimi",
            planning_model="kimi-k3",
            visual_qa_enabled=False,
        )

        dependencies = build_production_pipeline_dependencies(
            session,
            project.id,
            store,  # type: ignore[arg-type]
            projects_root=tmp_path / "projects",
        )

    assert isinstance(dependencies.planning_client, KimiVisualPlanningClient)
    assert dependencies.planning_client.model == "kimi-k3"
    assert dependencies.planning_client.api_key == "kimi-key"
    assert dependencies.planning_client.timeout == 720
    assert dependencies.planning_client.max_attempts == 1
    assert dependencies.planning_client.retry_base_delay == 0
    assert dependencies.planning_client.max_output_tokens == 20_000
    assert dependencies.planning_client.reasoning_effort == "high"
    engine.dispose()


def test_image_only_preview_does_not_require_tts_credentials(tmp_path: Path) -> None:
    engine = create_sqlite_engine(tmp_path / "preview.db")
    init_database(engine)
    factory = create_session_factory(engine)
    store = FakeSecretStore(
        {
            KIMI_API_KEY: "kimi-key",
            BYTEPLUS_API_KEY: "byteplus-key",
        }
    )
    with factory() as session:
        project = create_project(
            session,
            name="Style preview",
            story_text="Story",
            planning_provider="kimi",
            planning_model="kimi-k3",
            image_provider="seedream",
            image_model="seedream-5-0-260128",
            tts_provider="elevenlabs",
            tts_model="eleven_multilingual_v2",
        )
        update_project(session, project.id, visual_qa_enabled=False)

        dependencies = build_production_pipeline_dependencies(
            session,
            project.id,
            store,  # type: ignore[arg-type]
            projects_root=tmp_path / "projects",
            require_tts=False,
        )
        dependencies.preflight_validator(project)  # type: ignore[misc]

    with pytest.raises(RuntimeError, match="image-only"):
        dependencies.tts_provider_resolver("elevenlabs", None)
    engine.dispose()


def test_zimage_provider_uses_kie_key_and_project_aspect_ratio(tmp_path: Path) -> None:
    engine = create_sqlite_engine(tmp_path / "zimage.db")
    init_database(engine)
    factory = create_session_factory(engine)
    store = FakeSecretStore(
        {
            KIMI_API_KEY: "kimi-key",
            KIE_API_KEY: "kie-key",
            ELEVENLABS_API_KEY: "elevenlabs-key",
        }
    )
    with factory() as session:
        project = create_project(
            session,
            name="Z-Image project",
            story_text="Story",
            planning_provider="kimi",
            planning_model="kimi-k3",
            image_provider="zimage",
            image_model="z-image",
            tts_provider="elevenlabs",
            tts_model="eleven_multilingual_v2",
            width=1920,
            height=1080,
        )
        update_project(session, project.id, visual_qa_enabled=False)
        dependencies = build_production_pipeline_dependencies(
            session,
            project.id,
            store,  # type: ignore[arg-type]
            projects_root=tmp_path / "projects",
        )

        provider = dependencies.image_provider_resolver(
            "zimage", {"model": "z-image"}
        )

    assert isinstance(provider, KieZImageApiClient)
    assert provider.api_key == "kie-key"
    assert provider.aspect_ratio == "16:9"
    engine.dispose()
