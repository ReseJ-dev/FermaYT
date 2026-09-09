"""Tests for production provider selection and credential routing."""

from pathlib import Path

from app.clients.kimi_ai import KimiVisualPlanningClient
from app.database import create_session_factory, create_sqlite_engine, init_database
from app.repositories import create_project, update_project
from app.secret_store import BYTEPLUS_API_KEY, ELEVENLABS_API_KEY, KIMI_API_KEY
from app.services.pipeline_production import build_production_pipeline_dependencies


class FakeSecretStore:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    def get_secret(self, name: str) -> str | None:
        return self.values.get(name)


def test_kimi_planning_does_not_require_dashscope_when_qa_is_disabled(
    tmp_path: Path,
) -> None:
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
    engine.dispose()
