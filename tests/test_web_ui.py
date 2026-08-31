"""Integration tests for the local project and scene web interface."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.database import (
    create_session_factory,
    create_sqlite_engine,
    init_database,
)
from app.repositories import get_project, list_scenes


@pytest.fixture
def web_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, object, Path]]:
    engine = create_sqlite_engine(tmp_path / "web.db")
    init_database(engine)
    session_factory = create_session_factory(engine)
    projects_root = (tmp_path / "projects").resolve()
    projects_root.mkdir()
    monkeypatch.setattr(main_module, "SessionLocal", session_factory)
    monkeypatch.setattr(main_module, "PROJECTS_ROOT", projects_root)
    with TestClient(main_module.app) as client:
        yield client, session_factory, projects_root
    engine.dispose()


def _create_project(client: TestClient) -> str:
    response = client.post(
        "/projects",
        data={
            "name": "Magic Book",
            "story_text": "A boy discovers a magic book",
            "scene_count": "3",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    return response.headers["location"].rsplit("/", 1)[-1]


def test_dashboard_creates_and_opens_project(web_app: tuple) -> None:
    client, session_factory, _ = web_app

    empty_page = client.get("/")
    project_id = _create_project(client)
    dashboard = client.get("/")
    editor = client.get(f"/projects/{project_id}")

    assert "Создайте первый проект" in empty_page.text
    assert "Magic Book" in dashboard.text
    assert "История и стиль" in editor.text
    assert "Сцены" in editor.text
    with session_factory() as session:
        project = get_project(session, project_id)
        assert project is not None
        assert project.scene_count == 3


def test_project_settings_can_be_updated(web_app: tuple) -> None:
    client, session_factory, _ = web_app
    project_id = _create_project(client)

    response = client.post(
        f"/projects/{project_id}",
        data={
            "name": "Updated story",
            "story_text": "New story text",
            "global_image_style_prompt": "  paper cut art  ",
            "scene_count": "5",
            "image_provider": "qwen",
            "image_model": "qwen-image-3.0",
            "tts_provider": "qwen",
            "tts_model": "qwen3-tts-flash",
            "tts_voice": "Cherry",
            "tts_language": "Russian",
            "output_preset": "horizontal",
            "fps": "24",
            "image_fit": "contain",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    with session_factory() as session:
        project = get_project(session, project_id)
        assert project is not None
        assert project.name == "Updated story"
        assert project.global_image_style_prompt == "paper cut art"
        assert (project.width, project.height, project.fps) == (1920, 1080, 24)


def test_scene_crud_and_reordering_from_ui(web_app: tuple) -> None:
    client, session_factory, _ = web_app
    project_id = _create_project(client)
    for number in range(1, 4):
        response = client.post(
            f"/projects/{project_id}/scenes",
            data={
                "text": f"Narration {number}",
                "image_prompt": f"Image {number}",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

    with session_factory() as session:
        scenes = list_scenes(session, project_id)
        last_id = scenes[-1].id
        middle_id = scenes[1].id

    move = client.post(
        f"/projects/{project_id}/scenes/{last_id}/move/up",
        follow_redirects=False,
    )
    update = client.post(
        f"/projects/{project_id}/scenes/{last_id}",
        data={"text": "Moved narration", "image_prompt": "Moved image"},
        follow_redirects=False,
    )
    delete = client.post(
        f"/projects/{project_id}/scenes/{middle_id}/delete",
        follow_redirects=False,
    )

    assert move.status_code == update.status_code == delete.status_code == 303
    with session_factory() as session:
        scenes = list_scenes(session, project_id)
        assert [scene.position for scene in scenes] == [0, 1]
        assert scenes[1].id == last_id
        assert scenes[1].text == "Moved narration"


def test_project_media_is_served_without_path_traversal(web_app: tuple) -> None:
    client, _, projects_root = web_app
    project_id = _create_project(client)
    media = projects_root / project_id / "images" / "scene.png"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"png-data")
    outside = projects_root.parent / "private.txt"
    outside.write_text("secret", encoding="utf-8")

    response = client.get(f"/media/{project_id}/images/scene.png")
    traversal = client.get(f"/media/{project_id}/%2E%2E/%2E%2E/private.txt")

    assert response.status_code == 200
    assert response.content == b"png-data"
    assert traversal.status_code == 404


def test_settings_never_render_secret_values(
    web_app: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _ = web_app
    monkeypatch.setenv("BYTEPLUS_ARK_API_KEY", "byteplus-secret")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-secret")

    response = client.get("/settings")

    assert response.status_code == 200
    assert "Настроено" in response.text
    assert "byteplus-secret" not in response.text
    assert "dashscope-secret" not in response.text
