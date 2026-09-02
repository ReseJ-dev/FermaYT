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
from app.jobs import GenerationJobManager
from app.repositories import (
    create_scene,
    get_project,
    list_scenes,
    update_project,
)
from app.secret_store import (
    BYTEPLUS_API_KEY,
    DASHSCOPE_API_KEY,
    ELEVENLABS_API_KEY,
)


class FakeSecretStore:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def set_secret(self, name: str, value: str) -> None:
        self.values[name] = value

    def get_secret(self, name: str) -> str | None:
        return self.values.get(name)

    def has_secret(self, name: str) -> bool:
        return name in self.values

    def delete_secret(self, name: str) -> None:
        self.values.pop(name, None)


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
    monkeypatch.setattr(
        main_module,
        "job_manager",
        GenerationJobManager(tmp_path / "jobs.db"),
    )
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
    assert "Сгенерировать видео" in editor.text
    assert "Visual Director" in editor.text
    assert "Manual / Legacy tools" in editor.text
    with session_factory() as session:
        project = get_project(session, project_id)
        assert project is not None
        assert project.scene_count == 3


def test_dashboard_saves_permanent_image_prompt(web_app: tuple) -> None:
    client, session_factory, _ = web_app

    response = client.post(
        "/projects",
        data={
            "name": "Styled story",
            "story_text": "Story text",
            "global_image_style_prompt": "  paper art, warm colors  ",
        },
        follow_redirects=False,
    )
    project_id = response.headers["location"].rsplit("/", 1)[-1]

    assert response.status_code == 303
    with session_factory() as session:
        project = get_project(session, project_id)
        assert project is not None
        assert project.global_image_style_prompt == "paper art, warm colors"

    editor = client.get(f"/projects/{project_id}")
    assert "Постоянный промпт для всех изображений" in editor.text
    assert "paper art, warm colors" in editor.text


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


def test_settings_save_preserve_and_delete_api_keys(
    web_app: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _ = web_app
    store = FakeSecretStore()
    monkeypatch.setattr(main_module, "secret_store", store)

    saved = client.post(
        "/settings",
        data={
            "byteplus_api_key": "byteplus-private",
            "dashscope_api_key": "dashscope-private",
            "elevenlabs_api_key": "elevenlabs-private",
        },
        follow_redirects=False,
    )
    page = client.get("/settings")
    preserved = client.post(
        "/settings",
        data={"byteplus_api_key": "", "dashscope_api_key": ""},
        follow_redirects=False,
    )
    deleted = client.post(
        "/settings",
        data={"delete_byteplus_api_key": "on"},
        follow_redirects=False,
    )

    assert saved.status_code == preserved.status_code == deleted.status_code == 303
    assert store.values == {
        DASHSCOPE_API_KEY: "dashscope-private",
        ELEVENLABS_API_KEY: "elevenlabs-private",
    }
    assert "byteplus-private" not in page.text
    assert "dashscope-private" not in page.text
    assert "elevenlabs-private" not in page.text
    assert page.text.count("Настроено") >= 3
    assert BYTEPLUS_API_KEY not in store.values


def test_global_provider_selection_is_visible_and_used_for_new_project(
    web_app: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session_factory, _ = web_app
    monkeypatch.setattr(main_module, "secret_store", FakeSecretStore())

    settings_page = client.get("/settings")
    saved = client.post(
        "/settings",
        data={
            "default_image_provider": "qwen",
            "default_tts_provider": "elevenlabs",
            "qwen_image_endpoint": "https://example.com/qwen",
        },
        follow_redirects=False,
    )
    project_id = _create_project(client)

    assert "Генерация изображения" in settings_page.text
    assert "Генерация озвучки" in settings_page.text
    assert "BytePlus · Seedream" in settings_page.text
    assert "ElevenLabs" in settings_page.text
    assert saved.status_code == 303
    with session_factory() as session:
        project = get_project(session, project_id)
        assert project is not None
        assert project.image_provider == "qwen"
        assert project.image_model == "qwen-image-3.0"
        assert project.tts_provider == "elevenlabs"
        assert project.tts_model == "eleven_multilingual_v2"


def test_project_can_select_elevenlabs_and_generate_scene_audio(
    web_app: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session_factory, projects_root = web_app
    project_id = _create_project(client)
    store = FakeSecretStore()
    store.set_secret(ELEVENLABS_API_KEY, "elevenlabs-private")
    monkeypatch.setattr(main_module, "secret_store", store)

    with session_factory() as session:
        update_project(
            session,
            project_id,
            tts_provider="elevenlabs",
            tts_model="eleven_multilingual_v2",
            tts_voice="voice-123",
        )
        scene = create_scene(
            session,
            project_id=project_id,
            text="Текст для озвучки",
            image_prompt="Изображение",
        )
        scene_id = scene.id

    provider_call: tuple[str, dict[str, str]] | None = None

    def fake_provider(name: str, config: dict[str, str]) -> object:
        nonlocal provider_call
        provider_call = (name, config)
        return object()

    async def fake_generate_voice(
        text: str,
        output_path: str,
        client: object,
    ) -> str:
        del client
        assert text == "Обновлённый текст для озвучки"
        path = Path(output_path)
        path.write_bytes(b"mp3")
        return output_path

    monkeypatch.setattr(main_module, "get_tts_provider", fake_provider)
    monkeypatch.setattr(main_module, "generate_voice", fake_generate_voice)
    monkeypatch.setattr(main_module, "get_media_duration", lambda path: 2.75)

    response = client.post(
        f"/api/projects/{project_id}/scenes/{scene_id}/generate-audio",
        data={
            "text": "Обновлённый текст для озвучки",
            "image_prompt": "Обновлённое изображение",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert provider_call == (
        "elevenlabs",
        {
            "api_key": "elevenlabs-private",
            "model": "eleven_multilingual_v2",
            "voice": "voice-123",
        },
    )
    with session_factory() as session:
        saved_scene = list_scenes(session, project_id)[0]
        assert saved_scene.audio_path == str(
            projects_root / project_id / "audio" / f"{scene_id}.mp3"
        )
        assert saved_scene.duration == 2.75
        assert saved_scene.text == "Обновлённый текст для озвучки"

    page = client.get(f"/projects/{project_id}")
    assert "ElevenLabs" in page.text
    assert "Перегенерировать голос" in page.text


def test_selected_image_provider_receives_scene_and_global_style(
    web_app: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session_factory, projects_root = web_app
    project_id = _create_project(client)
    store = FakeSecretStore()
    store.set_secret(DASHSCOPE_API_KEY, "dashscope-private")
    monkeypatch.setattr(main_module, "secret_store", store)
    monkeypatch.setenv("QWEN_IMAGE_ENDPOINT", "https://qwen.example/image")

    with session_factory() as session:
        update_project(
            session,
            project_id,
            image_provider="qwen",
            image_model="qwen-image-3.0",
            global_image_style_prompt="layered cardboard, warm colors",
        )
        scene = create_scene(
            session,
            project_id=project_id,
            text="Scene narration",
            image_prompt="Old image prompt",
        )
        scene_id = scene.id

    provider_call: tuple[str, dict[str, str]] | None = None
    generation_call: tuple[str, str] | None = None

    def fake_provider(name: str, config: dict[str, str]) -> object:
        nonlocal provider_call
        provider_call = (name, config)
        return object()

    async def fake_generate_image(
        prompt: str,
        output_path: str,
        client: object,
    ) -> str:
        nonlocal generation_call
        del client
        generation_call = (prompt, output_path)
        path = Path(output_path)
        path.write_bytes(b"png")
        return output_path

    monkeypatch.setattr(main_module, "get_image_provider", fake_provider)
    monkeypatch.setattr(main_module, "generate_image", fake_generate_image)

    response = client.post(
        f"/api/projects/{project_id}/scenes/{scene_id}/generate-image",
        data={
            "text": "Updated narration",
            "image_prompt": "A boy opening a magic book",
        },
        follow_redirects=False,
    )

    expected_path = str(projects_root / project_id / "images" / f"{scene_id}.png")
    assert response.status_code == 303
    assert provider_call == (
        "qwen",
        {
            "api_key": "dashscope-private",
            "endpoint": "https://qwen.example/image",
            "model": "qwen-image-3.0",
        },
    )
    assert generation_call == (
        "A boy opening a magic book\n\nlayered cardboard, warm colors",
        expected_path,
    )
    with session_factory() as session:
        saved_scene = list_scenes(session, project_id)[0]
        assert saved_scene.image_path == expected_path
        assert saved_scene.image_prompt == "A boy opening a magic book"

    page = client.get(f"/projects/{project_id}")
    assert "Alibaba · Qwen Image" in page.text
    assert "Перегенерировать" in page.text


def test_generate_video_preflight_rejects_missing_keys_without_starting_job(
    web_app: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _ = web_app
    project_id = _create_project(client)
    monkeypatch.setattr(main_module, "secret_store", FakeSecretStore())
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("BYTEPLUS_ARK_API_KEY", raising=False)

    response = client.post(f"/api/projects/{project_id}/generate-video")

    assert response.status_code == 422
    assert "DASHSCOPE_API_KEY" in response.json()["detail"]


def test_style_reference_upload_is_png_only_and_immutable(web_app: tuple) -> None:
    client, session_factory, projects_root = web_app
    project_id = _create_project(client)
    first_png = b"\x89PNG\r\n\x1a\napproved-style"
    another_png = b"\x89PNG\r\n\x1a\ndifferent-style"

    wrong_type = client.post(
        f"/api/projects/{project_id}/style-reference",
        content=first_png,
        headers={"content-type": "image/jpeg"},
    )
    registered = client.post(
        f"/api/projects/{project_id}/style-reference",
        content=first_png,
        headers={"content-type": "image/png"},
    )
    replacement = client.post(
        f"/api/projects/{project_id}/style-reference",
        content=another_png,
        headers={"content-type": "image/png"},
    )

    assert wrong_type.status_code == 415
    assert registered.status_code == 200
    assert replacement.status_code == 422
    with session_factory() as session:
        project = get_project(session, project_id)
        assert project is not None
        reference = main_module.get_style_reference_asset(
            session, project_id, project.style_id
        )
        assert reference is not None
        assert Path(reference.file_path).read_bytes() == first_png
        assert Path(reference.file_path).is_relative_to(projects_root)
