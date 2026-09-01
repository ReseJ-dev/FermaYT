"""Tests for SQLAlchemy project and scene persistence."""

from collections.abc import Iterator
from datetime import UTC
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy.orm import Session

from app.database import (
    create_session_factory,
    create_sqlite_engine,
    init_database,
)
from app.repositories import (
    create_project,
    create_scene,
    delete_project,
    delete_scene,
    get_application_settings,
    get_project,
    get_scene,
    list_projects,
    list_scenes,
    move_scene,
    update_application_settings,
    update_project,
    update_scene,
)


@pytest.fixture
def session(tmp_path: Path) -> Iterator[Session]:
    engine = create_sqlite_engine(tmp_path / "app.db")
    init_database(engine)
    session_factory = create_session_factory(engine)
    with session_factory() as database_session:
        yield database_session
    engine.dispose()


def make_project(session: Session, **overrides: object):
    values: dict[str, object] = {
        "name": "Magic Book",
        "story_text": "A boy finds a magic book",
    }
    values.update(overrides)
    return create_project(session, **values)  # type: ignore[arg-type]


def test_create_project_with_uuid_and_defaults(session: Session) -> None:
    project = make_project(
        session,
        story_text="  A boy finds a magic book  ",
    )

    assert UUID(project.id).version == 4
    assert project.story_text == "A boy finds a magic book"
    assert project.global_image_style_prompt is None
    assert project.scene_count is None
    assert project.image_provider == "seedream"
    assert project.tts_provider == "qwen"
    assert project.created_at.tzinfo is UTC
    assert project.updated_at.tzinfo is UTC
    assert project.rendered_at is None


def test_application_provider_defaults_are_persisted(session: Session) -> None:
    defaults = get_application_settings(session)

    assert defaults.image_provider == "seedream"
    assert defaults.tts_provider == "qwen"
    assert defaults.qwen_image_endpoint is None

    updated = update_application_settings(
        session,
        image_provider="qwen",
        tts_provider="elevenlabs",
        qwen_image_endpoint="  https://example.com/qwen  ",
    )

    assert updated.image_provider == "qwen"
    assert updated.tts_provider == "elevenlabs"
    assert updated.qwen_image_endpoint == "https://example.com/qwen"
    assert get_application_settings(session).id == updated.id


@pytest.mark.parametrize("story_text", ["", "   ", "\t\n"])
def test_project_rejects_blank_story(
    session: Session,
    story_text: str,
) -> None:
    with pytest.raises(ValueError, match="story_text"):
        make_project(session, story_text=story_text)


@pytest.mark.parametrize("style_prompt", ["", "   ", "\t\n", None])
def test_project_normalizes_empty_style_to_none(
    session: Session,
    style_prompt: str | None,
) -> None:
    project = make_project(
        session,
        global_image_style_prompt=style_prompt,
    )

    assert project.global_image_style_prompt is None


@pytest.mark.parametrize("scene_count", [None, 1, 50])
def test_project_accepts_valid_scene_count(
    session: Session,
    scene_count: int | None,
) -> None:
    project = make_project(session, scene_count=scene_count)

    assert project.scene_count == scene_count


@pytest.mark.parametrize("scene_count", [0, 51])
def test_project_rejects_invalid_scene_count(
    session: Session,
    scene_count: int,
) -> None:
    with pytest.raises(ValueError, match="between 1 and 50"):
        make_project(session, scene_count=scene_count)


def test_project_crud(session: Session) -> None:
    first = make_project(session, name="First")
    second = make_project(session, name="Second")

    assert get_project(session, first.id) is first
    assert [project.id for project in list_projects(session)] == [
        first.id,
        second.id,
    ]

    updated = update_project(
        session,
        first.id,
        name="Updated",
        global_image_style_prompt="  paper art  ",
        scene_count=5,
    )

    assert updated is not None
    assert updated.name == "Updated"
    assert updated.global_image_style_prompt == "paper art"
    assert updated.scene_count == 5
    assert delete_project(session, first.id) is True
    assert get_project(session, first.id) is None
    assert delete_project(session, first.id) is False


def test_create_scene_relationship_and_order(session: Session) -> None:
    project = make_project(session)
    third = create_scene(
        session,
        project_id=project.id,
        text="First narration",
        image_prompt="First image",
    )
    first = create_scene(
        session,
        project_id=project.id,
        text="Second narration",
        image_prompt="Second image",
    )
    second = create_scene(
        session,
        project_id=project.id,
        text="Third narration",
        image_prompt="Third image",
    )

    scenes = list_scenes(session, project.id)

    assert UUID(third.id).version == 4
    assert [scene.position for scene in scenes] == [0, 1, 2]
    assert [scene.id for scene in scenes] == [third.id, first.id, second.id]
    assert all(scene.project_id == project.id for scene in scenes)
    assert scenes[0].project is project
    assert [scene.id for scene in project.scenes] == [
        third.id,
        first.id,
        second.id,
    ]


def test_scene_crud(session: Session) -> None:
    project = make_project(session)
    scene = create_scene(
        session,
        project_id=project.id,
        text="Narration",
        image_prompt="Image",
    )

    assert get_scene(session, scene.id) is scene
    updated = update_scene(
        session,
        scene.id,
        text="Updated narration",
        audio_path="audio.wav",
        duration=2.5,
    )

    assert updated is not None
    assert updated.text == "Updated narration"
    assert updated.audio_path == "audio.wav"
    assert updated.duration == 2.5
    assert delete_scene(session, scene.id) is True
    assert get_scene(session, scene.id) is None
    assert delete_scene(session, scene.id) is False


def test_move_and_delete_scene_keep_gap_free_order(session: Session) -> None:
    project = make_project(session)
    scenes = [
        create_scene(
            session,
            project_id=project.id,
            text=f"Narration {index}",
            image_prompt=f"Image {index}",
        )
        for index in range(3)
    ]

    move_scene(session, scenes[2].id, "up")
    assert [scene.id for scene in list_scenes(session, project.id)] == [
        scenes[0].id,
        scenes[2].id,
        scenes[1].id,
    ]

    delete_scene(session, scenes[2].id)
    remaining = list_scenes(session, project.id)
    assert [scene.position for scene in remaining] == [0, 1]


def test_delete_project_cascades_to_scenes(session: Session) -> None:
    project = make_project(session)
    scenes = [
        create_scene(
            session,
            project_id=project.id,
            text=f"Narration {index}",
            image_prompt=f"Image {index}",
        )
        for index in range(2)
    ]
    scene_ids = [scene.id for scene in scenes]

    assert delete_project(session, project.id) is True

    assert all(get_scene(session, scene_id) is None for scene_id in scene_ids)
