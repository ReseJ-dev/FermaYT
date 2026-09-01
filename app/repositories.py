"""Minimal repositories for persisted projects and scenes."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.persistence import ApplicationSettings, Project, Scene

PROJECT_UPDATE_FIELDS = frozenset(
    {
        "name",
        "story_text",
        "global_image_style_prompt",
        "scene_count",
        "image_provider",
        "image_model",
        "tts_provider",
        "tts_model",
        "tts_voice",
        "tts_language",
        "width",
        "height",
        "fps",
        "image_fit",
        "final_video_path",
        "rendered_at",
    }
)
SCENE_UPDATE_FIELDS = frozenset(
    {
        "text",
        "image_prompt",
        "image_path",
        "audio_path",
        "video_path",
        "duration",
    }
)


def get_application_settings(session: Session) -> ApplicationSettings:
    settings = session.get(ApplicationSettings, 1)
    if settings is None:
        settings = ApplicationSettings(id=1)
        session.add(settings)
        session.commit()
        session.refresh(settings)
    return settings


def update_application_settings(
    session: Session,
    *,
    image_provider: str,
    tts_provider: str,
    qwen_image_endpoint: str | None,
) -> ApplicationSettings:
    settings = get_application_settings(session)
    settings.image_provider = image_provider
    settings.tts_provider = tts_provider
    settings.qwen_image_endpoint = qwen_image_endpoint
    settings.updated_at = datetime.now(UTC)
    session.commit()
    session.refresh(settings)
    return settings


def create_project(
    session: Session,
    *,
    name: str,
    story_text: str,
    global_image_style_prompt: str | None = None,
    scene_count: int | None = None,
    image_provider: str = "seedream",
    image_model: str | None = None,
    tts_provider: str = "qwen",
    tts_model: str | None = None,
    tts_voice: str = "Cherry",
    tts_language: str = "Russian",
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    image_fit: str = "cover",
    final_video_path: str | None = None,
    rendered_at: datetime | None = None,
) -> Project:
    project = Project(
        name=name,
        story_text=story_text,
        global_image_style_prompt=global_image_style_prompt,
        scene_count=scene_count,
        image_provider=image_provider,
        image_model=image_model,
        tts_provider=tts_provider,
        tts_model=tts_model,
        tts_voice=tts_voice,
        tts_language=tts_language,
        width=width,
        height=height,
        fps=fps,
        image_fit=image_fit,
        final_video_path=final_video_path,
        rendered_at=rendered_at,
    )
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


def get_project(session: Session, project_id: str) -> Project | None:
    return session.get(Project, project_id)


def list_projects(session: Session) -> list[Project]:
    statement = select(Project).order_by(Project.created_at, Project.id)
    return list(session.scalars(statement))


def update_project(
    session: Session,
    project_id: str,
    **changes: Any,
) -> Project | None:
    project = get_project(session, project_id)
    if project is None:
        return None
    _validate_update_fields(changes, PROJECT_UPDATE_FIELDS)
    for field, value in changes.items():
        setattr(project, field, value)
    project.updated_at = datetime.now(UTC)
    session.commit()
    session.refresh(project)
    return project


def delete_project(session: Session, project_id: str) -> bool:
    project = get_project(session, project_id)
    if project is None:
        return False
    session.delete(project)
    session.commit()
    return True


def create_scene(
    session: Session,
    *,
    project_id: str,
    text: str,
    image_prompt: str,
    image_path: str | None = None,
    audio_path: str | None = None,
    video_path: str | None = None,
    duration: float | None = None,
) -> Scene:
    project = get_project(session, project_id)
    if project is None:
        raise ValueError(f"Project not found: {project_id}")
    last_position = session.scalar(
        select(func.max(Scene.position)).where(Scene.project_id == project_id)
    )
    position = 0 if last_position is None else last_position + 1
    scene = Scene(
        project=project,
        position=position,
        text=text,
        image_prompt=image_prompt,
        image_path=image_path,
        audio_path=audio_path,
        video_path=video_path,
        duration=duration,
    )
    session.add(scene)
    session.commit()
    session.refresh(scene)
    return scene


def get_scene(session: Session, scene_id: str) -> Scene | None:
    return session.get(Scene, scene_id)


def list_scenes(session: Session, project_id: str) -> list[Scene]:
    statement = (
        select(Scene)
        .where(Scene.project_id == project_id)
        .order_by(Scene.position, Scene.id)
    )
    return list(session.scalars(statement))


def update_scene(
    session: Session,
    scene_id: str,
    **changes: Any,
) -> Scene | None:
    scene = get_scene(session, scene_id)
    if scene is None:
        return None
    _validate_update_fields(changes, SCENE_UPDATE_FIELDS)
    for field, value in changes.items():
        setattr(scene, field, value)
    scene.updated_at = datetime.now(UTC)
    session.commit()
    session.refresh(scene)
    return scene


def delete_scene(session: Session, scene_id: str) -> bool:
    scene = get_scene(session, scene_id)
    if scene is None:
        return False
    project_id = scene.project_id
    session.delete(scene)
    session.flush()
    _normalize_scene_positions(session, project_id)
    session.commit()
    return True


def move_scene(
    session: Session,
    scene_id: str,
    direction: str,
) -> Scene | None:
    """Move a scene one position while preserving a gap-free order."""
    scene = get_scene(session, scene_id)
    if scene is None:
        return None
    offset = {"up": -1, "down": 1}.get(direction)
    if offset is None:
        raise ValueError("direction must be 'up' or 'down'")

    scenes = list_scenes(session, scene.project_id)
    current_index = next(
        index for index, item in enumerate(scenes) if item.id == scene.id
    )
    target_index = current_index + offset
    if not 0 <= target_index < len(scenes):
        return scene

    other = scenes[target_index]
    temporary_position = len(scenes)
    scene.position = temporary_position
    session.flush()
    other.position = current_index
    session.flush()
    scene.position = target_index
    session.commit()
    session.refresh(scene)
    return scene


def _normalize_scene_positions(session: Session, project_id: str) -> None:
    scenes = list_scenes(session, project_id)
    for position, scene in enumerate(scenes):
        scene.position = position
    session.flush()


def _validate_update_fields(
    changes: dict[str, Any],
    allowed_fields: frozenset[str],
) -> None:
    unknown_fields = changes.keys() - allowed_fields
    if unknown_fields:
        fields = ", ".join(sorted(unknown_fields))
        raise ValueError(f"Unsupported update fields: {fields}")
