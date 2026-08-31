"""Local web application entry point."""

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import parse_qs, quote, urlencode

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import SessionLocal, init_database
from app.database import engine as database_engine
from app.jobs import GenerationJobManager
from app.persistence import Scene
from app.repositories import (
    create_project,
    create_scene,
    delete_project,
    delete_scene,
    get_project,
    get_scene,
    list_projects,
    list_scenes,
    move_scene,
    update_project,
    update_scene,
)

APP_DIR = Path(__file__).resolve().parent
PROJECTS_ROOT = Path("data/projects").resolve()
job_manager = GenerationJobManager()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await asyncio.to_thread(init_database, database_engine)
    await job_manager.startup()
    try:
        yield
    finally:
        await job_manager.shutdown()


app = FastAPI(title="FermaYT", lifespan=lifespan)
app.mount(
    "/static",
    StaticFiles(directory=APP_DIR / "static"),
    name="static",
)
templates = Jinja2Templates(directory=APP_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render the project dashboard."""
    with SessionLocal() as session:
        projects = list_projects(session)
        project_cards = [
            {
                "project": project,
                "scene_count": len(list_scenes(session, project.id)),
            }
            for project in projects
        ]
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": "Проекты — FermaYT",
            "project_cards": project_cards,
            "notice": request.query_params.get("notice"),
            "error": request.query_params.get("error"),
        },
    )


@app.post("/projects")
async def create_project_route(request: Request) -> RedirectResponse:
    """Create a project from the dashboard form."""
    form = await _read_form(request)
    name = form.get("name", "").strip()
    story_text = form.get("story_text", "").strip()
    if not name or not story_text:
        return _redirect("/", error="Укажите название и текст истории.")

    try:
        with SessionLocal() as session:
            project = create_project(
                session,
                name=name,
                story_text=story_text,
                scene_count=_optional_int(form.get("scene_count")),
            )
    except ValueError as exc:
        return _redirect("/", error=_safe_validation_message(exc))
    return RedirectResponse(f"/projects/{project.id}", status_code=303)


@app.get("/projects/{project_id}", response_class=HTMLResponse)
async def project_editor(request: Request, project_id: str) -> HTMLResponse:
    """Render project settings and ordered scene cards."""
    with SessionLocal() as session:
        project = get_project(session, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        scenes = list_scenes(session, project_id)
        scene_cards = [
            {
                "scene": scene,
                "image_url": _stored_media_url(project_id, scene.image_path),
                "audio_url": _stored_media_url(project_id, scene.audio_path),
            }
            for scene in scenes
        ]
        final_video_url = _stored_media_url(
            project_id,
            project.final_video_path,
        )
        return templates.TemplateResponse(
            request=request,
            name="project.html",
            context={
                "title": f"{project.name} — FermaYT",
                "project": project,
                "scene_cards": scene_cards,
                "final_video_url": final_video_url,
                "notice": request.query_params.get("notice"),
                "error": request.query_params.get("error"),
            },
        )


@app.post("/projects/{project_id}")
async def update_project_route(
    request: Request,
    project_id: str,
) -> RedirectResponse:
    """Update editable project settings."""
    form = await _read_form(request)
    preset = form.get("output_preset", "vertical")
    dimensions = {
        "vertical": (1080, 1920),
        "horizontal": (1920, 1080),
        "square": (1080, 1080),
    }.get(preset)
    if dimensions is None:
        return _project_redirect(project_id, error="Неизвестный формат видео.")

    try:
        with SessionLocal() as session:
            project = update_project(
                session,
                project_id,
                name=_required(form, "name", "Название проекта"),
                story_text=_required(form, "story_text", "История"),
                global_image_style_prompt=form.get(
                    "global_image_style_prompt"
                ),
                scene_count=_optional_int(form.get("scene_count")),
                image_provider=_choice(
                    form,
                    "image_provider",
                    {"seedream", "qwen"},
                    "Провайдер изображений",
                ),
                image_model=form.get("image_model", "").strip() or None,
                tts_provider=_choice(
                    form,
                    "tts_provider",
                    {"qwen"},
                    "Провайдер озвучки",
                ),
                tts_model=form.get("tts_model", "").strip() or None,
                tts_voice=_required(form, "tts_voice", "Голос"),
                tts_language=_required(form, "tts_language", "Язык"),
                width=dimensions[0],
                height=dimensions[1],
                fps=int(_choice(form, "fps", {"24", "30", "60"}, "FPS")),
                image_fit=_choice(
                    form,
                    "image_fit",
                    {"cover", "contain"},
                    "Масштаб изображения",
                ),
            )
            if project is None:
                raise HTTPException(status_code=404, detail="Project not found")
    except ValueError as exc:
        return _project_redirect(
            project_id,
            error=_safe_validation_message(exc),
        )
    return _project_redirect(project_id, notice="Настройки проекта сохранены.")


@app.post("/projects/{project_id}/delete")
async def delete_project_route(project_id: str) -> RedirectResponse:
    """Delete a project and its database scenes."""
    with SessionLocal() as session:
        if not delete_project(session, project_id):
            raise HTTPException(status_code=404, detail="Project not found")
    return _redirect("/", notice="Проект удалён.")


@app.post("/projects/{project_id}/scenes")
async def create_scene_route(
    request: Request,
    project_id: str,
) -> RedirectResponse:
    """Append a scene to a project."""
    form = await _read_form(request)
    try:
        with SessionLocal() as session:
            create_scene(
                session,
                project_id=project_id,
                text=_required(form, "text", "Текст сцены"),
                image_prompt=_required(
                    form,
                    "image_prompt",
                    "Промпт изображения",
                ),
            )
    except ValueError as exc:
        return _project_redirect(
            project_id,
            error=_safe_validation_message(exc),
        )
    return _project_redirect(project_id, notice="Сцена добавлена.")


@app.post("/projects/{project_id}/scenes/{scene_id}")
async def update_scene_route(
    request: Request,
    project_id: str,
    scene_id: str,
) -> RedirectResponse:
    """Save scene narration and image prompt."""
    form = await _read_form(request)
    try:
        with SessionLocal() as session:
            scene = _project_scene(session, project_id, scene_id)
            update_scene(
                session,
                scene.id,
                text=_required(form, "text", "Текст сцены"),
                image_prompt=_required(
                    form,
                    "image_prompt",
                    "Промпт изображения",
                ),
            )
    except ValueError as exc:
        return _project_redirect(
            project_id,
            error=_safe_validation_message(exc),
        )
    return _project_redirect(project_id, notice="Сцена сохранена.")


@app.post("/projects/{project_id}/scenes/{scene_id}/delete")
async def delete_scene_route(
    project_id: str,
    scene_id: str,
) -> RedirectResponse:
    """Delete a scene belonging to this project."""
    with SessionLocal() as session:
        scene = _project_scene(session, project_id, scene_id)
        delete_scene(session, scene.id)
    return _project_redirect(project_id, notice="Сцена удалена.")


@app.post("/projects/{project_id}/scenes/{scene_id}/move/{direction}")
async def move_scene_route(
    project_id: str,
    scene_id: str,
    direction: str,
) -> RedirectResponse:
    """Move a scene up or down in its project."""
    with SessionLocal() as session:
        scene = _project_scene(session, project_id, scene_id)
        try:
            move_scene(session, scene.id, direction)
        except ValueError as exc:
            return _project_redirect(
                project_id,
                error=_safe_validation_message(exc),
            )
    return _project_redirect(project_id)


@app.get("/media/{project_id}/{media_path:path}")
async def project_media(project_id: str, media_path: str) -> FileResponse:
    """Serve only files located inside the requested project media root."""
    project_root = (PROJECTS_ROOT / project_id).resolve()
    candidate = (project_root / media_path).resolve()
    if (
        project_root.parent != PROJECTS_ROOT
        or not candidate.is_relative_to(project_root)
        or not candidate.is_file()
    ):
        raise HTTPException(status_code=404, detail="Media not found")
    return FileResponse(candidate)


@app.get("/settings", response_class=HTMLResponse)
async def settings(request: Request) -> HTMLResponse:
    """Show provider configuration status without exposing secrets."""
    return templates.TemplateResponse(
        request=request,
        name="settings.html",
        context={
            "title": "Настройки — FermaYT",
            "byteplus_configured": bool(
                os.getenv("BYTEPLUS_ARK_API_KEY", "").strip()
            ),
            "dashscope_configured": bool(
                os.getenv("DASHSCOPE_API_KEY", "").strip()
            ),
            "qwen_image_endpoint_configured": bool(
                os.getenv("QWEN_IMAGE_ENDPOINT", "").strip()
            ),
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Return application health status."""
    return {"status": "ok"}


async def _read_form(request: Request) -> dict[str, str]:
    content_type = request.headers.get("content-type", "")
    if "application/x-www-form-urlencoded" not in content_type:
        raise HTTPException(status_code=415, detail="Unsupported form type")
    body = (await request.body()).decode("utf-8")
    return {
        key: values[-1]
        for key, values in parse_qs(body, keep_blank_values=True).items()
    }


def _required(form: dict[str, str], field: str, label: str) -> str:
    value = form.get(field, "").strip()
    if not value:
        raise ValueError(f"{label}: поле не может быть пустым.")
    return value


def _optional_int(value: str | None) -> int | None:
    normalized = (value or "").strip()
    if not normalized:
        return None
    try:
        return int(normalized)
    except ValueError as exc:
        raise ValueError("Количество сцен должно быть целым числом.") from exc


def _choice(
    form: dict[str, str],
    field: str,
    allowed: set[str],
    label: str,
) -> str:
    value = form.get(field, "").strip()
    if value not in allowed:
        raise ValueError(f"{label}: выбрано недопустимое значение.")
    return value


def _project_scene(session: Session, project_id: str, scene_id: str) -> Scene:
    scene = get_scene(session, scene_id)
    if scene is None or scene.project_id != project_id:
        raise HTTPException(status_code=404, detail="Scene not found")
    return scene


def _redirect(path: str, **query: str) -> RedirectResponse:
    location = path
    if query:
        location = f"{path}?{urlencode(query)}"
    return RedirectResponse(location, status_code=303)


def _project_redirect(project_id: str, **query: str) -> RedirectResponse:
    return _redirect(f"/projects/{project_id}", **query)


def _safe_validation_message(error: ValueError) -> str:
    message = str(error).strip()
    return message[:300] if message else "Проверьте введённые данные."


def _stored_media_url(project_id: str, stored_path: str | None) -> str | None:
    if not stored_path:
        return None
    project_root = (PROJECTS_ROOT / project_id).resolve()
    candidate = Path(stored_path)
    if not candidate.is_absolute():
        candidate = candidate.resolve()
    try:
        relative_path = candidate.relative_to(project_root)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    encoded_path = quote(relative_path.as_posix(), safe="/")
    return f"/media/{quote(project_id)}/{encoded_path}"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
