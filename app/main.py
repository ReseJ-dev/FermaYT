"""Local web application entry point."""

import asyncio
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, quote, urlencode
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.budgets import ProjectBudgetGuard
from app.clients.image_api import ImageGenerationError
from app.costs import (
    estimate_project_generation_cost,
    load_pricing_config,
    summarize_project_cost,
)
from app.database import SessionLocal, init_database
from app.database import engine as database_engine
from app.errors import (
    MediaProbeError,
    StructuredAIProviderError,
    TTSGenerationError,
)
from app.generation_scope import GenerationScope, GenerationScopeType
from app.generators.image import (
    build_image_generation_prompt,
    generate_image,
)
from app.generators.master_scene import register_uploaded_master_scene
from app.generators.style_reference import register_approved_style_reference
from app.generators.voice import generate_voice
from app.jobs import (
    GenerationJob,
    GenerationJobManager,
    GenerationJobStatus,
    GenerationJobType,
)
from app.media.probe import get_media_duration
from app.persistence import PlanningProviderAttempt, Project, Scene
from app.production_profiles import ProductionProfile
from app.providers import get_image_provider, get_tts_provider
from app.repositories import (
    create_project,
    create_scene,
    delete_master_scene_asset,
    delete_project,
    delete_scene,
    get_application_settings,
    get_beat_visual_result,
    get_project,
    get_scene,
    get_style_reference_asset,
    list_master_scene_assets,
    list_project_video_renders,
    list_projects,
    list_scenes,
    move_scene,
    update_application_settings,
    update_project,
    update_scene,
)
from app.secret_store import (
    BYTEPLUS_API_KEY,
    DASHSCOPE_API_KEY,
    ELEVENLABS_API_KEY,
    KIE_API_KEY,
    KIMI_API_KEY,
    SecretStore,
    SecretStoreError,
)
from app.services.pipeline_production import build_production_pipeline_dependencies
from app.services.planning_attempts import (
    latest_uncertain_planning_attempt,
    planning_budget_snapshot,
)
from app.services.project_pipeline import run_project_video_pipeline
from app.services.prompt_assembly import (
    PromptTargetType,
    clear_prompt_override,
    get_prompt_detail,
    get_prompt_sheet,
    preview_generation_request,
    set_prompt_override,
)
from app.services.visual_planning import load_project_visual_plan_state
from app.storage import ProjectMediaPaths

APP_DIR = Path(__file__).resolve().parent
PROJECTS_ROOT = Path("data/projects").resolve()
PRICING_CONFIG_PATH = Path(
    os.environ.get("FERMAYT_PRICING_FILE", "config/provider_pricing.json")
)
job_manager = GenerationJobManager()
secret_store = SecretStore()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await asyncio.to_thread(init_database, database_engine)
    if PRICING_CONFIG_PATH.is_file():
        with SessionLocal() as session:
            load_pricing_config(session, PRICING_CONFIG_PATH)
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
            application_settings = get_application_settings(session)
            default_tts_provider = application_settings.tts_provider
            project = create_project(
                session,
                name=name,
                story_text=story_text,
                global_image_style_prompt=form.get("global_image_style_prompt"),
                scene_count=_optional_int(form.get("scene_count")),
                image_provider=application_settings.image_provider,
                image_model=(
                    "qwen-image-3.0"
                    if application_settings.image_provider == "qwen"
                    else (
                        "z-image"
                        if application_settings.image_provider == "zimage"
                        else "seedream-5-0-260128"
                    )
                ),
                tts_provider=default_tts_provider,
                tts_model=(
                    "eleven_multilingual_v2"
                    if default_tts_provider == "elevenlabs"
                    else "qwen3-tts-flash"
                ),
                tts_voice=(
                    "JBFqnCBsd6RMkjVDRZzb"
                    if default_tts_provider == "elevenlabs"
                    else "Cherry"
                ),
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
        draft_video_url = _stored_media_url(
            project_id,
            project.draft_video_path,
        )
        pilot_video_url = _stored_media_url(project_id, project.pilot_video_path)
        style_reference = get_style_reference_asset(
            session, project_id, project.style_id
        )
        renders = list_project_video_renders(session, project_id)
        video_render_cards: list[dict[str, object]] = []
        rendered_paths: set[str] = set()
        current_final_path = (
            str(Path(project.final_video_path).resolve())
            if project.final_video_path
            else None
        )
        for render in renders:
            if render.status != "SUCCEEDED" or not render.output_path:
                continue
            video_url = _stored_media_url(project_id, render.output_path)
            if video_url is None:
                continue
            rendered_paths.add(str(Path(render.output_path).resolve()))
            if render.generation_scope_type != GenerationScopeType.FULL.value:
                label = "PILOT / PARTIAL"
            elif render.production_profile == ProductionProfile.DRAFT.value:
                label = "DRAFT VIDEO"
            else:
                label = "FULL VIDEO"
            video_render_cards.append(
                {
                    "id": render.id,
                    "label": label,
                    "video_url": video_url,
                    "download_url": (
                        f"/api/projects/{quote(project_id)}/video/download"
                        if str(Path(render.output_path).resolve()) == current_final_path
                        else video_url
                    ),
                    "duration": render.duration,
                    "width": render.width,
                    "height": render.height,
                    "fps": render.fps,
                    "completed_at": render.completed_at or render.created_at,
                }
            )
        legacy_videos = (
            ("FULL VIDEO", project.final_video_path, project.rendered_at),
            ("DRAFT VIDEO", project.draft_video_path, project.draft_rendered_at),
            ("PILOT / PARTIAL", project.pilot_video_path, project.pilot_rendered_at),
        )
        for label, output_path, completed_at in legacy_videos:
            if not output_path or str(Path(output_path).resolve()) in rendered_paths:
                continue
            video_url = _stored_media_url(project_id, output_path)
            if video_url is not None:
                video_render_cards.append(
                    {
                        "id": None,
                        "label": label,
                        "video_url": video_url,
                        "download_url": (
                            f"/api/projects/{quote(project_id)}/video/download"
                            if str(Path(output_path).resolve()) == current_final_path
                            else video_url
                        ),
                        "duration": None,
                        "width": None,
                        "height": None,
                        "fps": None,
                        "completed_at": completed_at,
                    }
                )
        video_render_cards.sort(
            key=lambda card: (
                card["completed_at"].timestamp()
                if card["completed_at"] is not None
                else 0.0
            ),
            reverse=True,
        )
        latest_job = await job_manager.get_latest_project_job(project_id)
        style_preview_cards: list[dict[str, object]] = []
        if (
            latest_job is not None
            and latest_job.generation_scope_type == GenerationScopeType.STYLE_PREVIEW.value
            and isinstance(latest_job.report, dict)
        ):
            result_ids = latest_job.report.get("preview_result_ids", [])
            if isinstance(result_ids, list):
                for result_id in result_ids[:3]:
                    if not isinstance(result_id, str):
                        continue
                    result = get_beat_visual_result(session, result_id)
                    if result is None or not result.is_accepted:
                        continue
                    image_url = _stored_media_url(project_id, result.output_path)
                    if image_url is not None:
                        style_preview_cards.append(
                            {
                                "beat_id": result.beat_id,
                                "operation": result.resolved_operation,
                                "image_url": image_url,
                            }
                        )
        cost_summary = summarize_project_cost(
            session,
            project_id,
            job_id=latest_job.id if latest_job is not None else None,
        )
        cost_estimate = estimate_project_generation_cost(session, project_id)
        draft_cost_estimate = estimate_project_generation_cost(
            session, project_id, production_profile="DRAFT"
        )
        final_cost_estimate = estimate_project_generation_cost(
            session, project_id, production_profile="FINAL"
        )
        plan_state = load_project_visual_plan_state(session, project_id)
        master_definitions = (
            list(plan_state.plan.possible_master_scenes)
            if plan_state is not None and plan_state.is_current
            else []
        )
        master_assets_by_id = {
            asset.master_scene_id: asset
            for asset in list_master_scene_assets(session, project_id)
        }
        master_scene_cards = [
            {
                "master_scene_id": definition.id,
                "asset": master_assets_by_id.get(definition.id),
                "definition": definition,
                "image_url": _stored_media_url(
                    project_id,
                    master_assets_by_id[definition.id].file_path,
                ) if definition.id in master_assets_by_id else None,
            }
            for definition in master_definitions
        ]
        defined_master_ids = {definition.id for definition in master_definitions}
        master_scene_cards.extend(
            {
                "master_scene_id": asset.master_scene_id,
                "asset": asset,
                "definition": None,
                "image_url": _stored_media_url(project_id, asset.file_path),
            }
            for asset in master_assets_by_id.values()
            if asset.master_scene_id not in defined_master_ids
        )
        pilot_cost_estimates: dict[int, object] = {}
        style_preview_cost_estimate = None
        if plan_state is not None and plan_state.is_current:
            style_preview_scope = GenerationScope(
                GenerationScopeType.STYLE_PREVIEW
            )
            style_preview_cost_estimate = estimate_project_generation_cost(
                session,
                project_id,
                production_profile="FINAL",
                beat_ids=frozenset(
                    style_preview_scope.select_beat_ids(plan_state.plan)
                ),
            )
            for seconds in (30, 60):
                pilot_scope = GenerationScope(
                    GenerationScopeType.FIRST_SECONDS, seconds
                )
                pilot_cost_estimates[seconds] = estimate_project_generation_cost(
                    session,
                    project_id,
                    production_profile="FINAL",
                    beat_ids=frozenset(pilot_scope.select_beat_ids(plan_state.plan)),
                )
        budget_snapshot = ProjectBudgetGuard(session, project_id).snapshot()
        return templates.TemplateResponse(
            request=request,
            name="project.html",
            context={
                "title": f"{project.name} — FermaYT",
                "project": project,
                "scene_cards": scene_cards,
                "master_scene_cards": master_scene_cards,
                "style_preview_cards": style_preview_cards,
                "final_video_url": final_video_url,
                "draft_video_url": draft_video_url,
                "pilot_video_url": pilot_video_url,
                "video_render_cards": video_render_cards,
                "style_reference": style_reference,
                "latest_job": latest_job,
                "cost_summary": cost_summary,
                "cost_estimate": cost_estimate,
                "draft_cost_estimate": draft_cost_estimate,
                "final_cost_estimate": final_cost_estimate,
                "pilot_cost_estimates": pilot_cost_estimates,
                "style_preview_cost_estimate": style_preview_cost_estimate,
                "budget_snapshot": budget_snapshot,
                "notice": request.query_params.get("notice"),
                "error": request.query_params.get("error"),
            },
        )


@app.get("/projects/{project_id}/prompts", response_class=HTMLResponse)
async def project_prompt_sheet_page(
    request: Request,
    project_id: str,
) -> HTMLResponse:
    """Render the compact project-wide VisualPlan prompt inspector."""
    with SessionLocal() as session:
        project = get_project(session, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        try:
            rows = get_prompt_sheet(session, project_id)
        except ValueError:
            rows = []
        masters = [
            row
            for row in rows
            if row["target_type"] == "MASTER_SCENE"
            and row.get("override_state") != "ORPHANED"
        ]
        beats = [
            row
            for row in rows
            if row["target_type"] == "BEAT"
            and row.get("override_state") != "ORPHANED"
        ]
        orphaned = [
            row for row in rows if row.get("override_state") == "ORPHANED"
        ]
        return templates.TemplateResponse(
            request=request,
            name="prompt_sheet.html",
            context={
                "title": f"Prompts — {project.name} — FermaYT",
                "project": project,
                "master_rows": masters,
                "beat_rows": beats,
                "orphaned_rows": orphaned,
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
    try:
        with SessionLocal() as session:
            _update_project_from_form(session, project_id, form)
    except ValueError as exc:
        return _project_redirect(
            project_id,
            error=_safe_validation_message(exc),
        )
    return _project_redirect(project_id, notice="Настройки проекта сохранены.")


@app.get("/api/projects/{project_id}/prompt-sheet")
async def project_prompt_sheet_route(project_id: str) -> dict[str, object]:
    """Return stable beat/master prompt targets and their override state."""
    try:
        with SessionLocal() as session:
            return {"targets": get_prompt_sheet(session, project_id)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}/prompts/{target_type}/{target_id}")
async def project_prompt_detail_route(
    project_id: str,
    target_type: str,
    target_id: str,
) -> dict[str, object]:
    """Preview the structured request compiled by the production prompt path."""
    try:
        with SessionLocal() as session:
            return get_prompt_detail(
                session,
                project_id,
                target_type=PromptTargetType(target_type.upper()),
                target_id=target_id,
            )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/api/projects/{project_id}/prompts/{target_type}/{target_id}/override")
async def set_project_prompt_override_route(
    request: Request,
    project_id: str,
    target_type: str,
    target_id: str,
) -> dict[str, object]:
    """Set only the desired scene description for one current plan target."""
    try:
        payload = await request.json()
        scene_prompt = payload.get("scene_prompt_override")
        if not isinstance(scene_prompt, str):
            raise TypeError("scene_prompt_override must be a string")
        with SessionLocal() as session:
            override = set_prompt_override(
                session,
                project_id,
                target_type=PromptTargetType(target_type.upper()),
                target_id=target_id,
                scene_prompt_override=scene_prompt,
            )
            return {
                "override_id": override.id,
                "revision": override.revision,
                "enabled": override.enabled,
                "prompt": get_prompt_detail(
                    session,
                    project_id,
                    target_type=target_type.upper(),
                    target_id=target_id,
                ),
            }
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/projects/{project_id}/prompts/{target_type}/{target_id}/override")
async def clear_project_prompt_override_route(
    project_id: str,
    target_type: str,
    target_id: str,
) -> dict[str, object]:
    """Disable the current revision's override and return to AUTO mode."""
    try:
        with SessionLocal() as session:
            override = clear_prompt_override(
                session,
                project_id,
                target_type=PromptTargetType(target_type.upper()),
                target_id=target_id,
            )
            return {"cleared": override is not None}
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/prompts/{target_type}/{target_id}/preview")
async def preview_project_prompt_route(
    request: Request,
    project_id: str,
    target_type: str,
    target_id: str,
) -> dict[str, object]:
    """Compile against an optional provider/model without dispatching a request."""
    try:
        payload = await request.json()
        with SessionLocal() as session:
            assembly = preview_generation_request(
                session,
                project_id,
                target_type=PromptTargetType(target_type.upper()),
                target_id=target_id,
                provider=payload.get("provider"),
                model=payload.get("model"),
                qa_correction=payload.get("qa_correction"),
                operation=payload.get("operation"),
            )
            return assembly.as_dict()
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/generate-video")
async def generate_project_video_route(
    request: Request,
    project_id: str,
) -> dict[str, object]:
    """Save settings and immediately enqueue the resumable full pipeline."""
    active = await job_manager.get_active_project_job(project_id)
    if active is not None:
        return _job_payload(active)
    form = await _read_optional_form(request)
    budget_override = form.get("budget_override") == "1"
    planning_retry_anyway = form.get("planning_retry_anyway") == "1"
    latest_job = await job_manager.get_latest_project_job(project_id)
    planning_run_id: str | None = None
    if latest_job is not None and latest_job.status is GenerationJobStatus.PAUSED_PLANNING:
        with SessionLocal() as session:
            uncertain = latest_uncertain_planning_attempt(session, project_id)
        if uncertain is not None:
            if not planning_retry_anyway:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Planning request timed out locally. Provider billing status "
                        "is unknown. Retrying may create a second charge."
                    ),
                )
            planning_run_id = uncertain.planning_run_id
    elif (
        latest_job is not None
        and latest_job.status is GenerationJobStatus.PAUSED_BUDGET
        and latest_job.failed_stage == "PLANNING"
    ):
        # A budget increase resumes the same planning ledger; unknown reservations
        # must never disappear merely because the worker gets a new job id.
        with SessionLocal() as session:
            prior = session.scalar(
                select(PlanningProviderAttempt)
                .where(PlanningProviderAttempt.project_id == project_id)
                .order_by(PlanningProviderAttempt.started_at.desc())
                .limit(1)
            )
        if prior is not None:
            planning_run_id = prior.planning_run_id
    try:
        production_profile = ProductionProfile(
            form.get("production_profile", ProductionProfile.FINAL.value).upper()
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail="Unknown production profile"
        ) from exc
    try:
        scope_type = GenerationScopeType(
            form.get("generation_scope_type", GenerationScopeType.FULL.value).upper()
        )
        raw_scope_value = form.get("generation_scope_value", "").strip()
        scope_value = float(raw_scope_value) if raw_scope_value else None
        generation_scope = GenerationScope(scope_type, scope_value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid generation scope") from exc
    with SessionLocal() as session:
        project = get_project(session, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        if "name" in form:
            try:
                project = _update_project_from_form(session, project_id, form)
            except ValueError as exc:
                raise HTTPException(
                    status_code=422, detail=_safe_validation_message(exc)
                ) from exc
        try:
            dependencies = build_production_pipeline_dependencies(
                session,
                project_id,
                secret_store,
                projects_root=PROJECTS_ROOT,
                require_tts=not generation_scope.is_image_only,
            )
        except (SecretStoreError, StructuredAIProviderError, ValueError) as exc:
            raise HTTPException(
                status_code=422, detail=_safe_validation_message(exc)
            ) from exc

    async def operation(job_id: str) -> None:
        await asyncio.to_thread(
            _run_pipeline_worker,
            job_id,
            project_id,
            dependencies,
            budget_override,
            production_profile,
            generation_scope,
            planning_run_id,
            planning_retry_anyway,
        )

    job = await job_manager.enqueue(
        project_id,
        GenerationJobType.GENERATE_VIDEO,
        operation,
        production_profile=production_profile.value,
        generation_scope_type=generation_scope.type.value,
        generation_scope_value=(
            float(generation_scope.value)
            if generation_scope.value is not None
            else None
        ),
    )
    return _job_payload(job)


@app.get("/api/jobs/{job_id}")
async def generation_job(job_id: str) -> dict[str, object]:
    job = await job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_payload(job)


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_generation_job(job_id: str) -> dict[str, object]:
    job = await job_manager.cancel_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_payload(job)


@app.post("/api/projects/{project_id}/style-reference")
async def upload_style_reference(request: Request, project_id: str) -> dict[str, str]:
    """Register one approved PNG without trusting a browser filename."""
    if request.headers.get("content-type", "").split(";", 1)[0] != "image/png":
        raise HTTPException(status_code=415, detail="Style reference must be PNG")
    body = await request.body()
    if not body or len(body) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Style reference must be 1–10 MB")
    with SessionLocal() as session:
        project = get_project(session, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        staging = (
            ProjectMediaPaths(project_id, PROJECTS_ROOT).uploads_dir / f"{uuid4()}.png"
        )
        try:
            await asyncio.to_thread(staging.write_bytes, body)
            asset = register_approved_style_reference(
                session,
                project_id,
                staging,
                style_id=project.style_id,
                projects_root=PROJECTS_ROOT,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=422, detail=_safe_validation_message(exc)
            ) from exc
        finally:
            staging.unlink(missing_ok=True)
    return {"status": "registered", "style_id": asset.style_id}


@app.post("/api/projects/{project_id}/master-scenes")
async def upload_master_scene(
    request: Request,
    project_id: str,
    master_scene_id: str,
) -> dict[str, str]:
    """Register a user PNG in one master slot from the current visual plan."""
    if await job_manager.get_active_project_job(project_id) is not None:
        raise HTTPException(
            status_code=409,
            detail="Нельзя менять мастер-картинки во время генерации.",
        )
    if request.headers.get("content-type", "").split(";", 1)[0] != "image/png":
        raise HTTPException(status_code=415, detail="Мастер-картинка должна быть PNG")
    body = await request.body()
    if not body or len(body) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Размер мастер-картинки должен быть от 1 байта до 20 МБ",
        )
    with SessionLocal() as session:
        project = get_project(session, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        plan_state = load_project_visual_plan_state(session, project_id)
        if plan_state is None or not plan_state.is_current:
            raise HTTPException(
                status_code=409,
                detail="Сначала создайте актуальный VisualPlan проекта.",
            )
        staging = ProjectMediaPaths(
            project_id,
            PROJECTS_ROOT,
        ).uploads_dir / f"{uuid4()}.png"
        try:
            await asyncio.to_thread(staging.write_bytes, body)
            asset = register_uploaded_master_scene(
                session,
                project,
                plan_state.plan,
                master_scene_id,
                staging,
                projects_root=PROJECTS_ROOT,
                style_id=project.style_id,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=_safe_validation_message(exc),
            ) from exc
        finally:
            staging.unlink(missing_ok=True)
    return {
        "status": "registered",
        "master_scene_id": asset.master_scene_id,
    }


@app.post("/api/projects/{project_id}/master-scenes/{asset_id}/delete")
async def delete_master_scene(project_id: str, asset_id: str) -> dict[str, str]:
    """Delete one user-selected master record and its project-local image."""
    if await job_manager.get_active_project_job(project_id) is not None:
        raise HTTPException(
            status_code=409,
            detail="Нельзя менять мастер-картинки во время генерации.",
        )
    with SessionLocal() as session:
        asset = delete_master_scene_asset(session, project_id, asset_id)
        if asset is None:
            raise HTTPException(status_code=404, detail="Master scene not found")
        stored_path = asset.file_path
        master_scene_id = asset.master_scene_id
    project_root = (PROJECTS_ROOT / project_id).resolve()
    candidate = Path(stored_path).resolve()
    if candidate.is_relative_to(project_root):
        candidate.unlink(missing_ok=True)
    return {"status": "deleted", "master_scene_id": master_scene_id}


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


@app.post("/api/projects/{project_id}/scenes/{scene_id}/generate-audio")
async def generate_scene_audio_route(
    request: Request,
    project_id: str,
    scene_id: str,
) -> RedirectResponse:
    """Generate and persist audio for one scene using its project provider."""
    form = await _read_optional_form(request)
    with SessionLocal() as session:
        project = get_project(session, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        scene = _project_scene(session, project_id, scene_id)
        if form:
            try:
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
            scene = _project_scene(session, project_id, scene_id)
        provider_name = project.tts_provider
        scene_text = scene.text

    extension = ".mp3" if provider_name == "elevenlabs" else ".wav"
    output_path = str(
        ProjectMediaPaths(project_id, PROJECTS_ROOT).audio_path(
            scene_id,
            extension,
        )
    )
    try:
        provider_config = _tts_provider_config(project)
        provider = get_tts_provider(provider_name, provider_config)
        audio_path = await generate_voice(scene_text, output_path, provider)
        duration = await asyncio.to_thread(get_media_duration, audio_path)
    except (
        TTSGenerationError,
        MediaProbeError,
        SecretStoreError,
        httpx.HTTPError,
        OSError,
        ValueError,
    ) as exc:
        return _project_redirect(
            project_id,
            error=f"Не удалось создать озвучку: {_safe_validation_message(exc)}",
        )

    with SessionLocal() as session:
        updated_scene = _project_scene(session, project_id, scene_id)
        update_scene(
            session,
            updated_scene.id,
            audio_path=audio_path,
            duration=duration,
            video_path=None,
        )
    return _project_redirect(project_id, notice="Озвучка сцены готова.")


@app.post("/api/projects/{project_id}/scenes/{scene_id}/generate-image")
async def generate_scene_image_route(
    request: Request,
    project_id: str,
    scene_id: str,
) -> RedirectResponse:
    """Generate an image with the provider selected for this project."""
    form = await _read_optional_form(request)
    with SessionLocal() as session:
        project = get_project(session, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        scene = _project_scene(session, project_id, scene_id)
        if form:
            try:
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
            scene = _project_scene(session, project_id, scene_id)
        provider_name = project.image_provider
        scene_prompt = scene.image_prompt
        style_prompt = project.global_image_style_prompt

    output_path = str(ProjectMediaPaths(project_id, PROJECTS_ROOT).image_path(scene_id))
    try:
        final_prompt = build_image_generation_prompt(
            scene_prompt,
            style_prompt,
        )
        provider = get_image_provider(
            provider_name,
            _image_provider_config(project),
        )
        image_path = await generate_image(final_prompt, output_path, provider)
    except (
        ImageGenerationError,
        SecretStoreError,
        httpx.HTTPError,
        OSError,
        ValueError,
    ) as exc:
        return _project_redirect(
            project_id,
            error=f"Не удалось создать изображение: {_safe_validation_message(exc)}",
        )

    with SessionLocal() as session:
        updated_scene = _project_scene(session, project_id, scene_id)
        update_scene(
            session,
            updated_scene.id,
            image_path=image_path,
            video_path=None,
        )
    return _project_redirect(project_id, notice="Изображение сцены готово.")


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


@app.get("/api/projects/{project_id}/video/download")
async def download_project_video(project_id: str) -> FileResponse:
    """Download only the final video persisted for the requested project."""
    with SessionLocal() as session:
        project = get_project(session, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        stored_path = project.final_video_path
        project_name = project.name
    if not stored_path:
        raise HTTPException(status_code=404, detail="Final video not found")
    project_root = (PROJECTS_ROOT / project_id).resolve()
    candidate = Path(stored_path)
    if not candidate.is_absolute():
        candidate = candidate.resolve()
    if (
        project_root.parent != PROJECTS_ROOT
        or not candidate.is_relative_to(project_root)
        or not candidate.is_file()
    ):
        raise HTTPException(status_code=404, detail="Final video not found")
    safe_name = re.sub(r"[^\w.-]+", "_", project_name).strip("._") or "video"
    return FileResponse(
        candidate,
        media_type="video/mp4",
        filename=f"{safe_name}.mp4",
    )


@app.get("/settings", response_class=HTMLResponse)
async def settings(request: Request) -> HTMLResponse:
    """Show provider configuration status without exposing secrets."""
    byteplus_configured, byteplus_store_error = _secret_status(
        BYTEPLUS_API_KEY,
        "BYTEPLUS_ARK_API_KEY",
    )
    dashscope_configured, dashscope_store_error = _secret_status(
        DASHSCOPE_API_KEY,
        "DASHSCOPE_API_KEY",
    )
    elevenlabs_configured, elevenlabs_store_error = _secret_status(
        ELEVENLABS_API_KEY,
        "ELEVENLABS_API_KEY",
    )
    kimi_configured, kimi_store_error = _secret_status(
        KIMI_API_KEY,
        "MOONSHOT_API_KEY",
    )
    kie_configured, kie_store_error = _secret_status(KIE_API_KEY, "KIE_API_KEY")
    with SessionLocal() as session:
        application_settings = get_application_settings(session)
    return templates.TemplateResponse(
        request=request,
        name="settings.html",
        context={
            "title": "Настройки — FermaYT",
            "byteplus_configured": byteplus_configured,
            "dashscope_configured": dashscope_configured,
            "elevenlabs_configured": elevenlabs_configured,
            "kimi_configured": kimi_configured,
            "kie_configured": kie_configured,
            "qwen_image_endpoint_configured": bool(
                application_settings.qwen_image_endpoint
                or os.getenv("QWEN_IMAGE_ENDPOINT", "").strip()
            ),
            "application_settings": application_settings,
            "keyring_error": (
                byteplus_store_error
                or dashscope_store_error
                or elevenlabs_store_error
                or kimi_store_error
                or kie_store_error
            ),
            "notice": request.query_params.get("notice"),
            "error": request.query_params.get("error"),
        },
    )


@app.post("/settings")
async def update_settings(request: Request) -> RedirectResponse:
    """Save or explicitly remove API keys in the operating-system keyring."""
    form = await _read_form(request)
    with SessionLocal() as session:
        current_settings = get_application_settings(session)
        form.setdefault(
            "default_image_provider",
            current_settings.image_provider,
        )
        form.setdefault("default_tts_provider", current_settings.tts_provider)
        form.setdefault(
            "qwen_image_endpoint",
            current_settings.qwen_image_endpoint or "",
        )
    try:
        _update_secret_from_form(
            form,
            field="byteplus_api_key",
            delete_field="delete_byteplus_api_key",
            secret_name=BYTEPLUS_API_KEY,
        )
        _update_secret_from_form(
            form,
            field="dashscope_api_key",
            delete_field="delete_dashscope_api_key",
            secret_name=DASHSCOPE_API_KEY,
        )
        _update_secret_from_form(
            form,
            field="elevenlabs_api_key",
            delete_field="delete_elevenlabs_api_key",
            secret_name=ELEVENLABS_API_KEY,
        )
        _update_secret_from_form(
            form,
            field="kimi_api_key",
            delete_field="delete_kimi_api_key",
            secret_name=KIMI_API_KEY,
        )
        _update_secret_from_form(
            form,
            field="kie_api_key",
            delete_field="delete_kie_api_key",
            secret_name=KIE_API_KEY,
        )
        with SessionLocal() as session:
            update_application_settings(
                session,
                image_provider=_choice(
                    form,
                    "default_image_provider",
                    {"seedream", "qwen", "zimage"},
                    "Нейросеть изображений",
                ),
                tts_provider=_choice(
                    form,
                    "default_tts_provider",
                    {"qwen", "elevenlabs"},
                    "Нейросеть озвучки",
                ),
                qwen_image_endpoint=form.get("qwen_image_endpoint"),
            )
    except (SecretStoreError, ValueError) as exc:
        return _redirect("/settings", error=_safe_validation_message(exc))
    return _redirect("/settings", notice="API-ключи сохранены безопасно.")


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


async def _read_optional_form(request: Request) -> dict[str, str]:
    if not await request.body():
        return {}
    return await _read_form(request)


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


def _optional_float(value: str | None, label: str) -> float | None:
    normalized = (value or "").strip()
    if not normalized:
        return None
    try:
        return float(normalized)
    except ValueError as exc:
        raise ValueError(f"{label} должно быть числом.") from exc


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


def _safe_validation_message(error: Exception) -> str:
    message = str(error).strip()
    return message[:300] if message else "Проверьте введённые данные."


def _tts_provider_config(project: Project) -> dict[str, str]:
    provider_name = project.tts_provider
    if provider_name == "qwen":
        api_key = _configured_secret(DASHSCOPE_API_KEY, "DASHSCOPE_API_KEY")
        return {
            "api_key": api_key,
            "model": project.tts_model or "qwen3-tts-flash",
            "voice": project.tts_voice,
            "language": project.tts_language,
        }
    if provider_name == "elevenlabs":
        api_key = _configured_secret(
            ELEVENLABS_API_KEY,
            "ELEVENLABS_API_KEY",
        )
        return {
            "api_key": api_key,
            "model": project.tts_model or "eleven_multilingual_v2",
            "voice": project.tts_voice,
        }
    raise ValueError("Неизвестный провайдер озвучки.")


def _image_provider_config(project: Project) -> dict[str, str]:
    if project.image_provider == "seedream":
        return {
            "api_key": _configured_secret(
                BYTEPLUS_API_KEY,
                "BYTEPLUS_ARK_API_KEY",
            ),
            "model": project.image_model or "seedream-5-0-260128",
        }
    if project.image_provider == "qwen":
        with SessionLocal() as session:
            application_settings = get_application_settings(session)
            endpoint = (
                application_settings.qwen_image_endpoint
                or os.getenv("QWEN_IMAGE_ENDPOINT", "").strip()
            )
        if not endpoint:
            raise ValueError("Для Qwen Image настройте QWEN_IMAGE_ENDPOINT.")
        return {
            "api_key": _configured_secret(
                DASHSCOPE_API_KEY,
                "DASHSCOPE_API_KEY",
            ),
            "endpoint": endpoint,
            "model": project.image_model or "qwen-image-3.0",
        }
    if project.image_provider == "zimage":
        return {
            "api_key": _configured_secret(KIE_API_KEY, "KIE_API_KEY"),
            "model": project.image_model or "z-image",
            "aspect_ratio": _kie_aspect_ratio(project.width, project.height),
        }
    raise ValueError("Неизвестный провайдер изображений.")


def _kie_aspect_ratio(width: int, height: int) -> str:
    if width == height:
        return "1:1"
    return "9:16" if height > width else "16:9"


def _configured_secret(secret_name: str, environment_name: str) -> str:
    value = secret_store.get_secret(secret_name)
    if value:
        return value
    environment_value = os.getenv(environment_name, "").strip()
    if environment_value:
        return environment_value
    raise ValueError("Добавьте API-ключ провайдера в Настройках.")


def _secret_status(secret_name: str, environment_name: str) -> tuple[bool, str | None]:
    environment_configured = bool(os.getenv(environment_name, "").strip())
    try:
        return secret_store.has_secret(secret_name) or environment_configured, None
    except SecretStoreError as exc:
        return environment_configured, str(exc)


def _update_secret_from_form(
    form: dict[str, str],
    *,
    field: str,
    delete_field: str,
    secret_name: str,
) -> None:
    if form.get(delete_field) == "on":
        secret_store.delete_secret(secret_name)
        return
    value = form.get(field, "").strip()
    if value:
        secret_store.set_secret(secret_name, value)


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


def _update_project_from_form(
    session: Session,
    project_id: str,
    form: dict[str, str],
) -> Project:
    dimensions = {
        "vertical": (1080, 1920),
        "horizontal": (1920, 1080),
        "square": (1080, 1080),
    }.get(form.get("output_preset", "vertical"))
    if dimensions is None:
        raise ValueError("Неизвестный формат видео.")
    current = get_project(session, project_id)
    if current is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if "planning_provider" not in form:
        form.setdefault("planning_provider", current.planning_provider)
        form.setdefault("planning_model", current.planning_model)
        form.setdefault("visual_qa_enabled", "1" if current.visual_qa_enabled else "0")
        form.setdefault("visual_qa_provider", current.visual_qa_provider)
        form.setdefault("visual_qa_model", current.visual_qa_model)
        form.setdefault("style_id", current.style_id)
    form.setdefault(
        "generation_budget_enabled",
        "1" if current.generation_budget_enabled else "0",
    )
    form.setdefault(
        "generation_budget_amount",
        str(current.generation_budget_amount or ""),
    )
    form.setdefault("generation_budget_currency", current.generation_budget_currency)
    form.setdefault(
        "generation_budget_warning_threshold",
        str(current.generation_budget_warning_threshold),
    )
    form.setdefault(
        "planning_budget_amount", str(current.planning_budget_amount or "")
    )
    form.setdefault(
        "planning_max_paid_requests", str(current.planning_max_paid_requests)
    )
    form.setdefault("planning_max_input_tokens", str(current.planning_max_input_tokens))
    form.setdefault("planning_max_output_tokens", str(current.planning_max_output_tokens))
    form.setdefault(
        "planning_max_total_estimated_tokens",
        str(current.planning_max_total_estimated_tokens),
    )
    form.setdefault("draft_paid_visual_ratio", str(current.draft_paid_visual_ratio))
    form.setdefault("draft_width", str(current.draft_width))
    form.setdefault("draft_height", str(current.draft_height))
    project = update_project(
        session,
        project_id,
        name=_required(form, "name", "Название проекта"),
        story_text=_required(form, "story_text", "История"),
        global_image_style_prompt=form.get("global_image_style_prompt"),
        scene_count=_optional_int(form.get("scene_count")),
        planning_provider=_choice(
            form, "planning_provider", {"dashscope", "kimi"}, "Planning provider"
        ),
        planning_model=_required(form, "planning_model", "Planning model"),
        planning_budget_amount=_optional_float(
            form.get("planning_budget_amount"), "Planning budget"
        ),
        planning_max_paid_requests=int(
            _required(form, "planning_max_paid_requests", "Planning paid requests")
        ),
        planning_max_input_tokens=int(
            _required(form, "planning_max_input_tokens", "Planning input tokens")
        ),
        planning_max_output_tokens=int(
            _required(form, "planning_max_output_tokens", "Planning output tokens")
        ),
        planning_max_total_estimated_tokens=int(
            _required(
                form,
                "planning_max_total_estimated_tokens",
                "Planning total tokens",
            )
        ),
        visual_qa_enabled=form.get("visual_qa_enabled", "0") == "1",
        visual_qa_provider=_choice(
            form, "visual_qa_provider", {"dashscope"}, "Visual QA provider"
        ),
        visual_qa_model=_required(form, "visual_qa_model", "Visual QA model"),
        style_id=_required(form, "style_id", "Style ID"),
        generation_budget_enabled=form.get("generation_budget_enabled", "0") == "1",
        generation_budget_amount=_optional_float(
            form.get("generation_budget_amount"), "Ограничение бюджета"
        ),
        generation_budget_currency=_choice(
            form, "generation_budget_currency", {"EUR", "USD"}, "Валюта бюджета"
        ),
        generation_budget_warning_threshold=(
            float(
                _choice(
                    form,
                    "generation_budget_warning_threshold",
                    {"0.5", "0.7", "0.8", "0.9", "1.0"},
                    "Порог предупреждения",
                )
            )
        ),
        draft_paid_visual_ratio=float(
            _choice(
                form,
                "draft_paid_visual_ratio",
                {"0.2", "0.3", "0.4", "0.5"},
                "Доля ключевых Draft-кадров",
            )
        ),
        draft_width=int(_required(form, "draft_width", "Ширина Draft")),
        draft_height=int(_required(form, "draft_height", "Высота Draft")),
        image_provider=_choice(
            form,
            "image_provider",
            {"seedream", "qwen", "zimage"},
            "Провайдер изображений",
        ),
        image_model=form.get("image_model", "").strip() or None,
        tts_provider=_choice(
            form, "tts_provider", {"qwen", "elevenlabs"}, "Провайдер озвучки"
        ),
        tts_model=form.get("tts_model", "").strip() or None,
        tts_voice=_required(form, "tts_voice", "Голос"),
        tts_language=_required(form, "tts_language", "Язык"),
        width=dimensions[0],
        height=dimensions[1],
        fps=int(_choice(form, "fps", {"24", "30", "60"}, "FPS")),
        image_fit=_choice(
            form, "image_fit", {"cover", "contain"}, "Масштаб изображения"
        ),
    )
    assert project is not None
    return project


def _run_pipeline_worker(
    job_id: str,
    project_id: str,
    dependencies: object,
    budget_override: bool = False,
    production_profile: ProductionProfile = ProductionProfile.FINAL,
    generation_scope: GenerationScope | None = None,
    planning_run_id: str | None = None,
    allow_planning_retry_after_uncertain: bool = False,
) -> None:
    async def runner() -> None:
        async def progress(
            stage: object,
            overall: int,
            stage_progress: int,
            message: str,
            current_beat: int | None,
            total_beats: int | None,
            failed_beat: str | None,
        ) -> None:
            if job_manager.cancellation_event(job_id).is_set():
                raise asyncio.CancelledError
            await job_manager.update_pipeline_state(
                job_id,
                stage=getattr(stage, "value", str(stage)),
                progress=overall,
                stage_progress=stage_progress,
                message=message,
                current_beat=current_beat,
                total_beats=total_beats,
                failed_beat=failed_beat,
            )

        with SessionLocal() as session:
            report = await run_project_video_pipeline(
                session,
                project_id,
                dependencies,  # type: ignore[arg-type]
                progress=progress,
                job_id=job_id,
                budget_override=budget_override,
                production_profile=production_profile,
                generation_scope=generation_scope,
                planning_run_id=planning_run_id,
                allow_planning_retry_after_uncertain=(
                    allow_planning_retry_after_uncertain
                ),
                cancellation_requested=job_manager.cancellation_event(job_id).is_set,
            )
            await job_manager.set_pipeline_result(
                job_id,
                final_render_id=report.final_render_id,
                report=report.as_dict(),
            )

    asyncio.run(runner())


def _job_payload(job: GenerationJob) -> dict[str, object]:
    diagnostic = (
        job.report.get("failure")
        if isinstance(job.report, dict) and isinstance(job.report.get("failure"), dict)
        else None
    )
    budget_pause = (
        job.report.get("budget_pause")
        if isinstance(job.report, dict)
        and isinstance(job.report.get("budget_pause"), dict)
        else None
    )
    with SessionLocal() as session:
        costs = summarize_project_cost(session, job.project_id, job_id=job.id)
        current_job_attempts = list(
            session.scalars(
                select(PlanningProviderAttempt)
                .where(PlanningProviderAttempt.job_id == job.id)
                .order_by(PlanningProviderAttempt.started_at)
            )
        )
        planning_run_id = (
            current_job_attempts[-1].planning_run_id
            if current_job_attempts
            else job.id
        )
        planning_attempts = list(
            session.scalars(
                select(PlanningProviderAttempt)
                .where(PlanningProviderAttempt.planning_run_id == planning_run_id)
                .order_by(PlanningProviderAttempt.attempt_number)
            )
        )
        project = session.get(Project, job.project_id)
        planning_budget = (
            planning_budget_snapshot(
                session, job.project_id, planning_run_id
            )
            if project is not None
            else None
        )
        try:
            estimate = estimate_project_generation_cost(session, job.project_id)
            draft_estimate = estimate_project_generation_cost(
                session, job.project_id, production_profile="DRAFT"
            )
            final_estimate = estimate_project_generation_cost(
                session, job.project_id, production_profile="FINAL"
            )
            budget = ProjectBudgetGuard(session, job.project_id).snapshot()
        except ValueError:
            estimate = None
            draft_estimate = None
            final_estimate = None
            budget = None
    estimated_remaining = estimate.maximum if estimate is not None else None
    planning_progress = _planning_progress_payload(
        job,
        project,
        planning_attempts,
        planning_budget.as_dict() if planning_budget is not None else None,
    )
    return {
        "id": job.id,
        "project_id": job.project_id,
        "type": job.type.value,
        "production_profile": job.production_profile,
        "generation_scope": {
            "type": job.generation_scope_type,
            "value": job.generation_scope_value,
        },
        "status": job.status.value,
        "progress": job.progress,
        "stage_progress": job.stage_progress,
        "current_stage": job.current_stage,
        "current_beat": job.current_beat,
        "total_beats": job.total_beats,
        "message": job.message,
        "error": job.error,
        "failed_stage": job.failed_stage,
        "failed_beat": job.failed_beat,
        "final_render_id": job.final_render_id,
        "report": job.report,
        "diagnostic": diagnostic,
        "budget_pause": budget_pause,
        "planning_attempts": [
            {
                "id": item.id,
                "planning_run_id": item.planning_run_id,
                "kind": item.attempt_kind,
                "number": item.attempt_number,
                "provider": item.provider,
                "model": item.model,
                "status": item.status,
                "billing_status": item.billing_status,
                "cost_certainty": item.cost_certainty,
                "input_token_estimate": item.input_token_estimate,
                "max_output_tokens": item.configured_max_output_tokens,
                "estimated_max_cost": (
                    float(item.estimated_max_cost)
                    if item.estimated_max_cost is not None
                    else None
                ),
                "actual_cost": (
                    float(item.actual_cost) if item.actual_cost is not None else None
                ),
                "currency": item.currency,
                "provider_request_id": item.provider_request_id,
                "input_tokens": item.input_tokens,
                "output_tokens": item.output_tokens,
                "total_tokens": item.total_tokens,
                "safe_error": item.safe_error,
                "started_at": item.started_at.isoformat(),
                "finished_at": (
                    item.finished_at.isoformat() if item.finished_at else None
                ),
            }
            for item in planning_attempts
        ],
        "planning_attempt_limit": (
            project.planning_max_paid_requests if project is not None else 2
        ),
        "planning_budget": (
            planning_budget.as_dict() if planning_budget is not None else None
        ),
        "planning_progress": planning_progress,
        "budget": budget.as_dict() if budget is not None else None,
        "cost": costs.as_dict(),
        "cost_estimate": estimate.as_dict() if estimate is not None else None,
        "draft_cost_estimate": (
            draft_estimate.as_dict() if draft_estimate is not None else None
        ),
        "final_cost_estimate": (
            final_estimate.as_dict() if final_estimate is not None else None
        ),
        "estimated_remaining": estimated_remaining,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


def _planning_progress_payload(
    job: GenerationJob,
    project: Project | None,
    attempts: list[PlanningProviderAttempt],
    budget: dict[str, object] | None,
) -> dict[str, object]:
    """Build safe structured planning state without parsing logs or prompts."""
    active = attempts[-1] if attempts else None
    now = datetime.now(UTC)
    state: str | None = None
    if job.status is GenerationJobStatus.PAUSED_BUDGET and job.failed_stage == "PLANNING":
        state = "PAUSED_BUDGET"
    elif job.status is GenerationJobStatus.PAUSED_PLANNING:
        state = "PAUSED_AFTER_TIMEOUT"
    elif job.status is GenerationJobStatus.FAILED and job.failed_stage == "PLANNING":
        state = "FAILED"
    elif job.current_stage == "PLANNING":
        state = active.progress_state if active is not None else "PREPARING_SCOPE"
    elif active is not None:
        state = active.progress_state
    elif job.status is GenerationJobStatus.COMPLETED:
        state = "COMPLETED"

    elapsed_seconds: float | None = None
    request_started_at: str | None = None
    if active is not None:
        start = active.dispatched_at or active.started_at
        end = active.response_received_at or active.finished_at or now
        elapsed_seconds = round(max((end - start).total_seconds(), 0.0), 3)
        request_started_at = start.isoformat()

    scope_type = job.generation_scope_type
    scope_value = job.generation_scope_value
    scope_labels = {
        "FULL": "Full narration",
        "STYLE_PREVIEW": "Style preview — 3 images",
    }
    scope_label = scope_labels.get(scope_type, scope_type.replace("_", " ").title())
    if scope_type == "FIRST_SECONDS" and scope_value is not None:
        scope_label = f"First {scope_value:g} seconds"

    timeout = state == "PAUSED_AFTER_TIMEOUT"
    repair_reason = next(
        (
            item.validation_category
            for item in reversed(attempts)
            if item.validation_category
        ),
        None,
    )
    return {
        "state": state,
        "scope": {
            "type": scope_type,
            "value": scope_value,
            "label": scope_label,
        },
        "provider": {
            "name": active.provider if active is not None else (
                project.planning_provider if project is not None else None
            ),
            "model": active.model if active is not None else (
                project.planning_model if project is not None else None
            ),
        },
        "request": {
            "number": active.attempt_number if active is not None else None,
            "maximum": (
                project.planning_max_paid_requests if project is not None else 2
            ),
            "type": active.attempt_kind if active is not None else None,
            "status": active.status if active is not None else None,
            "started_at": request_started_at,
            "elapsed_seconds": elapsed_seconds,
            "is_running": bool(
                active is not None
                and active.progress_state == "WAITING_FOR_PROVIDER"
                and active.finished_at is None
            ),
        },
        "tokens": {
            "input_estimate": (
                active.input_token_estimate if active is not None else None
            ),
            "max_output": (
                active.configured_max_output_tokens if active is not None else None
            ),
            "actual_input": active.input_tokens if active is not None else None,
            "actual_output": active.output_tokens if active is not None else None,
        },
        "cost": {
            "estimated_max": (
                float(active.estimated_max_cost)
                if active is not None and active.estimated_max_cost is not None
                else None
            ),
            "actual_or_estimated": (
                float(active.actual_cost)
                if active is not None and active.actual_cost is not None
                else None
            ),
            "currency": active.currency if active is not None else (
                project.generation_budget_currency if project is not None else None
            ),
            "certainty": active.cost_certainty if active is not None else None,
            "budget": budget,
        },
        "repair": {
            "reason": repair_reason,
            "attempt": (
                active.attempt_number if active is not None
                and active.attempt_kind == "REPAIR" else None
            ),
            "maximum": (
                project.planning_max_paid_requests if project is not None else 2
            ),
        },
        "timeout": {
            "paused": timeout,
            "billing_unknown": bool(
                timeout
                or (active is not None and active.billing_status == "UNKNOWN")
            ),
            "automatic_retry_stopped": timeout,
            "message": (
                "The local request timed out. The provider may still have completed "
                "and billed it. Automatic retry has been stopped."
                if timeout
                else None
            ),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
