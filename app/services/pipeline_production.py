"""Production provider wiring and pre-flight validation for Stage 7."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.clients.dashscope_ai import (
    DashScopeVisualPlanningClient,
    DashScopeVisualQAClient,
)
from app.pipeline.visual_qa import VisualQAService
from app.providers import (
    ImageProvider,
    TTSProvider,
    get_image_provider,
    get_tts_provider,
)
from app.repositories import get_application_settings, get_project
from app.secret_store import (
    BYTEPLUS_API_KEY,
    DASHSCOPE_API_KEY,
    ELEVENLABS_API_KEY,
    SecretStore,
)
from app.services.project_pipeline import ProjectPipelineDependencies


def build_production_pipeline_dependencies(
    session: Session,
    project_id: str,
    secret_store: SecretStore,
    *,
    projects_root: str | Path = "data/projects",
) -> ProjectPipelineDependencies:
    project = get_project(session, project_id)
    if project is None:
        raise ValueError("Project not found")
    settings = get_application_settings(session)
    dashscope_key = _secret(secret_store, DASHSCOPE_API_KEY, "DASHSCOPE_API_KEY")

    image_key: str
    image_endpoint: str | None = None
    if project.image_provider == "seedream":
        image_key = _secret(secret_store, BYTEPLUS_API_KEY, "BYTEPLUS_ARK_API_KEY")
    elif project.image_provider == "qwen":
        image_key = dashscope_key
        image_endpoint = (
            settings.qwen_image_endpoint or os.getenv("QWEN_IMAGE_ENDPOINT", "").strip()
        )
        if not image_endpoint:
            raise ValueError("Настройте Qwen Image endpoint в Settings")
    else:
        raise ValueError("Выбран неизвестный image provider")

    if project.tts_provider == "qwen":
        tts_key = dashscope_key
    elif project.tts_provider == "elevenlabs":
        tts_key = _secret(secret_store, ELEVENLABS_API_KEY, "ELEVENLABS_API_KEY")
    else:
        raise ValueError("Выбран неизвестный TTS provider")

    planning_client = DashScopeVisualPlanningClient(
        api_key=dashscope_key,
        model=project.planning_model,
    )
    qa_service = (
        VisualQAService(
            DashScopeVisualQAClient(
                api_key=dashscope_key,
                model=project.visual_qa_model,
            ),
            provider=project.visual_qa_provider,
            model=project.visual_qa_model,
        )
        if project.visual_qa_enabled
        else None
    )

    def image_resolver(
        name: str,
        config: Mapping[str, Any] | None,
    ) -> ImageProvider:
        options = dict(config or {})
        options["api_key"] = image_key
        if name == "qwen":
            options["endpoint"] = image_endpoint
        return get_image_provider(name, options)

    def tts_resolver(
        name: str,
        config: Mapping[str, Any] | None,
    ) -> TTSProvider:
        options = dict(config or {})
        options["api_key"] = tts_key
        return get_tts_provider(name, options)

    return ProjectPipelineDependencies(
        planning_client=planning_client,
        image_provider_resolver=image_resolver,
        tts_provider_resolver=tts_resolver,
        visual_qa_service=qa_service,
        projects_root=projects_root,
        preflight_validator=lambda current: _validate_models(current),
    )


def _secret(store: SecretStore, name: str, environment_name: str) -> str:
    value = store.get_secret(name) or os.getenv(environment_name, "").strip()
    if not value:
        raise ValueError(f"Добавьте {environment_name} в Settings")
    return value


def _validate_models(project: Any) -> None:
    required = {
        "planning model": project.planning_model,
        "image model": project.image_model,
        "TTS model": project.tts_model,
    }
    if project.visual_qa_enabled:
        required["Visual QA model"] = project.visual_qa_model
    missing = [
        label for label, value in required.items() if not str(value or "").strip()
    ]
    if missing:
        raise ValueError("Не настроено: " + ", ".join(missing))
