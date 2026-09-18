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
from app.clients.kimi_ai import KimiVisualPlanningClient
from app.clients.structured_completion import missing_key_error
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
    KIE_API_KEY,
    KIMI_API_KEY,
    SecretStore,
)
from app.services.project_pipeline import ProjectPipelineDependencies


def build_production_pipeline_dependencies(
    session: Session,
    project_id: str,
    secret_store: SecretStore,
    *,
    projects_root: str | Path = "data/projects",
    require_tts: bool = True,
    require_planning: bool = True,
) -> ProjectPipelineDependencies:
    project = get_project(session, project_id)
    if project is None:
        raise ValueError("Project not found")
    settings = get_application_settings(session)
    planning_timeout = _environment_float(
        "VISUAL_PLANNING_TIMEOUT_SECONDS", 600.0, minimum=1.0
    )
    planning_max_tokens = _environment_int(
        "VISUAL_PLANNING_MAX_OUTPUT_TOKENS", 32_768, minimum=1
    )
    planning_reasoning_effort = (
        os.getenv("VISUAL_PLANNING_REASONING_EFFORT", "low").strip().lower()
    )
    if planning_reasoning_effort not in {"low", "high", "max"}:
        raise ValueError("VISUAL_PLANNING_REASONING_EFFORT must be low, high, or max")
    elevenlabs_timeout = _environment_float(
        "ELEVENLABS_TTS_TIMEOUT_SECONDS", 300.0, minimum=1.0
    )
    needs_dashscope = (
        (require_planning and project.planning_provider == "dashscope")
        or project.visual_qa_enabled
        or project.image_provider == "qwen"
        or (require_tts and project.tts_provider == "qwen")
    )
    dashscope_key = (
        secret_store.get_secret(DASHSCOPE_API_KEY)
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
        if needs_dashscope
        else None
    )
    if needs_dashscope and not dashscope_key:
        if project.planning_provider == "dashscope":
            raise missing_key_error(
                provider="dashscope",
                model=project.planning_model,
                operation="visual_planning",
                environment_name="DASHSCOPE_API_KEY",
            )
        raise ValueError("Добавьте DASHSCOPE_API_KEY в Settings")

    image_key: str
    image_endpoint: str | None = None
    if project.image_provider == "seedream":
        image_key = _secret(secret_store, BYTEPLUS_API_KEY, "BYTEPLUS_ARK_API_KEY")
    elif project.image_provider == "qwen":
        assert dashscope_key is not None
        image_key = dashscope_key
        image_endpoint = (
            settings.qwen_image_endpoint or os.getenv("QWEN_IMAGE_ENDPOINT", "").strip()
        )
        if not image_endpoint:
            raise ValueError("Настройте Qwen Image endpoint в Settings")
    elif project.image_provider == "zimage":
        image_key = (
            secret_store.get_secret(KIE_API_KEY) or os.getenv("KIE_API_KEY", "").strip()
        )
        if not image_key:
            raise ValueError("Добавьте KIE_API_KEY в Settings")
    else:
        raise ValueError("Выбран неизвестный image provider")

    tts_key: str | None = None
    if require_tts:
        if project.tts_provider == "qwen":
            assert dashscope_key is not None
            tts_key = dashscope_key
        elif project.tts_provider == "elevenlabs":
            tts_key = _secret(secret_store, ELEVENLABS_API_KEY, "ELEVENLABS_API_KEY")
        else:
            raise ValueError("Выбран неизвестный TTS provider")

    if not require_planning:
        planning_client = None
    elif project.planning_provider == "dashscope":
        assert dashscope_key is not None
        planning_client = DashScopeVisualPlanningClient(
            api_key=dashscope_key,
            model=project.planning_model,
            timeout=planning_timeout,
            max_attempts=1,
            retry_base_delay=0,
            max_output_tokens=planning_max_tokens,
        )
    elif project.planning_provider == "kimi":
        kimi_key = (
            secret_store.get_secret(KIMI_API_KEY)
            or os.getenv("MOONSHOT_API_KEY", "").strip()
            or os.getenv("KIMI_API_KEY", "").strip()
        )
        if not kimi_key:
            raise missing_key_error(
                provider="kimi",
                model=project.planning_model,
                operation="visual_planning",
                environment_name="MOONSHOT_API_KEY",
            )
        planning_client = KimiVisualPlanningClient(
            api_key=kimi_key,
            model=project.planning_model,
            timeout=planning_timeout,
            max_attempts=1,
            retry_base_delay=0,
            max_output_tokens=planning_max_tokens,
            reasoning_effort=planning_reasoning_effort,
        )
    else:
        raise ValueError("Выбран неизвестный planning provider")
    qa_service = (
        VisualQAService(
            DashScopeVisualQAClient(
                api_key=_required_key(dashscope_key),
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
        elif name == "zimage":
            options["aspect_ratio"] = _kie_aspect_ratio(project.width, project.height)
        return get_image_provider(name, options)

    def tts_resolver(
        name: str,
        config: Mapping[str, Any] | None,
    ) -> TTSProvider:
        if tts_key is None:
            raise RuntimeError("TTS is not available in image-only preview mode")
        options = dict(config or {})
        options["api_key"] = tts_key
        if name == "elevenlabs":
            options["timeout"] = elevenlabs_timeout
        return get_tts_provider(name, options)

    return ProjectPipelineDependencies(
        planning_client=planning_client,
        image_provider_resolver=image_resolver,
        tts_provider_resolver=tts_resolver,
        visual_qa_service=qa_service,
        projects_root=projects_root,
        preflight_validator=lambda current: _validate_models(
            current,
            require_tts=require_tts,
            require_planning=require_planning,
        ),
    )


def _secret(store: SecretStore, name: str, environment_name: str) -> str:
    value = store.get_secret(name) or os.getenv(environment_name, "").strip()
    if not value:
        raise ValueError(f"Добавьте {environment_name} в Settings")
    return value


def _required_key(value: str | None) -> str:
    if value is None:
        raise ValueError("Добавьте DASHSCOPE_API_KEY в Settings")
    return value


def _environment_float(name: str, default: float, *, minimum: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _environment_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _kie_aspect_ratio(width: int, height: int) -> str:
    if width == height:
        return "1:1"
    return "9:16" if height > width else "16:9"


def _validate_models(
    project: Any,
    *,
    require_tts: bool = True,
    require_planning: bool = True,
) -> None:
    required = {"image model": project.image_model}
    if require_planning:
        required["planning model"] = project.planning_model
    if require_tts:
        required["TTS model"] = project.tts_model
    if project.visual_qa_enabled:
        required["Visual QA model"] = project.visual_qa_model
    missing = [
        label for label, value in required.items() if not str(value or "").strip()
    ]
    if missing:
        raise ValueError("Не настроено: " + ", ".join(missing))
