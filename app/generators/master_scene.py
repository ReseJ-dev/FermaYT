"""Generation and continuity use of immutable master-scene assets."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from sqlalchemy.orm import Session

from app.errors import MasterSceneError
from app.generators.image import generate_image, validate_image_prompt
from app.models.visual_plan import MasterScene, VisualBeat, VisualOperation, VisualPlan
from app.persistence import MasterSceneAsset, Project
from app.pipeline.visual_operation_engine import VisualProviderCapabilities
from app.repositories import create_master_scene_asset, get_master_scene_asset
from app.storage import ProjectMediaPaths
from app.utils.download import download_file


class MasterSceneImageClient(Protocol):
    async def generate(self, prompt: str) -> str: ...


@runtime_checkable
class ReferenceImageClient(Protocol):
    async def generate_with_references(
        self,
        prompt: str,
        reference_image_paths: tuple[str, ...],
    ) -> str: ...


@runtime_checkable
class ImageEditingClient(Protocol):
    async def edit(self, prompt: str, source_image_path: str) -> str: ...


@dataclass(frozen=True, slots=True)
class ContinuityGenerationRequest:
    """Validated input for generating a descendant of an immutable master."""

    operation: VisualOperation
    prompt: str
    master_scene_id: str | None = None
    master_image_path: str | None = None
    style_version: str | None = None
    reference_image_paths: tuple[str, ...] = ()
    reference_hashes: tuple[str, ...] = ()


async def generate_required_master_scenes(
    session: Session,
    project: Project,
    plan: VisualPlan,
    client: MasterSceneImageClient,
    *,
    projects_root: str | Path = "data/projects",
) -> list[MasterSceneAsset]:
    """Generate referenced masters in plan order before dependent visual beats."""
    required_ids = _required_master_scene_ids(plan)
    definitions = {master.id: master for master in plan.possible_master_scenes}
    style_version = build_style_version(project.global_image_style_prompt)
    paths = ProjectMediaPaths(project.id, projects_root)
    assets: list[MasterSceneAsset] = []

    for master_scene_id in required_ids:
        definition = definitions[master_scene_id]
        existing = get_master_scene_asset(session, project.id, master_scene_id)
        if existing is not None:
            if existing.style_version != style_version:
                raise MasterSceneError(
                    "Master scene style changed; create an explicit new master version"
                )
            await verify_master_scene_asset(existing)
            assets.append(existing)
            continue

        prompt = build_master_scene_generation_prompt(
            definition,
            project.global_image_style_prompt,
        )
        output_path = str(paths.master_scene_path(master_scene_id, style_version))
        if Path(output_path).exists():
            raise MasterSceneError(
                f"Untracked master scene file already exists: {master_scene_id}"
            )
        try:
            generated_path = await generate_image(prompt, output_path, client)  # type: ignore[arg-type]
            file_sha256 = await asyncio.to_thread(_sha256_file, generated_path)
            asset = create_master_scene_asset(
                session,
                project_id=project.id,
                master_scene_id=master_scene_id,
                file_path=generated_path,
                file_sha256=file_sha256,
                style_version=style_version,
                generation_prompt=prompt,
                provider=project.image_provider,
                model=project.image_model,
                seed=None,
                reference_hashes=[],
            )
        except MasterSceneError:
            raise
        except Exception as exc:
            raise MasterSceneError(
                f"Failed to generate master scene {master_scene_id}"
            ) from exc
        assets.append(asset)

    return assets


def build_master_scene_generation_prompt(
    master: MasterScene,
    global_style_prompt: str | None,
) -> str:
    """Build and retain the exact prompt used for an immutable master."""
    parts = [
        f"MASTER SCENE ID: {master.id}",
        f"Environment: {master.description}",
        f"Stable geometry: {master.environment_geometry}",
        f"Recurring object positions: {master.recurring_object_positions}",
        f"Overall color palette: {master.color_palette}",
        f"Basic composition: {master.basic_composition}",
        (
            "Create a clear reusable environment reference. Prioritize readable "
            "spatial layout over incidental detail."
        ),
    ]
    if global_style_prompt is not None and global_style_prompt.strip():
        parts.append(f"Project visual style: {global_style_prompt.strip()}")
    return "\n".join(parts)


def build_continuity_generation_request(
    plan: VisualPlan,
    beat: VisualBeat,
    operation: VisualOperation,
    beat_prompt: str,
    master_assets: Mapping[str, MasterSceneAsset],
    capabilities: VisualProviderCapabilities,
) -> ContinuityGenerationRequest:
    """Use a master image when supported, otherwise inject its stable description."""
    normalized_prompt = validate_image_prompt(beat_prompt)
    if beat.master_scene_id is None:
        return ContinuityGenerationRequest(operation=operation, prompt=normalized_prompt)

    master = next(
        (
            item
            for item in plan.possible_master_scenes
            if item.id == beat.master_scene_id
        ),
        None,
    )
    asset = master_assets.get(beat.master_scene_id)
    if master is None or asset is None:
        raise MasterSceneError(
            f"Required master scene is unavailable: {beat.master_scene_id}"
        )
    _verify_master_scene_asset_sync(asset)

    can_use_image = (
        operation is VisualOperation.EDIT_EXISTING and capabilities.image_editing
    ) or capabilities.reference_generation
    if can_use_image:
        return ContinuityGenerationRequest(
            operation=operation,
            prompt=normalized_prompt,
            master_scene_id=master.id,
            master_image_path=asset.file_path,
            style_version=asset.style_version,
            reference_image_paths=(asset.file_path,),
            reference_hashes=(asset.file_sha256,),
        )

    continuity_description = _structured_master_description(master, asset.style_version)
    return ContinuityGenerationRequest(
        operation=operation,
        prompt=f"{normalized_prompt}\n\n{continuity_description}",
        master_scene_id=master.id,
        master_image_path=asset.file_path,
        style_version=asset.style_version,
        reference_hashes=(asset.file_sha256,),
    )


async def generate_continuity_image(
    request: ContinuityGenerationRequest,
    output_path: str,
    client: MasterSceneImageClient,
) -> str:
    """Execute a prepared request without silently ignoring reference images."""
    if request.operation is VisualOperation.EDIT_EXISTING:
        if not request.reference_image_paths or not isinstance(client, ImageEditingClient):
            raise MasterSceneError("Selected image provider cannot execute master edit")
        image_url = await client.edit(request.prompt, request.reference_image_paths[0])
    elif request.reference_image_paths:
        if not isinstance(client, ReferenceImageClient):
            raise MasterSceneError(
                "Selected image provider cannot execute reference generation"
            )
        image_url = await client.generate_with_references(
            request.prompt,
            request.reference_image_paths,
        )
    else:
        image_url = await client.generate(request.prompt)
    return await download_file(image_url, output_path)


async def verify_master_scene_asset(asset: MasterSceneAsset) -> None:
    await asyncio.to_thread(_verify_master_scene_asset_sync, asset)


def build_style_version(global_style_prompt: str | None) -> str:
    normalized_style = (global_style_prompt or "").strip().encode()
    return hashlib.sha256(normalized_style).hexdigest()[:12]


def _required_master_scene_ids(plan: VisualPlan) -> list[str]:
    master_ids = {master.id for master in plan.possible_master_scenes}
    referenced = {
        visual_id
        for beat in plan.visual_beats
        for visual_id in (
            beat.master_scene_id,
            beat.source_visual_id,
            beat.geography_established_by,
        )
        if visual_id in master_ids
    }
    return [
        master.id for master in plan.possible_master_scenes if master.id in referenced
    ]


def _structured_master_description(master: MasterScene, style_version: str) -> str:
    return "\n".join(
        (
            "IMMUTABLE MASTER SCENE CONTINUITY:",
            f"master_scene_id: {master.id}",
            f"style_version: {style_version}",
            f"environment_geometry: {master.environment_geometry}",
            f"recurring_object_positions: {master.recurring_object_positions}",
            f"color_palette: {master.color_palette}",
            f"basic_composition: {master.basic_composition}",
            (
                "Preserve this recognizable environment. Change only people, objects, "
                "water, damage, or lighting explicitly required by this beat."
            ),
        )
    )


def _verify_master_scene_asset_sync(asset: MasterSceneAsset) -> None:
    path = Path(asset.file_path)
    if not path.is_file():
        raise MasterSceneError(f"Master scene file is missing: {asset.master_scene_id}")
    if _sha256_file(path) != asset.file_sha256:
        raise MasterSceneError(
            f"Master scene file failed integrity check: {asset.master_scene_id}"
        )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
