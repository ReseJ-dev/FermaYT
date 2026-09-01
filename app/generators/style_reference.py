"""Explicit registration and provider instructions for permanent style references."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from sqlalchemy.orm import Session

from app.errors import StyleReferenceError
from app.persistence import StyleReferenceAsset
from app.providers import (
    ImageEditingProvider,
    ImageProvider,
    ImageReference,
    ImageReferenceRole,
    ReferenceImageProvider,
)
from app.repositories import (
    create_style_reference_asset,
    get_style_reference_asset,
)
from app.storage import ProjectMediaPaths
from app.style_contracts import (
    DEFAULT_IMAGE_STYLE_ID,
    apply_image_style_contract,
    get_image_style_contract,
)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class StyleReferenceBoundProvider:
    """Provider adapter that permanently keeps one approved style reference first."""

    def __init__(
        self,
        provider: ImageProvider,
        style_reference: StyleReferenceAsset,
        *,
        supports_references: bool,
        supports_editing: bool = False,
    ) -> None:
        verify_style_reference_asset(style_reference)
        self._provider = provider
        self._style_reference = to_style_image_reference(style_reference)
        self._style_id = style_reference.style_id
        self._supports_references = supports_references
        self._supports_editing = supports_editing

    async def generate(self, prompt: str) -> str:
        if not self._supports_references:
            return await self._provider.generate(
                apply_image_style_contract(prompt, self._style_id)
            )
        references = (self._style_reference,)
        contracted_prompt = self._prepare_prompt(prompt, references)
        provider = self._reference_provider()
        return await provider.generate_with_references(contracted_prompt, references)

    async def generate_with_references(
        self,
        prompt: str,
        references: tuple[ImageReference, ...],
    ) -> str:
        if not self._supports_references:
            raise StyleReferenceError("Image provider does not support references")
        ordered = self._prepend_style_reference(references)
        contracted_prompt = self._prepare_prompt(prompt, ordered)
        provider = self._reference_provider()
        return await provider.generate_with_references(contracted_prompt, ordered)

    async def edit(
        self,
        prompt: str,
        references: tuple[ImageReference, ...],
    ) -> str:
        if not self._supports_editing or not isinstance(
            self._provider,
            ImageEditingProvider,
        ):
            raise StyleReferenceError("Image provider does not support editing")
        ordered = self._prepend_style_reference(references)
        contracted_prompt = self._prepare_prompt(prompt, ordered)
        return await self._provider.edit(contracted_prompt, ordered)

    def _prepend_style_reference(
        self,
        references: tuple[ImageReference, ...],
    ) -> tuple[ImageReference, ...]:
        if any(
            reference.role is ImageReferenceRole.STYLE for reference in references
        ):
            raise StyleReferenceError(
                "Callers cannot override the bound global style reference"
            )
        return (self._style_reference, *references)

    def _prepare_prompt(
        self,
        prompt: str,
        references: tuple[ImageReference, ...],
    ) -> str:
        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise ValueError("image prompt must not be empty")
        rendered_contract = get_image_style_contract(self._style_id).render()
        if normalized_prompt.endswith(rendered_contract):
            normalized_prompt = normalized_prompt[: -len(rendered_contract)].rstrip()
        role_instruction = build_reference_role_instruction(references)
        with_roles = normalized_prompt
        if role_instruction not in with_roles:
            with_roles = f"{with_roles}\n\n{role_instruction}"
        return apply_image_style_contract(with_roles, self._style_id)

    def _reference_provider(self) -> ReferenceImageProvider:
        if not isinstance(self._provider, ReferenceImageProvider):
            raise StyleReferenceError(
                "Provider capability declaration does not match its implementation"
            )
        return self._provider


def register_approved_style_reference(
    session: Session,
    project_id: str,
    source_path: str | Path,
    *,
    projects_root: str | Path = "data/projects",
    style_id: str = DEFAULT_IMAGE_STYLE_ID,
) -> StyleReferenceAsset:
    """Copy one user-approved PNG; generated frames never call this function."""
    get_image_style_contract(style_id)
    source = Path(source_path)
    _validate_png(source)
    source_hash = _sha256_file(source)

    existing = get_style_reference_asset(session, project_id, style_id)
    if existing is not None:
        verify_style_reference_asset(existing)
        if existing.file_sha256 != source_hash:
            raise StyleReferenceError(
                "A different style reference already exists for this style version"
            )
        return existing

    destination = ProjectMediaPaths(project_id, projects_root).style_reference_path(
        style_id
    )
    if destination.exists():
        raise StyleReferenceError(
            "Untracked style reference file already exists; it will not be overwritten"
        )
    try:
        shutil.copyfile(source, destination)
        _validate_png(destination)
        copied_hash = _sha256_file(destination)
        if copied_hash != source_hash:
            raise StyleReferenceError("Style reference copy failed integrity verification")
        return create_style_reference_asset(
            session,
            project_id=project_id,
            style_id=style_id,
            file_path=str(destination),
            file_sha256=copied_hash,
        )
    except StyleReferenceError:
        raise
    except Exception as exc:
        raise StyleReferenceError("Failed to register approved style reference") from exc


def verify_style_reference_asset(asset: StyleReferenceAsset) -> None:
    path = Path(asset.file_path)
    _validate_png(path)
    if _sha256_file(path) != asset.file_sha256:
        raise StyleReferenceError("Style reference failed integrity verification")


def to_style_image_reference(asset: StyleReferenceAsset) -> ImageReference:
    verify_style_reference_asset(asset)
    return ImageReference(
        reference_id=asset.style_id,
        file_path=asset.file_path,
        sha256=asset.file_sha256,
        role=ImageReferenceRole.STYLE,
    )


def build_reference_role_instruction(
    references: tuple[ImageReference, ...],
) -> str:
    """Explain reference responsibilities and deterministic conflict precedence."""
    _validate_reference_order(references)
    lines = ["IMAGE REFERENCE RESPONSIBILITIES:"]
    for index, reference in enumerate(references, start=1):
        if reference.role is ImageReferenceRole.STYLE:
            responsibility = (
                "controls line thickness, simplicity, detail ceiling, flatness, human "
                "proportions, handmade imperfections, and overall visual complexity; "
                "do not copy its objects or environment"
            )
        else:
            responsibility = (
                "controls content, location layout, recurring objects, character "
                "identity, and physical continuity"
            )
        lines.append(
            f"REFERENCE {index} [{reference.role.value}]: {responsibility}."
        )
    lines.extend(
        (
            "If references conflict, STYLE wins for realism, detail, and rendering style.",
            (
                "CONTENT_CONTINUITY wins for location layout, objects, character "
                "identity, and physical continuity."
            ),
        )
    )
    return "\n".join(lines)


def _validate_reference_order(references: tuple[ImageReference, ...]) -> None:
    style_positions = [
        index
        for index, reference in enumerate(references)
        if reference.role is ImageReferenceRole.STYLE
    ]
    if len(style_positions) > 1:
        raise StyleReferenceError("Only one global style reference is allowed")
    if style_positions and style_positions[0] != 0:
        raise StyleReferenceError("The global style reference must be REFERENCE 1")


def _validate_png(path: Path) -> None:
    if not path.is_file():
        raise StyleReferenceError("Approved style reference file is missing")
    with path.open("rb") as file_handle:
        if file_handle.read(len(_PNG_SIGNATURE)) != _PNG_SIGNATURE:
            raise StyleReferenceError("Approved style reference must be a PNG image")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
