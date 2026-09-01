"""Tests for explicit immutable visual style references."""

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.database import create_session_factory, create_sqlite_engine, init_database
from app.errors import StyleReferenceError
from app.generators.style_reference import (
    StyleReferenceBoundProvider,
    build_reference_role_instruction,
    register_approved_style_reference,
    to_style_image_reference,
    verify_style_reference_asset,
)
from app.providers import ImageReference, ImageReferenceRole
from app.repositories import create_project


@pytest.fixture
def session(tmp_path: Path) -> Iterator[Session]:
    engine = create_sqlite_engine(tmp_path / "app.db")
    init_database(engine)
    session_factory = create_session_factory(engine)
    with session_factory() as database_session:
        yield database_session
    engine.dispose()


def _project_id(session: Session) -> str:
    return create_project(
        session,
        name="Style reference project",
        story_text="A coherent visual story",
    ).id


def _png(path: Path, payload: bytes = b"approved-style") -> Path:
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + payload)
    return path


def test_approved_png_is_copied_and_registered_once(
    session: Session,
    tmp_path: Path,
) -> None:
    project_id = _project_id(session)
    source = _png(tmp_path / "STYLE_REFERENCE.png")

    first = register_approved_style_reference(
        session,
        project_id,
        source,
        projects_root=tmp_path / "projects",
    )
    second = register_approved_style_reference(
        session,
        project_id,
        source,
        projects_root=tmp_path / "projects",
    )

    assert first.id == second.id
    assert first.style_id == "rough_explainer_v1"
    assert Path(first.file_path).name == "rough_explainer_v1.png"
    assert Path(first.file_path).read_bytes() == source.read_bytes()
    assert len(first.file_sha256) == 64


def test_different_image_cannot_replace_style_reference(
    session: Session,
    tmp_path: Path,
) -> None:
    project_id = _project_id(session)
    first = _png(tmp_path / "first.png", b"first")
    second = _png(tmp_path / "second.png", b"second")
    register_approved_style_reference(
        session,
        project_id,
        first,
        projects_root=tmp_path / "projects",
    )

    with pytest.raises(StyleReferenceError, match="different style reference"):
        register_approved_style_reference(
            session,
            project_id,
            second,
            projects_root=tmp_path / "projects",
        )


def test_style_reference_integrity_is_verified(
    session: Session,
    tmp_path: Path,
) -> None:
    asset = register_approved_style_reference(
        session,
        _project_id(session),
        _png(tmp_path / "STYLE_REFERENCE.png"),
        projects_root=tmp_path / "projects",
    )
    Path(asset.file_path).write_bytes(b"\x89PNG\r\n\x1a\ntampered")

    with pytest.raises(StyleReferenceError, match="integrity"):
        verify_style_reference_asset(asset)


def test_non_png_reference_is_rejected(
    session: Session,
    tmp_path: Path,
) -> None:
    source = tmp_path / "STYLE_REFERENCE.png"
    source.write_bytes(b"not a png")

    with pytest.raises(StyleReferenceError, match="PNG"):
        register_approved_style_reference(
            session,
            _project_id(session),
            source,
            projects_root=tmp_path / "projects",
        )


def test_provider_references_put_global_style_first(
    session: Session,
    tmp_path: Path,
) -> None:
    asset = register_approved_style_reference(
        session,
        _project_id(session),
        _png(tmp_path / "STYLE_REFERENCE.png"),
        projects_root=tmp_path / "projects",
    )
    style_reference = to_style_image_reference(asset)
    continuity_reference = ImageReference(
        reference_id="mine_master",
        file_path="master.png",
        sha256="1" * 64,
        role=ImageReferenceRole.CONTENT_CONTINUITY,
    )

    instruction = build_reference_role_instruction(
        (style_reference, continuity_reference)
    )

    assert "REFERENCE 1 [STYLE]" in instruction
    assert "do not copy its objects or environment" in instruction
    assert "REFERENCE 2 [CONTENT_CONTINUITY]" in instruction
    assert "STYLE wins for realism, detail, and rendering style" in instruction
    assert "CONTENT_CONTINUITY wins for location layout" in instruction


def test_style_reference_cannot_be_placed_after_content_reference(
    session: Session,
    tmp_path: Path,
) -> None:
    asset = register_approved_style_reference(
        session,
        _project_id(session),
        _png(tmp_path / "STYLE_REFERENCE.png"),
        projects_root=tmp_path / "projects",
    )
    continuity_reference = ImageReference(
        reference_id="mine_master",
        file_path="master.png",
        sha256="1" * 64,
        role=ImageReferenceRole.CONTENT_CONTINUITY,
    )

    with pytest.raises(StyleReferenceError, match="REFERENCE 1"):
        build_reference_role_instruction(
            (continuity_reference, to_style_image_reference(asset))
        )


def test_bound_provider_always_prepends_approved_style_reference(
    session: Session,
    tmp_path: Path,
) -> None:
    asset = register_approved_style_reference(
        session,
        _project_id(session),
        _png(tmp_path / "STYLE_REFERENCE.png"),
        projects_root=tmp_path / "projects",
    )

    class ReferenceProvider:
        received: tuple[str, tuple[ImageReference, ...]] | None = None

        async def generate(self, prompt: str) -> str:
            raise AssertionError("reference path must be used")

        async def generate_with_references(
            self,
            prompt: str,
            references: tuple[ImageReference, ...],
        ) -> str:
            self.received = (prompt, references)
            return "https://example.com/image.png"

    provider = ReferenceProvider()
    bound = StyleReferenceBoundProvider(
        provider,
        asset,
        supports_references=True,
    )

    result = asyncio.run(bound.generate("A mine cutaway"))

    assert result == "https://example.com/image.png"
    assert provider.received is not None
    prompt, references = provider.received
    assert [reference.role for reference in references] == [
        ImageReferenceRole.STYLE
    ]
    assert "REFERENCE 1 [STYLE]" in prompt
    assert prompt.count("STYLE CONTRACT [rough_explainer_v1]") == 1


def test_bound_provider_keeps_style_first_when_content_reference_is_added(
    session: Session,
    tmp_path: Path,
) -> None:
    asset = register_approved_style_reference(
        session,
        _project_id(session),
        _png(tmp_path / "STYLE_REFERENCE.png"),
        projects_root=tmp_path / "projects",
    )

    class ReferenceProvider:
        received: tuple[ImageReference, ...] = ()

        async def generate(self, prompt: str) -> str:
            return "unused"

        async def generate_with_references(
            self,
            prompt: str,
            references: tuple[ImageReference, ...],
        ) -> str:
            del prompt
            self.received = references
            return "https://example.com/image.png"

    provider = ReferenceProvider()
    bound = StyleReferenceBoundProvider(
        provider,
        asset,
        supports_references=True,
    )
    content = ImageReference(
        reference_id="master",
        file_path="master.png",
        sha256="2" * 64,
        role=ImageReferenceRole.CONTENT_CONTINUITY,
    )

    asyncio.run(bound.generate_with_references("Same mine", (content,)))

    assert [reference.role for reference in provider.received] == [
        ImageReferenceRole.STYLE,
        ImageReferenceRole.CONTENT_CONTINUITY,
    ]
