"""Tests for the Scene model."""

from uuid import UUID

import pytest
from pydantic import ValidationError

from app.models.scene import Scene


def test_scene_generates_uuid() -> None:
    scene = Scene(text="Narration", image_prompt="A quiet forest")

    assert str(UUID(scene.id)) == scene.id


@pytest.mark.parametrize("field", ["text", "image_prompt"])
@pytest.mark.parametrize("value", ["", "   "])
def test_scene_rejects_empty_text_fields(field: str, value: str) -> None:
    data = {"text": "Narration", "image_prompt": "A quiet forest"}
    data[field] = value

    with pytest.raises(ValidationError):
        Scene(**data)


@pytest.mark.parametrize("duration", [0, -1, -0.1])
def test_scene_rejects_non_positive_duration(duration: float) -> None:
    with pytest.raises(ValidationError):
        Scene(
            text="Narration",
            image_prompt="A quiet forest",
            duration=duration,
        )


def test_scene_accepts_positive_duration() -> None:
    scene = Scene(
        text="Narration",
        image_prompt="A quiet forest",
        duration=2.5,
    )

    assert scene.duration == 2.5
