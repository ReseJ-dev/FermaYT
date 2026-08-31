"""Tests for the generated SceneDraft DTO."""

import pytest
from pydantic import ValidationError

from app.models.scene_draft import SceneDraft


def test_scene_draft_trims_text_fields() -> None:
    draft = SceneDraft(
        text="  A boy opens the magic book  ",
        image_prompt="  A glowing book in a dark attic  ",
    )

    assert draft.text == "A boy opens the magic book"
    assert draft.image_prompt == "A glowing book in a dark attic"


@pytest.mark.parametrize("field", ["text", "image_prompt"])
@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_scene_draft_rejects_blank_fields(field: str, value: str) -> None:
    values = {
        "text": "Narration",
        "image_prompt": "Visual description",
    }
    values[field] = value

    with pytest.raises(ValidationError):
        SceneDraft(**values)


@pytest.mark.parametrize("missing_field", ["text", "image_prompt"])
def test_scene_draft_requires_both_fields(missing_field: str) -> None:
    values = {
        "text": "Narration",
        "image_prompt": "Visual description",
    }
    values.pop(missing_field)

    with pytest.raises(ValidationError):
        SceneDraft(**values)


def test_scene_draft_contains_only_generated_content() -> None:
    draft = SceneDraft(text="Narration", image_prompt="Visual description")

    assert draft.model_dump() == {
        "text": "Narration",
        "image_prompt": "Visual description",
    }


def test_scene_draft_rejects_persistence_fields() -> None:
    with pytest.raises(ValidationError):
        SceneDraft(
            text="Narration",
            image_prompt="Visual description",
            id="scene-1",
        )
