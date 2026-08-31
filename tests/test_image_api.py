"""Tests for image API client errors."""

import pytest

from app.clients.image_api import ImageGenerationError


def test_image_generation_error_preserves_message() -> None:
    with pytest.raises(ImageGenerationError, match="API request failed"):
        raise ImageGenerationError("API request failed")
