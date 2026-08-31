"""Image generation orchestration."""


def validate_image_prompt(prompt: str) -> str:
    """Return a trimmed image prompt or reject a blank value."""
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise ValueError("image prompt must not be empty")
    return normalized_prompt
