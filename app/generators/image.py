"""Image generation orchestration."""

from app.clients.image_api import ImageApiClient
from app.style_contracts import DEFAULT_IMAGE_STYLE_ID, apply_image_style_contract
from app.utils.download import download_file


def validate_image_prompt(prompt: str) -> str:
    """Return a trimmed image prompt or reject a blank value."""
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise ValueError("image prompt must not be empty")
    return normalized_prompt


def build_image_generation_prompt(
    scene_prompt: str,
    global_image_style_prompt: str | None,
) -> str:
    """Combine scene content and project style without mutating either."""
    normalized_scene_prompt = validate_image_prompt(scene_prompt)
    if global_image_style_prompt is None:
        return normalized_scene_prompt

    normalized_style_prompt = global_image_style_prompt.strip()
    if not normalized_style_prompt:
        return normalized_scene_prompt
    return f"{normalized_scene_prompt}\n\n{normalized_style_prompt}"


async def generate_image(
    prompt: str,
    output_path: str,
    client: ImageApiClient | None = None,
    *,
    style_id: str = DEFAULT_IMAGE_STYLE_ID,
) -> str:
    """Generate an image, download it, and return the output path."""
    validated_prompt = validate_image_prompt(prompt)
    contracted_prompt = apply_image_style_contract(validated_prompt, style_id)
    image_client = client if client is not None else ImageApiClient()
    image_url = await image_client.generate(contracted_prompt)
    return await download_file(image_url, output_path)
