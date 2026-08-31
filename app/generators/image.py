"""Image generation orchestration."""

from app.clients.image_api import ImageApiClient
from app.utils.download import download_file


def validate_image_prompt(prompt: str) -> str:
    """Return a trimmed image prompt or reject a blank value."""
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise ValueError("image prompt must not be empty")
    return normalized_prompt


async def generate_image(
    prompt: str,
    output_path: str,
    client: ImageApiClient | None = None,
) -> str:
    """Generate an image, download it, and return the output path."""
    validated_prompt = validate_image_prompt(prompt)
    image_client = client if client is not None else ImageApiClient()
    image_url = await image_client.generate(validated_prompt)
    return await download_file(image_url, output_path)
