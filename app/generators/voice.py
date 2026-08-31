"""Voice generation orchestration."""

from app.clients.tts_api import QwenTTSApiClient
from app.utils.download import download_file


def validate_tts_text(text: str) -> str:
    """Return trimmed TTS text or reject a blank value."""
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("TTS text must not be empty")
    return normalized_text


async def generate_voice(
    text: str,
    output_path: str,
    client: QwenTTSApiClient | None = None,
) -> str:
    """Generate speech, download it, and return the output path."""
    validated_text = validate_tts_text(text)
    tts_client = client if client is not None else QwenTTSApiClient()
    audio_url = await tts_client.generate(validated_text)
    return await download_file(audio_url, output_path)
