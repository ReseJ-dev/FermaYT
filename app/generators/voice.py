"""Voice generation orchestration."""

import asyncio
from pathlib import Path
from typing import Protocol

from app.clients.tts_api import QwenTTSApiClient
from app.errors import TTSGenerationError
from app.utils.download import download_file


class TTSClient(Protocol):
    async def generate(self, text: str) -> str | bytes: ...


def validate_tts_text(text: str) -> str:
    """Return trimmed TTS text or reject a blank value."""
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("TTS text must not be empty")
    return normalized_text


async def generate_voice(
    text: str,
    output_path: str,
    client: TTSClient | None = None,
) -> str:
    """Generate speech and persist either URL-based or raw audio."""
    validated_text = validate_tts_text(text)
    tts_client = client if client is not None else QwenTTSApiClient()
    audio_result = await tts_client.generate(validated_text)
    if isinstance(audio_result, bytes):
        if not audio_result:
            raise TTSGenerationError("TTS provider returned empty audio")
        destination = Path(output_path)
        await asyncio.to_thread(destination.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(destination.write_bytes, audio_result)
        return output_path
    return await download_file(audio_result, output_path)
