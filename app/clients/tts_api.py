"""Text-to-speech API clients."""

import os
from typing import ClassVar

import httpx

from app.errors import TTSGenerationError
from app.generators.voice import validate_tts_text


class QwenTTSApiClient:
    """Minimal client for Qwen3-TTS-Flash on Alibaba Model Studio."""

    API_URL: ClassVar[str] = (
        "https://dashscope-intl.aliyuncs.com/api/v1/services/"
        "aigc/multimodal-generation/generation"
    )
    MODEL_ID: ClassVar[str] = "qwen3-tts-flash"
    VOICE: ClassVar[str] = "Cherry"
    LANGUAGE_TYPE: ClassVar[str] = "Russian"
    TIMEOUT_SECONDS: ClassVar[float] = 30.0

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        model: str = MODEL_ID,
        voice: str = VOICE,
        language: str = LANGUAGE_TYPE,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint if endpoint is not None else self.API_URL
        self.model = model
        self.voice = voice
        self.language = language

    async def generate(self, text: str) -> str:
        """Generate speech and return its temporary audio URL."""
        try:
            validated_text = validate_tts_text(text)
        except ValueError as exc:
            raise TTSGenerationError("Invalid TTS text") from exc

        api_key = (
            self.api_key
            if self.api_key is not None
            else os.getenv("DASHSCOPE_API_KEY")
        )
        if not api_key or not api_key.strip():
            raise TTSGenerationError(
                "DASHSCOPE_API_KEY environment variable is not set"
            )

        payload = {
            "model": self.model,
            "input": {
                "text": validated_text,
                "voice": self.voice,
                "language_type": self.language,
            },
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.TIMEOUT_SECONDS
            ) as client:
                response = await client.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise TTSGenerationError("TTS API request timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise TTSGenerationError(
                f"TTS API returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise TTSGenerationError("TTS API request failed") from exc

        try:
            response_data = response.json()
        except ValueError as exc:
            raise TTSGenerationError(
                "TTS API returned invalid JSON"
            ) from exc

        if not isinstance(response_data, dict):
            raise TTSGenerationError(
                "TTS API response does not contain output"
            )

        error_code = response_data.get("code")
        error_message = response_data.get("message")
        if error_code or error_message:
            raise TTSGenerationError(
                f"TTS API error {error_code}: {error_message}"
            )

        output = response_data.get("output")
        if not isinstance(output, dict):
            raise TTSGenerationError(
                "TTS API response does not contain output"
            )

        audio = output.get("audio")
        if not isinstance(audio, dict):
            raise TTSGenerationError(
                "TTS API response does not contain audio"
            )

        audio_url = audio.get("url")
        if not isinstance(audio_url, str) or not audio_url.strip():
            raise TTSGenerationError(
                "TTS API response does not contain an audio URL"
            )

        return audio_url.strip()


TTSApiClient = QwenTTSApiClient
