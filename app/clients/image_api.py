"""Image generation API client."""

import os
from typing import ClassVar

import httpx


class ImageGenerationError(Exception):
    """Raised when an external image generation API fails."""


class BytePlusImageApiClient:
    """Minimal client for BytePlus ModelArk image generation."""

    API_URL: ClassVar[str] = (
        "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
    )
    MODEL_ID: ClassVar[str] = "seedream-5-0-260128"
    TIMEOUT_SECONDS: ClassVar[float] = 30.0

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        model: str = MODEL_ID,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint if endpoint is not None else self.API_URL
        self.model = model

    async def generate(self, prompt: str) -> str:
        """Generate one image and return its temporary URL."""
        from app.generators.image import validate_image_prompt

        try:
            validated_prompt = validate_image_prompt(prompt)
        except ValueError as exc:
            raise ImageGenerationError("Invalid image prompt") from exc

        api_key = (
            self.api_key
            if self.api_key is not None
            else os.getenv("BYTEPLUS_ARK_API_KEY")
        )
        if not api_key or not api_key.strip():
            raise ImageGenerationError(
                "BYTEPLUS_ARK_API_KEY environment variable is not set"
            )

        payload: dict[str, str | bool] = {
            "model": self.model,
            "prompt": validated_prompt,
            "size": "2K",
            "output_format": "png",
            "response_format": "url",
            "watermark": False,
            "sequential_image_generation": "disabled",
            "stream": False,
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
            raise ImageGenerationError(
                "BytePlus Image API request timed out"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ImageGenerationError(
                "BytePlus Image API returned "
                f"HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise ImageGenerationError(
                "BytePlus Image API request failed"
            ) from exc

        try:
            response_data = response.json()
        except ValueError as exc:
            raise ImageGenerationError(
                "BytePlus Image API returned invalid JSON"
            ) from exc

        if not isinstance(response_data, dict):
            raise ImageGenerationError(
                "BytePlus Image API response does not contain image data"
            )

        images = response_data.get("data")
        if not isinstance(images, list) or not images:
            raise ImageGenerationError(
                "BytePlus Image API response does not contain image data"
            )

        first_image = images[0]
        if not isinstance(first_image, dict):
            raise ImageGenerationError(
                "BytePlus Image API response does not contain an image URL"
            )

        image_url = first_image.get("url")
        if not isinstance(image_url, str) or not image_url.strip():
            raise ImageGenerationError(
                "BytePlus Image API response does not contain an image URL"
            )

        return image_url.strip()


class QwenImageApiClient:
    """Minimal client for Qwen Image on Alibaba Cloud Model Studio."""

    MODEL_ID: ClassVar[str] = "qwen-image-3.0"
    TIMEOUT_SECONDS: ClassVar[float] = 30.0

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        model: str = MODEL_ID,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model

    async def generate(self, prompt: str) -> str:
        """Generate one image and return its temporary URL."""
        from app.generators.image import validate_image_prompt

        try:
            validated_prompt = validate_image_prompt(prompt)
        except ValueError as exc:
            raise ImageGenerationError("Invalid image prompt") from exc

        api_key = (
            self.api_key
            if self.api_key is not None
            else os.getenv("DASHSCOPE_API_KEY")
        )
        if not api_key or not api_key.strip():
            raise ImageGenerationError(
                "DASHSCOPE_API_KEY environment variable is not set"
            )

        endpoint = (
            self.endpoint
            if self.endpoint is not None
            else os.getenv("QWEN_IMAGE_ENDPOINT")
        )
        if not endpoint or not endpoint.strip():
            raise ImageGenerationError(
                "QWEN_IMAGE_ENDPOINT environment variable is not set"
            )

        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": validated_prompt}],
                    }
                ]
            },
            "parameters": {
                "prompt_extend": True,
                "n": 1,
                "watermark": False,
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
                    endpoint.strip(),
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ImageGenerationError(
                "Qwen Image API request timed out"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ImageGenerationError(
                f"Qwen Image API returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise ImageGenerationError(
                "Qwen Image API request failed"
            ) from exc

        try:
            response_data = response.json()
        except ValueError as exc:
            raise ImageGenerationError(
                "Qwen Image API returned invalid JSON"
            ) from exc

        if not isinstance(response_data, dict):
            raise ImageGenerationError(
                "Qwen Image API response does not contain output"
            )

        error_code = response_data.get("code")
        error_message = response_data.get("message")
        if error_code or error_message:
            raise ImageGenerationError(
                f"Qwen Image API error {error_code}: {error_message}"
            )

        output = response_data.get("output")
        if not isinstance(output, dict):
            raise ImageGenerationError(
                "Qwen Image API response does not contain output"
            )

        choices = output.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ImageGenerationError(
                "Qwen Image API response does not contain choices"
            )

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ImageGenerationError(
                "Qwen Image API response does not contain an image URL"
            )

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ImageGenerationError(
                "Qwen Image API response does not contain an image URL"
            )

        content = message.get("content")
        if not isinstance(content, list) or not content:
            raise ImageGenerationError(
                "Qwen Image API response does not contain an image URL"
            )

        first_content = content[0]
        if not isinstance(first_content, dict):
            raise ImageGenerationError(
                "Qwen Image API response does not contain an image URL"
            )

        image_url = first_content.get("image")
        if not isinstance(image_url, str) or not image_url.strip():
            raise ImageGenerationError(
                "Qwen Image API response does not contain an image URL"
            )

        return image_url.strip()


SeedreamImageProvider = BytePlusImageApiClient
QwenImageProvider = QwenImageApiClient
ImageApiClient = SeedreamImageProvider
