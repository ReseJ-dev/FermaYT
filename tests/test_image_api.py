"""Tests for image API client errors."""

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from app.clients.image_api import (
    BytePlusImageApiClient,
    ImageGenerationError,
    QwenImageApiClient,
)


ResponseHandler = Callable[[httpx.Request], httpx.Response]


def install_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: ResponseHandler,
) -> None:
    real_async_client = httpx.AsyncClient

    def create_mock_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", create_mock_client)


def run_generate(prompt: str = "A mountain") -> str:
    return asyncio.run(BytePlusImageApiClient().generate(prompt))


def run_qwen_generate(prompt: str = "A mountain") -> str:
    return asyncio.run(QwenImageApiClient().generate(prompt))


def configure_qwen(
    monkeypatch: pytest.MonkeyPatch,
    handler: ResponseHandler,
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "qwen-secret-key")
    monkeypatch.setenv(
        "QWEN_IMAGE_ENDPOINT",
        "https://workspace.ap-southeast-1.maas.aliyuncs.com/api/v1/generation",
    )
    install_mock_transport(monkeypatch, handler)


def test_image_generation_error_preserves_message() -> None:
    with pytest.raises(ImageGenerationError, match="API request failed"):
        raise ImageGenerationError("API request failed")


def test_generate_returns_image_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == BytePlusImageApiClient.API_URL
        assert request.headers["Authorization"] == "Bearer secret-key"
        assert request.headers["Content-Type"] == "application/json"
        assert json.loads(request.content) == {
            "model": "seedream-5-0-260128",
            "prompt": "A mountain",
            "size": "2K",
            "output_format": "png",
            "response_format": "url",
            "watermark": False,
            "sequential_image_generation": "disabled",
            "stream": False,
        }
        return httpx.Response(
            200,
            json={"data": [{"url": "https://example.com/image.png"}]},
        )

    monkeypatch.setenv("BYTEPLUS_ARK_API_KEY", "secret-key")
    install_mock_transport(monkeypatch, handler)

    assert run_generate() == "https://example.com/image.png"


def test_seedream_uses_explicit_constructor_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://custom.example.com/seedream"
        assert request.headers["Authorization"] == "Bearer explicit-key"
        assert json.loads(request.content)["model"] == "custom-seedream"
        return httpx.Response(
            200,
            json={"data": [{"url": "https://example.com/image.png"}]},
        )

    monkeypatch.setenv("BYTEPLUS_ARK_API_KEY", "environment-key")
    install_mock_transport(monkeypatch, handler)
    client = BytePlusImageApiClient(
        api_key="explicit-key",
        endpoint="https://custom.example.com/seedream",
        model="custom-seedream",
    )

    result = asyncio.run(client.generate("A mountain"))

    assert result == "https://example.com/image.png"


@pytest.mark.parametrize("prompt", ["", "   "])
def test_generate_rejects_empty_prompt(prompt: str) -> None:
    with pytest.raises(ImageGenerationError, match="Invalid image prompt"):
        run_generate(prompt)


def test_generate_rejects_missing_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BYTEPLUS_ARK_API_KEY", raising=False)

    with pytest.raises(ImageGenerationError, match="BYTEPLUS_ARK_API_KEY"):
        run_generate()


def test_generate_handles_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    monkeypatch.setenv("BYTEPLUS_ARK_API_KEY", "secret-key")
    install_mock_transport(monkeypatch, handler)

    with pytest.raises(ImageGenerationError, match="timed out"):
        run_generate()


@pytest.mark.parametrize("status_code", [400, 429, 500])
def test_generate_handles_http_errors(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"error": "failure"})

    monkeypatch.setenv("BYTEPLUS_ARK_API_KEY", "secret-key")
    install_mock_transport(monkeypatch, handler)

    with pytest.raises(ImageGenerationError, match=f"HTTP {status_code}"):
        run_generate()


def test_generate_handles_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    monkeypatch.setenv("BYTEPLUS_ARK_API_KEY", "secret-key")
    install_mock_transport(monkeypatch, handler)

    with pytest.raises(ImageGenerationError, match="invalid JSON"):
        run_generate()


@pytest.mark.parametrize(
    "response_data",
    [
        {},
        [],
        {"data": []},
    ],
)
def test_generate_requires_non_empty_data(
    monkeypatch: pytest.MonkeyPatch,
    response_data: object,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_data)

    monkeypatch.setenv("BYTEPLUS_ARK_API_KEY", "secret-key")
    install_mock_transport(monkeypatch, handler)

    with pytest.raises(ImageGenerationError, match="image data"):
        run_generate()


@pytest.mark.parametrize(
    "response_data",
    [
        {"data": [None]},
        {"data": [{}]},
        {"data": [{"url": ""}]},
    ],
)
def test_generate_requires_image_url(
    monkeypatch: pytest.MonkeyPatch,
    response_data: dict[str, object],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_data)

    monkeypatch.setenv("BYTEPLUS_ARK_API_KEY", "secret-key")
    install_mock_transport(monkeypatch, handler)

    with pytest.raises(ImageGenerationError, match="image URL"):
        run_generate()


def test_qwen_generate_returns_image_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == (
            "https://workspace.ap-southeast-1.maas.aliyuncs.com/"
            "api/v1/generation"
        )
        assert request.headers["Authorization"] == "Bearer qwen-secret-key"
        assert request.headers["Content-Type"] == "application/json"
        assert json.loads(request.content) == {
            "model": "qwen-image-3.0",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": "A mountain"}],
                    }
                ]
            },
            "parameters": {
                "prompt_extend": True,
                "n": 1,
                "watermark": False,
            },
        }
        return httpx.Response(
            200,
            json={
                "output": {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"image": "https://example.com/qwen.png"}
                                ]
                            }
                        }
                    ]
                }
            },
        )

    configure_qwen(monkeypatch, handler)

    assert run_qwen_generate() == "https://example.com/qwen.png"


def test_qwen_uses_explicit_constructor_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://custom.example.com/qwen"
        assert request.headers["Authorization"] == "Bearer explicit-key"
        assert json.loads(request.content)["model"] == "custom-qwen"
        return httpx.Response(
            200,
            json={
                "output": {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"image": "https://example.com/qwen.png"}
                                ]
                            }
                        }
                    ]
                }
            },
        )

    monkeypatch.setenv("DASHSCOPE_API_KEY", "environment-key")
    monkeypatch.setenv("QWEN_IMAGE_ENDPOINT", "https://env.example.com")
    install_mock_transport(monkeypatch, handler)
    client = QwenImageApiClient(
        api_key="explicit-key",
        endpoint="https://custom.example.com/qwen",
        model="custom-qwen",
    )

    result = asyncio.run(client.generate("A mountain"))

    assert result == "https://example.com/qwen.png"


@pytest.mark.parametrize("prompt", ["", "   "])
def test_qwen_generate_rejects_empty_prompt(prompt: str) -> None:
    with pytest.raises(ImageGenerationError, match="Invalid image prompt"):
        run_qwen_generate(prompt)


def test_qwen_generate_rejects_missing_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("QWEN_IMAGE_ENDPOINT", "https://example.com/generate")

    with pytest.raises(ImageGenerationError, match="DASHSCOPE_API_KEY"):
        run_qwen_generate()


def test_qwen_generate_rejects_missing_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "qwen-secret-key")
    monkeypatch.delenv("QWEN_IMAGE_ENDPOINT", raising=False)

    with pytest.raises(ImageGenerationError, match="QWEN_IMAGE_ENDPOINT"):
        run_qwen_generate()


def test_qwen_generate_handles_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    configure_qwen(monkeypatch, handler)

    with pytest.raises(ImageGenerationError, match="timed out"):
        run_qwen_generate()


@pytest.mark.parametrize("status_code", [400, 429, 500])
def test_qwen_generate_handles_http_errors(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"message": "failure"})

    configure_qwen(monkeypatch, handler)

    with pytest.raises(ImageGenerationError, match=f"HTTP {status_code}"):
        run_qwen_generate()


def test_qwen_generate_handles_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    configure_qwen(monkeypatch, handler)

    with pytest.raises(ImageGenerationError, match="invalid JSON"):
        run_qwen_generate()


def test_qwen_generate_handles_api_error_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"code": "InvalidParameter", "message": "Bad prompt"},
        )

    configure_qwen(monkeypatch, handler)

    with pytest.raises(
        ImageGenerationError,
        match="InvalidParameter: Bad prompt",
    ):
        run_qwen_generate()


def test_qwen_generate_requires_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    configure_qwen(monkeypatch, handler)

    with pytest.raises(ImageGenerationError, match="output"):
        run_qwen_generate()


@pytest.mark.parametrize(
    "response_data",
    [
        {"output": {}},
        {"output": {"choices": []}},
    ],
)
def test_qwen_generate_requires_non_empty_choices(
    monkeypatch: pytest.MonkeyPatch,
    response_data: dict[str, object],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_data)

    configure_qwen(monkeypatch, handler)

    with pytest.raises(ImageGenerationError, match="choices"):
        run_qwen_generate()


@pytest.mark.parametrize(
    "response_data",
    [
        {"output": {"choices": [{}]}},
        {
            "output": {
                "choices": [{"message": {"content": [{}]}}]
            }
        },
        {
            "output": {
                "choices": [
                    {"message": {"content": [{"image": ""}]}}
                ]
            }
        },
    ],
)
def test_qwen_generate_requires_image_url(
    monkeypatch: pytest.MonkeyPatch,
    response_data: dict[str, object],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_data)

    configure_qwen(monkeypatch, handler)

    with pytest.raises(ImageGenerationError, match="image URL"):
        run_qwen_generate()
