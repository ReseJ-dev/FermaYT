"""Tests for image API client errors."""

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from app.clients.image_api import (
    BytePlusImageApiClient,
    ImageGenerationError,
    KieZImageApiClient,
    QwenImageApiClient,
    _fit_kie_zimage_prompt,
)
from app.providers import ImageReference, ImageReferenceRole

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


def run_zimage_generate(prompt: str = "A mountain") -> str:
    return asyncio.run(KieZImageApiClient(poll_interval=0).generate(prompt))


def test_zimage_generate_creates_and_polls_kie_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer kie-secret"
        if request.method == "POST":
            assert request.url == KieZImageApiClient.CREATE_URL
            assert json.loads(request.content) == {
                "model": "z-image",
                "input": {
                    "prompt": "A mountain",
                    "aspect_ratio": "9:16",
                    "nsfw_checker": True,
                },
            }
            return httpx.Response(
                200,
                json={"code": 200, "msg": "success", "data": {"taskId": "task-1"}},
            )
        assert request.url.params["taskId"] == "task-1"
        return httpx.Response(
            200,
            json={
                "code": 200,
                "msg": "success",
                "data": {
                    "state": "success",
                    "resultJson": json.dumps(
                        {"resultUrls": ["https://kie.example/result.png"]}
                    ),
                },
            },
        )

    monkeypatch.setenv("KIE_API_KEY", "kie-secret")
    install_mock_transport(monkeypatch, handler)

    assert run_zimage_generate() == "https://kie.example/result.png"


def test_zimage_polling_recovers_without_creating_a_second_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posts = 0
    polls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts, polls
        if request.method == "POST":
            posts += 1
            return httpx.Response(
                200,
                json={"code": 200, "data": {"taskId": "task-slow"}},
            )
        polls += 1
        if polls == 1:
            return httpx.Response(503, json={"message": "temporary overload"})
        if polls == 2:
            raise httpx.ReadTimeout("temporary polling timeout", request=request)
        return httpx.Response(
            200,
            json={
                "code": 200,
                "data": {
                    "state": "success",
                    "resultJson": '{"resultUrls":["https://kie.example/recovered.png"]}',
                },
            },
        )

    monkeypatch.setenv("KIE_API_KEY", "kie-secret")
    install_mock_transport(monkeypatch, handler)

    result = asyncio.run(
        KieZImageApiClient(poll_interval=0, generation_timeout=10).generate(
            "A mountain"
        )
    )

    assert result == "https://kie.example/recovered.png"
    assert posts == 1
    assert polls == 3


def test_zimage_generation_timeout_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIE_ZIMAGE_GENERATION_TIMEOUT_SECONDS", "2400")

    client = KieZImageApiClient()

    assert client.generation_timeout == 2400


def test_zimage_default_generation_timeout_is_ten_minutes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KIE_ZIMAGE_GENERATION_TIMEOUT_SECONDS", raising=False)

    assert KieZImageApiClient().generation_timeout == 600


@pytest.mark.parametrize("value", [0, -1])
def test_zimage_rejects_non_positive_generation_timeout(value: float) -> None:
    with pytest.raises(ValueError, match="generation timeout must be positive"):
        KieZImageApiClient(generation_timeout=value)


def test_zimage_rejects_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KIE_API_KEY", raising=False)

    with pytest.raises(ImageGenerationError, match="KIE_API_KEY") as raised:
        run_zimage_generate()

    assert raised.value.safe_diagnostic is not None
    assert raised.value.safe_diagnostic.provider == "zimage"


def test_zimage_compacts_assembled_prompt_to_kie_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_prompt = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal observed_prompt
        if request.method == "POST":
            observed_prompt = json.loads(request.content)["input"]["prompt"]
            return httpx.Response(
                200, json={"code": 200, "data": {"taskId": "task-compact"}}
            )
        return httpx.Response(
            200,
            json={
                "code": 200,
                "data": {
                    "state": "success",
                    "resultJson": '{"resultUrls":["https://kie.example/image.png"]}',
                },
            },
        )

    monkeypatch.setenv("KIE_API_KEY", "kie-secret")
    install_mock_transport(monkeypatch, handler)
    prompt = "Scene semantics " * 100 + "STYLE CONTRACT [rough_explainer_v1] " * 20

    run_zimage_generate(prompt)

    assert 1 <= len(observed_prompt) <= 800
    assert "Scene semantics" in observed_prompt
    assert "NO photorealism" in observed_prompt


def test_zimage_compaction_preserves_beat_semantics_over_verbose_project_style() -> None:
    verbose_style = "canonical master style direction " * 100
    common = f"""Create one illustration with no visible text.
LOCATION CONTINUITY:
Long mine tunnel with the exit at the far end.
PROJECT STYLE DIRECTION:
{verbose_style}
CHARACTER CONTINUITY:
Three miners in helmets.
OBJECT CONTINUITY:
Ceiling duct and work lights.
CURRENT CAMERA / COMPOSITION:
Medium view toward the far exit.
CURRENT PHYSICAL STATE:
Dust and small rocks fall from the ceiling near the exit.
WHAT CHANGED:
The stable tunnel begins shedding debris and the lights flicker.
VISUAL FOCUS:
The viewer first notices falling rock and dust near the far passage.
DO NOT SHOW:
The completed collapse.
STYLE CONTRACT [rough_explainer_v1]
Permanent style details.
"""

    compact = _fit_kie_zimage_prompt(common)
    calm = _fit_kie_zimage_prompt(
        common.replace(
            "The viewer first notices falling rock and dust near the far passage.",
            "The viewer first notices the open and safely lit far passage.",
        )
    )

    assert 1 <= len(compact) <= 800
    assert "falling rock and dust" in compact
    assert "Dust and small rocks fall" in compact
    assert "begins shedding debris" in compact
    assert "canonical master style direction" not in compact
    assert "NO photorealism" in compact
    assert compact != calm


def test_zimage_prompt_compaction_enforces_exact_provider_boundary() -> None:
    assert _fit_kie_zimage_prompt("x" * 800) == "x" * 800

    compact = _fit_kie_zimage_prompt("x" * 801)

    assert 1 <= len(compact) <= 800


def test_zimage_compaction_prioritizes_first_notice_instruction() -> None:
    prompt = f"""VISUAL FOCUS:
Purpose: {"context " * 40}
First notice: Falling rock above the tunnel exit.
STYLE CONTRACT [rough_explainer_v1]
{"style " * 200}
"""

    compact = _fit_kie_zimage_prompt(prompt)

    assert len(compact) <= 800
    assert "First notice: Falling rock above the tunnel exit" in compact


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


def test_seedream_generate_with_references_sends_image_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["image"] == [
            "https://example.com/style.png",
            "https://example.com/master.png",
        ]
        assert payload["sequential_image_generation"] == "disabled"
        return httpx.Response(
            200,
            json={"data": [{"url": "https://example.com/result.png"}]},
        )

    monkeypatch.setenv("BYTEPLUS_ARK_API_KEY", "secret-key")
    install_mock_transport(monkeypatch, handler)
    references = (
        ImageReference(
            reference_id="style",
            file_path="https://example.com/style.png",
            sha256="a" * 64,
            role=ImageReferenceRole.STYLE,
        ),
        ImageReference(
            reference_id="master",
            file_path="https://example.com/master.png",
            sha256="b" * 64,
            role=ImageReferenceRole.CONTENT_CONTINUITY,
        ),
    )

    result = asyncio.run(
        BytePlusImageApiClient().generate_with_references("Same mine", references)
    )

    assert result == "https://example.com/result.png"


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

    with pytest.raises(ImageGenerationError, match="timed out") as raised:
        run_generate()

    diagnostic = raised.value.safe_diagnostic
    assert diagnostic is not None
    assert diagnostic.error_type == "timeout"
    assert diagnostic.http_status is None
    assert diagnostic.request_stage == "provider_request"


def test_seedream_http_error_exposes_only_safe_structured_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "code": "InvalidParameter",
                "message": (
                    "Invalid size parameter; Authorization: Bearer private-token; "
                    "data:image/png;base64,QUJDREVGRw==; " + "A" * 300
                ),
                "debug": "must-not-be-exposed",
            },
        )

    monkeypatch.setenv("BYTEPLUS_ARK_API_KEY", "secret-key")
    install_mock_transport(monkeypatch, handler)

    with pytest.raises(ImageGenerationError) as raised:
        run_generate()

    diagnostic = raised.value.safe_diagnostic
    assert diagnostic is not None
    assert diagnostic.provider == "seedream"
    assert diagnostic.model == "seedream-5-0-260128"
    assert diagnostic.operation == "generate"
    assert diagnostic.error_type == "http"
    assert diagnostic.http_status == 400
    assert diagnostic.request_stage == "provider_response"
    assert "Invalid size parameter" in (diagnostic.provider_error or "")
    assert "private-token" not in (diagnostic.provider_error or "")
    assert "QUJDREVGRw" not in (diagnostic.provider_error or "")
    assert "A" * 256 not in (diagnostic.provider_error or "")
    assert "must-not-be-exposed" not in (diagnostic.provider_error or "")


def test_seedream_edit_network_error_records_operation_without_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            "connection failed Authorization: Bearer private-token",
            request=request,
        )

    monkeypatch.setenv("BYTEPLUS_ARK_API_KEY", "secret-key")
    install_mock_transport(monkeypatch, handler)
    reference = ImageReference(
        reference_id="master",
        file_path="https://example.com/master.png",
        sha256="a" * 64,
        role=ImageReferenceRole.CONTENT_CONTINUITY,
    )

    with pytest.raises(ImageGenerationError) as raised:
        asyncio.run(BytePlusImageApiClient().edit("Raise the water", (reference,)))

    diagnostic = raised.value.safe_diagnostic
    assert diagnostic is not None
    assert diagnostic.operation == "edit"
    assert diagnostic.error_type == "network"
    assert diagnostic.request_stage == "provider_request"
    assert "private-token" not in (diagnostic.provider_error or "")


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


def test_qwen_image_2_model_is_sent_to_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["model"] == "qwen-image-2.0"
        return httpx.Response(
            200,
            json={
                "output": {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"image": "https://example.com/qwen-2.png"}
                                ]
                            }
                        }
                    ]
                }
            },
        )

    install_mock_transport(monkeypatch, handler)
    client = QwenImageApiClient(
        api_key="explicit-key",
        endpoint="https://custom.example.com/qwen",
        model="qwen-image-2.0",
    )

    result = asyncio.run(client.generate("A mountain"))

    assert result == "https://example.com/qwen-2.png"


def test_qwen_edit_sends_images_before_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        content = payload["input"]["messages"][0]["content"]
        assert content == [
            {"image": "https://example.com/master.png"},
            {"text": "Raise the water in the same cave"},
        ]
        return httpx.Response(
            200,
            json={
                "output": {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"image": "https://example.com/edited.png"}
                                ]
                            }
                        }
                    ]
                }
            },
        )

    configure_qwen(monkeypatch, handler)
    reference = ImageReference(
        reference_id="master",
        file_path="https://example.com/master.png",
        sha256="c" * 64,
        role=ImageReferenceRole.CONTENT_CONTINUITY,
    )

    result = asyncio.run(
        QwenImageApiClient().edit(
            "Raise the water in the same cave",
            (reference,),
        )
    )

    assert result == "https://example.com/edited.png"


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
    ) as raised:
        run_qwen_generate()

    diagnostic = raised.value.safe_diagnostic
    assert diagnostic is not None
    assert diagnostic.provider == "qwen"
    assert diagnostic.model == "qwen-image-3.0"
    assert diagnostic.operation == "generate"
    assert diagnostic.error_type == "provider_validation"
    assert diagnostic.request_stage == "provider_response_validation"
    assert "InvalidParameter" in (diagnostic.provider_error or "")


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
