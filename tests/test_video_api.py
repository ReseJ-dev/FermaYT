"""Provider adapter request/response mappings with no paid network calls."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from app.clients.video_api import (
    SeedanceVideoProvider,
    ViduVideoProvider,
    WanVideoProvider,
)
from app.errors import VideoGenerationError
from app.video_model_profiles import (
    MODEL_PROFILES,
    VideoPayloadStrategy,
    get_verified_video_model_profile,
)
from app.video_providers import (
    RemoteVideoTaskState,
    VideoGenerationRequest,
    VideoOperation,
    VideoReference,
    VideoReferenceRole,
)


def install_mock_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    real_async_client = httpx.AsyncClient

    def create_mock_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", create_mock_client)


def _request(tmp_path: Path) -> VideoGenerationRequest:
    image = tmp_path / "source.png"
    image.write_bytes(b"png")
    return VideoGenerationRequest(
        VideoOperation.IMAGE_TO_VIDEO,
        "Miner slowly turns his head.",
        5,
        "720p",
        "16:9",
        references=(
            VideoReference(
                "first",
                VideoReferenceRole.FIRST_FRAME,
                str(image),
                hashlib.sha256(image.read_bytes()).hexdigest(),
            ),
        ),
    )


def _remote_reference(
    reference_id: str, role: VideoReferenceRole, url: str
) -> VideoReference:
    return VideoReference(
        reference_id,
        role,
        url,
        hashlib.sha256(url.encode()).hexdigest(),
    )


def _capture_submission(
    monkeypatch: pytest.MonkeyPatch,
    provider,
    request: VideoGenerationRequest,
    *,
    task_response: dict | None = None,
) -> tuple[str, dict]:
    captured: dict[str, object] = {}

    def handler(http_request: httpx.Request) -> httpx.Response:
        captured["url"] = str(http_request.url)
        captured["payload"] = json.loads(http_request.content)
        return httpx.Response(200, json=task_response or {"task_id": "task-1"})

    install_mock_transport(monkeypatch, handler)
    asyncio.run(provider.submit(request))
    return str(captured["url"]), captured["payload"]  # type: ignore[return-value]


def test_vidu_create_and_poll_same_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        assert request.headers["Authorization"] == "Token secret"
        if request.method == "POST":
            payload = json.loads(request.content)
            assert payload["audio"] is False
            assert payload["bgm"] is False
            return httpx.Response(200, json={"task_id": "vidu-1", "state": "created"})
        return httpx.Response(
            200,
            json={
                "state": "success",
                "creations": [{"url": "https://x/v.mp4"}],
                "credits": 4,
            },
        )

    install_mock_transport(monkeypatch, handler)
    provider = ViduVideoProvider(api_key="secret", endpoint="https://vidu.test/ent/v2")
    submission = asyncio.run(provider.submit(_request(tmp_path)))
    result = asyncio.run(provider.get_task_status(submission.remote_task_id))
    assert submission.remote_task_id == "vidu-1"
    assert result.state is RemoteVideoTaskState.SUCCEEDED
    assert result.consumed_credits == 4
    assert seen[-1].endswith("/tasks/vidu-1/creations")


def test_wan_create_and_poll_same_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            assert request.headers["X-DashScope-Async"] == "enable"
            payload = json.loads(request.content)
            assert payload["model"] == "wan2.7-i2v-2026-04-25"
            return httpx.Response(200, json={"output": {"task_id": "wan-1"}})
        assert str(request.url).endswith("/tasks/wan-1")
        return httpx.Response(
            200,
            json={
                "output": {"task_status": "SUCCEEDED", "video_url": "https://x/w.mp4"}
            },
        )

    install_mock_transport(monkeypatch, handler)
    provider = WanVideoProvider(api_key="secret", endpoint="https://wan.test/api/v1")
    submission = asyncio.run(provider.submit(_request(tmp_path)))
    result = asyncio.run(provider.get_task_status(submission.remote_task_id))
    assert submission.remote_task_id == "wan-1"
    assert result.state is RemoteVideoTaskState.SUCCEEDED


def test_seedance_create_and_poll_same_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            payload = json.loads(request.content)
            assert "generate_audio" not in payload
            assert payload["content"][1]["role"] == "first_frame"
            return httpx.Response(200, json={"id": "seedance-1", "status": "queued"})
        assert str(request.url).endswith("/contents/generations/tasks/seedance-1")
        return httpx.Response(
            200,
            json={"status": "succeeded", "content": {"video_url": "https://x/s.mp4"}},
        )

    install_mock_transport(monkeypatch, handler)
    provider = SeedanceVideoProvider(
        api_key="secret", endpoint="https://seed.test/api/v3"
    )
    submission = asyncio.run(provider.submit(_request(tmp_path)))
    result = asyncio.run(provider.get_task_status(submission.remote_task_id))
    assert submission.remote_task_id == "seedance-1"
    assert result.state is RemoteVideoTaskState.SUCCEEDED


def test_seedance_known_model_capabilities_follow_official_model_variants() -> None:
    pro = SeedanceVideoProvider(api_key="secret", model="seedance-1-0-pro-250528")
    fast = SeedanceVideoProvider(api_key="secret", model="seedance-1-0-pro-fast-251015")

    assert pro.capabilities.supports_text_to_video is True
    assert pro.capabilities.supports_first_last_frame is True
    assert fast.capabilities.supports_text_to_video is True
    assert fast.capabilities.supports_image_to_video is True
    assert fast.capabilities.supports_first_last_frame is False


def test_exact_known_models_use_verified_profiles() -> None:
    for provider_id, model_id in (
        ("vidu", "viduq3-turbo"),
        ("wan", "wan2.7-i2v-2026-04-25"),
        ("wan", "wan2.7-t2v-2026-06-12"),
        ("wan", "wan2.7-videoedit"),
        ("seedance", "seedance-1-0-lite-i2v-250428"),
        ("seedance", "seedance-1-0-pro-250528"),
        ("seedance", "seedance-1-0-pro-fast-251015"),
    ):
        profile = get_verified_video_model_profile(provider_id, model_id)
        assert profile is not None
        assert profile.verified is True
        assert profile.official_documentation


@pytest.mark.parametrize(
    ("provider", "model"),
    (
        (ViduVideoProvider, "viduq3-turbo-experimental"),
        (WanVideoProvider, "my-wan3-everything"),
        (WanVideoProvider, "unsafe-edit-model"),
        (SeedanceVideoProvider, "seedance-1-0-pro-future"),
    ),
)
def test_unknown_model_names_never_inherit_paid_capabilities(provider, model: str) -> None:
    instance = provider(api_key="secret", model=model)
    assert instance.model_profile.verified is False
    assert instance.model_profile.operations == {}
    assert all(
        not instance.capabilities.supports(operation) for operation in VideoOperation
    )


def test_vidu_reference_operation_uses_normal_images_not_subjects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = get_verified_video_model_profile("vidu", "viduq3-turbo")
    assert profile is not None
    operation = profile.operation_profile(VideoOperation.REFERENCE_TO_VIDEO)
    assert operation is not None
    assert operation.payload_strategy is VideoPayloadStrategy.VIDU_REFERENCE_IMAGES
    assert operation.payload_strategy is not VideoPayloadStrategy.VIDU_REFERENCE_SUBJECTS

    image = tmp_path / "character.png"
    image.write_bytes(b"png")
    request = VideoGenerationRequest(
        VideoOperation.REFERENCE_TO_VIDEO,
        "The established miner walks through the tunnel.",
        5,
        "720p",
        "16:9",
        references=(
            VideoReference(
                "character",
                VideoReferenceRole.CHARACTER_REFERENCE,
                str(image),
                hashlib.sha256(image.read_bytes()).hexdigest(),
            ),
        ),
    )

    def handler(http_request: httpx.Request) -> httpx.Response:
        payload = json.loads(http_request.content)
        assert str(http_request.url).endswith("/reference2video")
        assert payload["images"] == [
            "data:image/png;base64," + base64.b64encode(b"png").decode()
        ]
        assert "subjects" not in payload
        return httpx.Response(200, json={"task_id": "vidu-reference-1"})

    install_mock_transport(monkeypatch, handler)
    provider = ViduVideoProvider(api_key="secret", endpoint="https://vidu.test/ent/v2")
    submission = asyncio.run(provider.submit(request))
    assert submission.remote_task_id == "vidu-reference-1"


def test_seedance_profiles_expose_only_documented_operation_sets() -> None:
    lite = SeedanceVideoProvider(
        api_key="secret", model="seedance-1-0-lite-i2v-250428"
    )
    pro = SeedanceVideoProvider(api_key="secret", model="seedance-1-0-pro-250528")
    fast = SeedanceVideoProvider(
        api_key="secret", model="seedance-1-0-pro-fast-251015"
    )

    assert set(lite.model_profile.operations) == {
        VideoOperation.IMAGE_TO_VIDEO,
        VideoOperation.FIRST_LAST_TO_VIDEO,
        VideoOperation.REFERENCE_TO_VIDEO,
    }
    assert set(pro.model_profile.operations) == {
        VideoOperation.TEXT_TO_VIDEO,
        VideoOperation.IMAGE_TO_VIDEO,
        VideoOperation.FIRST_LAST_TO_VIDEO,
    }
    assert set(fast.model_profile.operations) == {
        VideoOperation.TEXT_TO_VIDEO,
        VideoOperation.IMAGE_TO_VIDEO,
    }
    lite_i2v = lite.model_profile.operation_profile(VideoOperation.IMAGE_TO_VIDEO)
    lite_first_last = lite.model_profile.operation_profile(
        VideoOperation.FIRST_LAST_TO_VIDEO
    )
    lite_reference = lite.model_profile.operation_profile(
        VideoOperation.REFERENCE_TO_VIDEO
    )
    assert lite_i2v is not None
    assert lite_first_last is not None
    assert lite_reference is not None
    assert lite_i2v.supported_durations == (5, 10)
    assert lite_i2v.supported_resolutions == ("480p", "720p", "1080p")
    assert lite_i2v.supported_aspect_ratios == ("16:9", "9:16", "1:1", "adaptive")
    assert lite_i2v.accepted_reference_roles == {VideoReferenceRole.FIRST_FRAME}
    assert lite_first_last.accepted_reference_roles == {
        VideoReferenceRole.FIRST_FRAME,
        VideoReferenceRole.LAST_FRAME,
    }
    assert lite_reference.max_reference_images == 4


def test_every_verified_operation_explicitly_classifies_every_reference_role() -> None:
    valid = {
        "SUPPORTED_AND_MAPPED",
        "IGNORED_BY_DESIGN",
        "UNSUPPORTED_ERROR",
    }
    for profile in MODEL_PROFILES.values():
        for operation in profile.operations.values():
            mapping = {
                role: operation.reference_role_behavior(role)
                for role in VideoReferenceRole
            }
            assert set(mapping) == set(VideoReferenceRole)
            assert set(mapping.values()) <= valid
            assert {
                role
                for role, behavior in mapping.items()
                if behavior == "SUPPORTED_AND_MAPPED"
            } == set(operation.accepted_reference_roles)


def test_adapter_capabilities_and_endpoint_come_from_same_model_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = WanVideoProvider(
        api_key="secret",
        model="wan2.7-i2v-2026-04-25",
        endpoint="https://wan.test/api/v1",
    )
    assert provider.capabilities == provider.model_profile.capabilities
    operation = provider.model_profile.operation_profile(VideoOperation.IMAGE_TO_VIDEO)
    assert operation is not None

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith("/" + operation.create_path)
        payload = json.loads(request.content)
        assert "ratio" not in payload["parameters"]
        return httpx.Response(200, json={"output": {"task_id": "wan-profile-1"}})

    install_mock_transport(monkeypatch, handler)
    submission = asyncio.run(provider.submit(_request(tmp_path)))
    assert submission.remote_task_id == "wan-profile-1"


def test_vidu_i2v_full_payload_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ViduVideoProvider(api_key="redacted", endpoint="https://vidu.test/ent/v2")
    request = VideoGenerationRequest(
        VideoOperation.IMAGE_TO_VIDEO,
        "Slow controlled head turn.",
        5,
        "720p",
        "16:9",
        references=(
            _remote_reference(
                "first", VideoReferenceRole.FIRST_FRAME, "https://media.test/first.png"
            ),
        ),
    )
    url, payload = _capture_submission(monkeypatch, provider, request)
    assert url == "https://vidu.test/ent/v2/img2video"
    assert payload == {
        "model": "viduq3-turbo",
        "prompt": "Slow controlled head turn.",
        "duration": 5,
        "resolution": "720p",
        "aspect_ratio": "16:9",
        "audio": False,
        "bgm": False,
        "images": ["https://media.test/first.png"],
    }


def test_vidu_reference_full_payload_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ViduVideoProvider(api_key="redacted", endpoint="https://vidu.test/ent/v2")
    request = VideoGenerationRequest(
        VideoOperation.REFERENCE_TO_VIDEO,
        "The same miner walks through the established tunnel.",
        5,
        "720p",
        "16:9",
        references=(
            _remote_reference(
                "character",
                VideoReferenceRole.CHARACTER_REFERENCE,
                "https://media.test/miner.png",
            ),
            _remote_reference(
                "location",
                VideoReferenceRole.MASTER_LOCATION,
                "https://media.test/tunnel.png",
            ),
        ),
    )
    url, payload = _capture_submission(monkeypatch, provider, request)
    assert url == "https://vidu.test/ent/v2/reference2video"
    assert payload == {
        "model": "viduq3-turbo",
        "prompt": "The same miner walks through the established tunnel.",
        "duration": 5,
        "resolution": "720p",
        "aspect_ratio": "16:9",
        "audio": False,
        "bgm": False,
        "images": [
            "https://media.test/miner.png",
            "https://media.test/tunnel.png",
        ],
    }
    assert "subjects" not in payload


def test_vidu_first_last_full_payload_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ViduVideoProvider(api_key="redacted", endpoint="https://vidu.test/ent/v2")
    request = VideoGenerationRequest(
        VideoOperation.FIRST_LAST_TO_VIDEO,
        "Move from stable tunnel to falling debris.",
        5,
        "1080p",
        "16:9",
        references=(
            _remote_reference("first", VideoReferenceRole.FIRST_FRAME, "https://media.test/a.png"),
            _remote_reference("last", VideoReferenceRole.LAST_FRAME, "https://media.test/b.png"),
        ),
    )
    url, payload = _capture_submission(monkeypatch, provider, request)
    assert url == "https://vidu.test/ent/v2/start-end2video"
    assert payload == {
        "model": "viduq3-turbo",
        "prompt": "Move from stable tunnel to falling debris.",
        "duration": 5,
        "resolution": "1080p",
        "aspect_ratio": "16:9",
        "audio": False,
        "bgm": False,
        "images": ["https://media.test/a.png", "https://media.test/b.png"],
    }


def test_vidu_cancel_full_payload_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={})

    install_mock_transport(monkeypatch, handler)
    provider = ViduVideoProvider(api_key="redacted", endpoint="https://vidu.test/ent/v2")
    asyncio.run(provider.cancel_task("vidu-task-7"))
    assert captured == {
        "method": "POST",
        "url": "https://vidu.test/ent/v2/tasks/vidu-task-7/cancel",
        "payload": {"id": "vidu-task-7"},
    }


def test_wan_i2v_full_payload_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = WanVideoProvider(api_key="redacted", endpoint="https://wan.test/api/v1")
    request = VideoGenerationRequest(
        VideoOperation.IMAGE_TO_VIDEO,
        "The miner raises a lamp.",
        2,
        "720p",
        "16:9",
        references=(
            _remote_reference("first", VideoReferenceRole.FIRST_FRAME, "https://media.test/first.png"),
        ),
    )
    url, payload = _capture_submission(monkeypatch, provider, request)
    assert url.endswith("/services/aigc/video-generation/video-synthesis")
    assert payload == {
        "model": "wan2.7-i2v-2026-04-25",
        "input": {
            "prompt": "The miner raises a lamp.",
            "media": [{"type": "first_frame", "url": "https://media.test/first.png"}],
        },
        "parameters": {
            "resolution": "720P",
            "duration": 2,
            "prompt_extend": True,
            "watermark": False,
        },
    }


def test_wan_edit_full_payload_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = WanVideoProvider(
        api_key="redacted",
        model="wan2.7-videoedit",
        endpoint="https://wan.test/api/v1",
    )
    request = VideoGenerationRequest(
        VideoOperation.VIDEO_EDIT,
        "Replace the miner's coat with the referenced coat.",
        5,
        "1080p",
        "16:9",
        references=(
            _remote_reference(
                "source", VideoReferenceRole.SOURCE_VIDEO, "https://media.test/source.mp4"
            ),
            _remote_reference(
                "coat", VideoReferenceRole.CHARACTER_REFERENCE, "https://media.test/coat.png"
            ),
        ),
    )
    url, payload = _capture_submission(monkeypatch, provider, request)
    assert url.endswith("/services/aigc/video-generation/video-synthesis")
    assert payload == {
        "model": "wan2.7-videoedit",
        "input": {
            "prompt": "Replace the miner's coat with the referenced coat.",
            "media": [
                {"type": "video", "url": "https://media.test/source.mp4"},
                {"type": "reference_image", "url": "https://media.test/coat.png"},
            ],
        },
        "parameters": {
            "resolution": "1080P",
            "duration": 5,
            "prompt_extend": True,
            "watermark": False,
            "ratio": "16:9",
        },
    }


def test_wan_continuation_full_payload_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = WanVideoProvider(api_key="redacted", endpoint="https://wan.test/api/v1")
    request = VideoGenerationRequest(
        VideoOperation.VIDEO_CONTINUATION,
        "Continue as the miner enters the next tunnel.",
        5,
        "720p",
        "16:9",
        references=(
            _remote_reference(
                "source", VideoReferenceRole.SOURCE_VIDEO, "https://media.test/source.mp4"
            ),
        ),
    )
    _, payload = _capture_submission(monkeypatch, provider, request)
    assert payload == {
        "model": "wan2.7-i2v-2026-04-25",
        "input": {
            "prompt": "Continue as the miner enters the next tunnel.",
            "media": [{"type": "first_clip", "url": "https://media.test/source.mp4"}],
        },
        "parameters": {
            "resolution": "720P",
            "duration": 5,
            "prompt_extend": True,
            "watermark": False,
        },
    }


def test_wan_source_video_rejects_local_base64_transport_before_http(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"output": {"task_id": "must-not-submit"}})

    install_mock_transport(monkeypatch, handler)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"not-uploaded")
    provider = WanVideoProvider(
        api_key="redacted",
        model="wan2.7-videoedit",
        endpoint="https://wan.test/api/v1",
    )
    request = VideoGenerationRequest(
        VideoOperation.VIDEO_EDIT,
        "Change the style.",
        5,
        "720p",
        "16:9",
        references=(
            VideoReference(
                "source",
                VideoReferenceRole.SOURCE_VIDEO,
                str(source),
                hashlib.sha256(source.read_bytes()).hexdigest(),
            ),
        ),
    )
    with pytest.raises(VideoGenerationError) as raised:
        asyncio.run(provider.submit(request))
    assert "public URL" in str(raised.value)
    assert calls == 0


def test_seedance_i2v_full_payload_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SeedanceVideoProvider(api_key="redacted", endpoint="https://seed.test/api/v3")
    request = VideoGenerationRequest(
        VideoOperation.IMAGE_TO_VIDEO,
        "Subtle dust movement.",
        5,
        "720p",
        "16:9",
        references=(
            _remote_reference("first", VideoReferenceRole.FIRST_FRAME, "https://media.test/first.png"),
        ),
    )
    url, payload = _capture_submission(
        monkeypatch, provider, request, task_response={"id": "seed-1"}
    )
    assert url == "https://seed.test/api/v3/contents/generations/tasks"
    assert payload == {
        "model": "seedance-1-0-lite-i2v-250428",
        "content": [
            {"type": "text", "text": "Subtle dust movement."},
            {
                "type": "image_url",
                "image_url": {"url": "https://media.test/first.png"},
                "role": "first_frame",
            },
        ],
        "duration": 5,
        "resolution": "720p",
        "ratio": "16:9",
        "watermark": False,
    }


def test_seedance_first_last_full_payload_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SeedanceVideoProvider(api_key="redacted", endpoint="https://seed.test/api/v3")
    request = VideoGenerationRequest(
        VideoOperation.FIRST_LAST_TO_VIDEO,
        "Transition between the two established states.",
        10,
        "1080p",
        "16:9",
        references=(
            _remote_reference("first", VideoReferenceRole.FIRST_FRAME, "https://media.test/a.png"),
            _remote_reference("last", VideoReferenceRole.LAST_FRAME, "https://media.test/b.png"),
        ),
    )
    _, payload = _capture_submission(
        monkeypatch, provider, request, task_response={"id": "seed-2"}
    )
    assert payload["content"] == [
        {"type": "text", "text": "Transition between the two established states."},
        {
            "type": "image_url",
            "image_url": {"url": "https://media.test/a.png"},
            "role": "first_frame",
        },
        {
            "type": "image_url",
            "image_url": {"url": "https://media.test/b.png"},
            "role": "last_frame",
        },
    ]
    assert payload == {
        "model": "seedance-1-0-lite-i2v-250428",
        "content": payload["content"],
        "duration": 10,
        "resolution": "1080p",
        "ratio": "16:9",
        "watermark": False,
    }


def test_seedance_reference_full_payload_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SeedanceVideoProvider(api_key="redacted", endpoint="https://seed.test/api/v3")
    request = VideoGenerationRequest(
        VideoOperation.REFERENCE_TO_VIDEO,
        "The referenced miner enters the referenced tunnel.",
        5,
        "720p",
        "16:9",
        references=(
            _remote_reference(
                "character", VideoReferenceRole.CHARACTER_REFERENCE, "https://media.test/miner.png"
            ),
            _remote_reference(
                "location", VideoReferenceRole.MASTER_LOCATION, "https://media.test/tunnel.png"
            ),
        ),
    )
    _, payload = _capture_submission(
        monkeypatch, provider, request, task_response={"id": "seed-3"}
    )
    assert payload == {
        "model": "seedance-1-0-lite-i2v-250428",
        "content": [
            {"type": "text", "text": "The referenced miner enters the referenced tunnel."},
            {
                "type": "image_url",
                "image_url": {"url": "https://media.test/miner.png"},
                "role": "reference_image",
            },
            {
                "type": "image_url",
                "image_url": {"url": "https://media.test/tunnel.png"},
                "role": "reference_image",
            },
        ],
        "duration": 5,
        "resolution": "720p",
        "ratio": "16:9",
        "watermark": False,
    }


def test_seedance_i2v_rejects_unverified_mixed_reference_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"id": "must-not-submit"})

    install_mock_transport(monkeypatch, handler)
    provider = SeedanceVideoProvider(api_key="redacted", endpoint="https://seed.test/api/v3")
    request = VideoGenerationRequest(
        VideoOperation.IMAGE_TO_VIDEO,
        "Animate without redesign.",
        5,
        "720p",
        "16:9",
        references=(
            _remote_reference("first", VideoReferenceRole.FIRST_FRAME, "https://media.test/first.png"),
            _remote_reference(
                "style", VideoReferenceRole.STYLE_REFERENCE, "https://media.test/style.png"
            ),
        ),
    )
    with pytest.raises(VideoGenerationError) as raised:
        asyncio.run(provider.submit(request))
    assert "STYLE_REFERENCE" in str(raised.value)
    assert calls == 0
