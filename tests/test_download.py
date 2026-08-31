"""Tests for generic file downloading."""

import asyncio
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from app.utils.download import download_file


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


def run_download(url: str, output_path: Path) -> str:
    return asyncio.run(download_file(url, str(output_path)))


def test_download_file_saves_response_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url == "https://example.com/image.png"
        return httpx.Response(200, content=b"image-content")

    install_mock_transport(monkeypatch, handler)
    output_path = tmp_path / "image.png"

    result = run_download("https://example.com/image.png", output_path)

    assert result == str(output_path)
    assert output_path.read_bytes() == b"image-content"


def test_download_file_creates_parent_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"image-content")

    install_mock_transport(monkeypatch, handler)
    output_path = tmp_path / "missing" / "nested" / "image.png"

    run_download("https://example.com/image.png", output_path)

    assert output_path.is_file()


def test_download_file_raises_for_http_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"not found")

    install_mock_transport(monkeypatch, handler)
    output_path = tmp_path / "image.png"

    with pytest.raises(httpx.HTTPStatusError):
        run_download("https://example.com/image.png", output_path)

    assert not output_path.exists()
