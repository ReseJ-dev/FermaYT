"""Tests for the local FastAPI application shell."""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_index_returns_html() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "FermaYT" in response.text
    assert "/static/app.css" in response.text
    assert "/static/app.js" in response.text


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_static_file_is_served_locally() -> None:
    response = client.get("/static/app.css")

    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]
