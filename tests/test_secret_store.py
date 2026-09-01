"""Tests for persistent OS-backed API key storage."""

import keyring
import pytest
from keyring.errors import NoKeyringError, PasswordDeleteError

from app.secret_store import (
    BYTEPLUS_API_KEY,
    DASHSCOPE_API_KEY,
    ELEVENLABS_API_KEY,
    SERVICE_NAME,
    SecretStore,
    SecretStoreError,
)


def test_secret_store_uses_keyring_without_exposing_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(
        keyring,
        "set_password",
        lambda service, name, value: values.__setitem__((service, name), value),
    )
    monkeypatch.setattr(
        keyring,
        "get_password",
        lambda service, name: values.get((service, name)),
    )

    SecretStore().set_secret(BYTEPLUS_API_KEY, "  private-key  ")
    restored = SecretStore().get_secret(BYTEPLUS_API_KEY)

    assert restored == "private-key"
    assert values[(SERVICE_NAME, BYTEPLUS_API_KEY)] == "private-key"
    assert SecretStore().get_secret(ELEVENLABS_API_KEY) is None
    assert "private-key" not in repr(SecretStore())


def test_secret_store_deletes_existing_or_missing_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deleted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        keyring,
        "delete_password",
        lambda service, name: deleted.append((service, name)),
    )

    SecretStore().delete_secret(DASHSCOPE_API_KEY)

    assert deleted == [(SERVICE_NAME, DASHSCOPE_API_KEY)]

    def missing_secret(service: str, name: str) -> None:
        del service, name
        raise PasswordDeleteError("missing")

    monkeypatch.setattr(keyring, "delete_password", missing_secret)
    SecretStore().delete_secret(DASHSCOPE_API_KEY)


def test_secret_store_reports_unavailable_backend_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(service: str, name: str, value: str) -> None:
        del service, name, value
        raise NoKeyringError("backend unavailable")

    monkeypatch.setattr(keyring, "set_password", unavailable)

    with pytest.raises(SecretStoreError) as error:
        SecretStore().set_secret(BYTEPLUS_API_KEY, "never-show-this")

    assert "never-show-this" not in str(error.value)


@pytest.mark.parametrize("name", ["", "unknown", "api_key"])
def test_secret_store_rejects_unknown_names(name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        SecretStore().get_secret(name)
