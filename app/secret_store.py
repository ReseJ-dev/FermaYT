"""Persistent secret storage backed by the operating-system keyring."""

import keyring
from keyring.errors import KeyringError, PasswordDeleteError

SERVICE_NAME = "FermaYT"
BYTEPLUS_API_KEY = "byteplus_api_key"
DASHSCOPE_API_KEY = "dashscope_api_key"
ELEVENLABS_API_KEY = "elevenlabs_api_key"
SUPPORTED_SECRETS = frozenset(
    {BYTEPLUS_API_KEY, DASHSCOPE_API_KEY, ELEVENLABS_API_KEY}
)


class SecretStoreError(RuntimeError):
    """Raised when the operating-system secret store is unavailable."""


class SecretStore:
    """Store API keys without exposing them through application persistence."""

    def set_secret(self, name: str, value: str) -> None:
        secret_name = _validate_name(name)
        secret_value = value.strip()
        if not secret_value:
            raise ValueError("secret value must not be empty")
        try:
            keyring.set_password(SERVICE_NAME, secret_name, secret_value)
        except KeyringError as exc:
            raise SecretStoreError(
                "Системное хранилище паролей недоступно."
            ) from exc

    def get_secret(self, name: str) -> str | None:
        secret_name = _validate_name(name)
        try:
            value = keyring.get_password(SERVICE_NAME, secret_name)
        except KeyringError as exc:
            raise SecretStoreError(
                "Системное хранилище паролей недоступно."
            ) from exc
        return value if value and value.strip() else None

    def delete_secret(self, name: str) -> None:
        secret_name = _validate_name(name)
        try:
            keyring.delete_password(SERVICE_NAME, secret_name)
        except PasswordDeleteError:
            return
        except KeyringError as exc:
            raise SecretStoreError(
                "Системное хранилище паролей недоступно."
            ) from exc

    def has_secret(self, name: str) -> bool:
        return self.get_secret(name) is not None


def _validate_name(name: str) -> str:
    if name not in SUPPORTED_SECRETS:
        raise ValueError("unsupported secret name")
    return name
