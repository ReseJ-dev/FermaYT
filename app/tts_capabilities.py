"""Validated capabilities of the selected text-to-speech integration."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TTSProviderCapabilities:
    supports_word_timestamps: bool = False
    supports_character_timestamps: bool = False
    supports_sentence_timestamps: bool = False
    supports_streaming: bool = False
    supports_speech_marks: bool = False

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)

