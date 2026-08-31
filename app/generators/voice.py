"""Voice generation orchestration."""


def validate_tts_text(text: str) -> str:
    """Return trimmed TTS text or reject a blank value."""
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("TTS text must not be empty")
    return normalized_text
