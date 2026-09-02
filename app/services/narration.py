"""Idempotent story-level narration generation and timing normalization."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from sqlalchemy.orm import Session

from app.generators.voice import validate_tts_text
from app.media.probe import get_media_duration
from app.models.timeline import (
    NarrationTimingGranularity,
    NarrationTimingItem,
    NarrationTimingTrack,
    TimestampedNarrationOutput,
    TimestampSource,
)
from app.persistence import Project, ProjectNarrationAsset
from app.providers import (
    TTSProvider,
    get_tts_provider,
    get_tts_provider_capabilities,
)
from app.repositories import (
    create_project_narration_asset,
    get_project,
    get_project_narration_asset_by_revision,
)
from app.storage import ProjectMediaPaths
from app.utils.download import download_file

ProviderResolver = Callable[[str, Mapping[str, Any] | None], TTSProvider]
Downloader = Callable[[str, str], Awaitable[str]]
DurationProbe = Callable[[str | Path], float]


@runtime_checkable
class TimestampedTTSProvider(Protocol):
    async def generate_with_timestamps(
        self,
        text: str,
    ) -> TimestampedNarrationOutput: ...


async def generate_project_narration(
    session: Session,
    project_id: str,
    *,
    provider_resolver: ProviderResolver = get_tts_provider,
    downloader: Downloader = download_file,
    duration_probe: DurationProbe = get_media_duration,
    projects_root: str | Path = "data/projects",
) -> ProjectNarrationAsset:
    """Generate or reuse one immutable canonical narration revision."""
    project = get_project(session, project_id)
    if project is None:
        raise ValueError(f"Project not found: {project_id}")
    story_text = validate_tts_text(project.story_text)
    provider_config = _provider_config(project)
    provider = provider_resolver(project.tts_provider, provider_config)
    capabilities = get_tts_provider_capabilities(provider)
    actual_model = getattr(provider, "model", project.tts_model)
    actual_voice = str(getattr(provider, "voice", project.tts_voice))
    actual_language = (
        getattr(provider, "language", project.tts_language)
        if project.tts_provider == "qwen"
        else None
    )
    provider_generation_settings = {
        "language": actual_language,
        "output_format": getattr(provider, "output_format", None),
    }
    revision = _stable_hash(
        {
            "project_id": project.id,
            "story_text_hash": _text_hash(story_text),
            "provider": project.tts_provider,
            "model": actual_model,
            "voice": actual_voice,
            "language": actual_language,
            "provider_generation_settings": provider_generation_settings,
            "capabilities": capabilities.snapshot(),
            "narration_generation_version": "project_narration_v1",
        }
    )
    existing = get_project_narration_asset_by_revision(session, revision)
    if existing is not None and await _verify_audio(existing):
        return existing

    timestamped_output: TimestampedNarrationOutput | None = None
    if _has_native_timing(capabilities) and isinstance(
        provider,
        TimestampedTTSProvider,
    ):
        timestamped_output = await provider.generate_with_timestamps(story_text)
        audio_result = timestamped_output.audio
    else:
        audio_result = await provider.generate(story_text)
    extension = ".mp3" if project.tts_provider == "elevenlabs" else ".wav"
    output_path = str(
        ProjectMediaPaths(project.id, projects_root).narration_audio_path(
            revision,
            extension,
        )
    )
    audio_path = await _persist_audio(audio_result, output_path, downloader)
    duration = await asyncio.to_thread(duration_probe, audio_path)
    if duration <= 0:
        raise ValueError("narration duration must be positive")
    if timestamped_output is not None:
        timing = normalize_provider_timing(
            story_text,
            duration,
            timestamped_output.timing,
        )
    else:
        timing = build_proportional_timing(story_text, duration)
    audio_sha256 = await asyncio.to_thread(_sha256_file, audio_path)
    return create_project_narration_asset(
        session,
        project_id=project.id,
        story_text_hash=_text_hash(story_text),
        provider=project.tts_provider,
        model=actual_model,
        voice=actual_voice,
        language=actual_language,
        capability_snapshot=capabilities.snapshot(),
        audio_path=audio_path,
        audio_sha256=audio_sha256,
        duration=duration,
        generation_revision=revision,
        timestamp_source=timing.source.value,
        timing_granularity=timing.granularity.value,
        timing_confidence=timing.confidence,
        timing_data=[item.model_dump(mode="json") for item in timing.items],
        timing_warnings=list(timing.warnings),
    )


def narration_timing_from_asset(
    asset: ProjectNarrationAsset,
) -> NarrationTimingTrack:
    return NarrationTimingTrack(
        source=asset.timestamp_source,
        granularity=asset.timing_granularity,
        confidence=asset.timing_confidence,
        items=asset.timing_data,
        warnings=asset.timing_warnings,
    )


def normalize_provider_timing(
    text: str,
    duration: float,
    timing: NarrationTimingTrack,
) -> NarrationTimingTrack:
    """Validate provider timing and attach monotonic source character ranges."""
    if timing.items[-1].end > duration + 0.05:
        raise ValueError("provider timestamps exceed narration duration")
    cursor = 0
    normalized_items: list[NarrationTimingItem] = []
    warnings = list(timing.warnings)
    for item in timing.items:
        start_char = item.source_start_char
        end_char = item.source_end_char
        if start_char is None or end_char is None:
            matched = _find_normalized_text(text, item.text, cursor)
            if matched is None:
                start_char = cursor
                end_char = min(len(text), cursor + max(len(item.text), 1))
                warnings.append(
                    f"Could not map provider timing text exactly: {item.text!r}"
                )
            else:
                start_char, end_char = matched
        if start_char < cursor:
            raise ValueError("provider timestamp source ranges are not monotonic")
        cursor = end_char
        normalized_items.append(
            item.model_copy(
                update={
                    "source_start_char": start_char,
                    "source_end_char": end_char,
                }
            )
        )
    return timing.model_copy(
        update={"items": normalized_items, "warnings": warnings}
    )


def build_proportional_timing(
    text: str,
    duration: float,
) -> NarrationTimingTrack:
    """Deterministically distribute real audio duration across narration tokens."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    matches = list(re.finditer(r"\S+", text))
    if not matches:
        raise ValueError("narration text must contain tokens")
    weights = [max(len(re.sub(r"\W", "", item.group())), 1) for item in matches]
    total_weight = sum(weights)
    elapsed = 0.0
    items: list[NarrationTimingItem] = []
    for index, (match, weight) in enumerate(zip(matches, weights, strict=True)):
        start = elapsed
        elapsed = duration if index == len(matches) - 1 else elapsed + duration * weight / total_weight
        items.append(
            NarrationTimingItem(
                text=match.group(),
                start=start,
                end=elapsed,
                source_start_char=match.start(),
                source_end_char=match.end(),
            )
        )
    return NarrationTimingTrack(
        source=TimestampSource.PROPORTIONAL_FALLBACK,
        granularity=NarrationTimingGranularity.WORD,
        confidence=0.55,
        items=items,
        warnings=[
            "Provider does not expose timestamps; deterministic proportional timing used"
        ],
    )


def _provider_config(project: Project) -> dict[str, str]:
    if project.tts_provider == "qwen":
        return {
            "model": project.tts_model or "qwen3-tts-flash",
            "voice": project.tts_voice,
            "language": project.tts_language,
        }
    if project.tts_provider == "elevenlabs":
        return {
            "model": project.tts_model or "eleven_multilingual_v2",
            "voice": project.tts_voice,
        }
    return {}


def _has_native_timing(capabilities: Any) -> bool:
    return any(
        (
            capabilities.supports_word_timestamps,
            capabilities.supports_character_timestamps,
            capabilities.supports_sentence_timestamps,
            capabilities.supports_speech_marks,
        )
    )


async def _persist_audio(
    audio: str | bytes,
    output_path: str,
    downloader: Downloader,
) -> str:
    if isinstance(audio, bytes):
        if not audio:
            raise ValueError("TTS provider returned empty audio")
        destination = Path(output_path)
        await asyncio.to_thread(destination.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(destination.write_bytes, audio)
        return output_path
    if not audio.strip():
        raise ValueError("TTS provider returned an empty audio URL")
    return await downloader(audio, output_path)


async def _verify_audio(asset: ProjectNarrationAsset) -> bool:
    path = Path(asset.audio_path)
    if not path.is_file():
        return False
    return await asyncio.to_thread(_sha256_file, path) == asset.audio_sha256


def _find_normalized_text(
    source: str,
    fragment: str,
    cursor: int,
) -> tuple[int, int] | None:
    direct = source.find(fragment, cursor)
    if direct >= 0:
        return direct, direct + len(fragment)
    wanted = [item.group().casefold() for item in re.finditer(r"\w+", fragment)]
    source_tokens = list(re.finditer(r"\w+", source[cursor:]))
    values = [item.group().casefold() for item in source_tokens]
    for index in range(len(values) - len(wanted) + 1):
        if values[index : index + len(wanted)] == wanted:
            first = source_tokens[index]
            last = source_tokens[index + len(wanted) - 1]
            return cursor + first.start(), cursor + last.end()
    return None


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
