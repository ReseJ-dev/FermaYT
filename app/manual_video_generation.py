"""Explicit one-provider smoke command for paid AI-video APIs.

Nothing is dispatched until the operator supplies --confirm-paid-request.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import time
from pathlib import Path

from app.clients.video_api import (
    SeedanceVideoProvider,
    ViduVideoProvider,
    WanVideoProvider,
)
from app.media.probe import probe_media
from app.video_providers import (
    RemoteVideoTaskState,
    VideoGenerationRequest,
    VideoOperation,
    VideoReference,
    VideoReferenceRole,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one explicit paid video smoke test"
    )
    parser.add_argument(
        "--provider", required=True, choices=("vidu", "wan", "seedance")
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--resolution", default="720p")
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--confirm-paid-request", action="store_true")
    return parser.parse_args()


def _provider(name: str, model: str):
    return {
        "vidu": ViduVideoProvider,
        "wan": WanVideoProvider,
        "seedance": SeedanceVideoProvider,
    }[name](model=model)


async def _run(args: argparse.Namespace) -> None:
    image = args.image.resolve()
    if not image.is_file():
        raise SystemExit(f"Input image not found: {image}")
    if not args.confirm_paid_request:
        raise SystemExit("Refusing paid submission without --confirm-paid-request")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    provider = _provider(args.provider, args.model)
    request = VideoGenerationRequest(
        operation=VideoOperation.IMAGE_TO_VIDEO,
        prompt=args.prompt,
        duration_seconds=args.duration,
        resolution=args.resolution,
        aspect_ratio=args.aspect_ratio,
        references=(
            VideoReference(
                "manual-first-frame", VideoReferenceRole.FIRST_FRAME, str(image), digest
            ),
        ),
    )
    started = time.monotonic()
    submission = await provider.submit(request)
    print(f"provider={provider.provider_id} model={provider.model}")
    print(f"operation={request.operation.value} task_id={submission.remote_task_id}")
    while True:
        result = await provider.get_task_status(submission.remote_task_id)
        print(f"status={result.state.value} elapsed={time.monotonic() - started:.1f}s")
        if result.state is RemoteVideoTaskState.SUCCEEDED:
            break
        if result.state in {
            RemoteVideoTaskState.FAILED,
            RemoteVideoTaskState.CANCELLED,
        }:
            raise SystemExit(result.error_message or result.state.value)
        await asyncio.sleep(args.poll_interval)
    if not result.result_url:
        raise SystemExit("Provider succeeded without a result URL")
    from app.services.video_generation import _download_video

    output = await _download_video(result.result_url, args.output.resolve())
    metadata = probe_media(output)
    print(f"output={output}")
    print(
        f"duration={metadata.duration:.3f}s resolution={metadata.width}x{metadata.height}"
    )
    print(f"actual_cost={result.actual_cost} credits={result.consumed_credits}")


def main() -> None:
    asyncio.run(_run(_arguments()))


if __name__ == "__main__":
    main()
