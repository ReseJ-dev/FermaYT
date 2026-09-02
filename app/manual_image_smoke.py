"""Optional real-provider smoke checks; never imported by the test suite."""

import argparse
import asyncio
import hashlib
from pathlib import Path

from app.providers import (
    ImageEditingProvider,
    ImageReference,
    ImageReferenceRole,
    ReferenceImageProvider,
    get_image_provider,
)
from app.utils.download import download_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manually smoke-test a configured image provider",
    )
    parser.add_argument("provider", choices=("seedream", "qwen"))
    parser.add_argument("operation", choices=("text", "reference", "edit"))
    parser.add_argument("prompt")
    parser.add_argument("output_path")
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        help="Local reference image path; repeat for multiple references",
    )
    return parser


async def _run(args: argparse.Namespace) -> str:
    provider = get_image_provider(args.provider)
    references = tuple(
        _local_reference(Path(path), index)
        for index, path in enumerate(args.reference, start=1)
    )
    if args.operation == "text":
        image_url = await provider.generate(args.prompt)
    elif args.operation == "reference":
        if not isinstance(provider, ReferenceImageProvider):
            raise RuntimeError("Selected provider does not implement references")
        image_url = await provider.generate_with_references(args.prompt, references)
    else:
        if not isinstance(provider, ImageEditingProvider):
            raise RuntimeError("Selected provider does not implement editing")
        image_url = await provider.edit(args.prompt, references)
    return await download_file(image_url, args.output_path)


def _local_reference(path: Path, index: int) -> ImageReference:
    if not path.is_file():
        raise ValueError(f"Reference image is missing: {path}")
    return ImageReference(
        reference_id=f"manual_reference_{index}",
        file_path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        role=ImageReferenceRole.CONTENT_CONTINUITY,
    )


def main() -> None:
    args = _parser().parse_args()
    try:
        output_path = asyncio.run(_run(args))
    except Exception as exc:
        raise SystemExit(
            f"{args.provider} {args.operation} smoke test failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    print(f"Saved image: {output_path}")


if __name__ == "__main__":
    main()
