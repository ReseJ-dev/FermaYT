"""File download utilities."""

import asyncio
from pathlib import Path

import httpx


DOWNLOAD_TIMEOUT_SECONDS = 30.0


async def download_file(url: str, output_path: str) -> str:
    """Download a file to the requested path and return that path."""
    destination = Path(output_path)
    await asyncio.to_thread(
        destination.parent.mkdir,
        parents=True,
        exist_ok=True,
    )

    async with httpx.AsyncClient(
        timeout=DOWNLOAD_TIMEOUT_SECONDS
    ) as client:
        response = await client.get(url)
        response.raise_for_status()

    await asyncio.to_thread(destination.write_bytes, response.content)
    return output_path
