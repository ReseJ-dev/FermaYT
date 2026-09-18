"""Run the local FermaYT web application."""

import asyncio
import os
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import uvicorn

HOST = "127.0.0.1"
PORT = 8000
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
PID_FILE = DATA_DIR / "fermayt.pid"


def prepare_data_directories() -> None:
    """Create local directories required before application startup."""
    (DATA_DIR / "projects").mkdir(parents=True, exist_ok=True)


def write_pid_file() -> None:
    """Record the actual server process so the stop launcher can find it."""
    PID_FILE.write_text(f"{os.getpid()}\n", encoding="utf-8")


def remove_own_pid_file() -> None:
    """Remove the PID file only when it still belongs to this process."""
    try:
        recorded_pid = PID_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return
    if recorded_pid == str(os.getpid()):
        PID_FILE.unlink(missing_ok=True)


def run_server() -> None:
    """Run Uvicorn without joining abandoned provider threads on loop shutdown."""
    config = uvicorn.Config(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=False,
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    _run_server_coroutine(server.serve())


def _run_server_coroutine(server: Coroutine[Any, Any, None]) -> None:
    """Own the loop so Ctrl+C is not blocked by asyncio's executor join."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(server)
    except KeyboardInterrupt:
        # Uvicorn re-raises the captured SIGINT after completing its graceful
        # shutdown. Treat that expected Ctrl+C as a normal desktop-app exit.
        pass
    finally:
        # FastAPI/Uvicorn has already run its lifespan shutdown here. Closing a
        # manually owned loop tells its executor to stop accepting work without
        # waiting for an in-flight synchronous provider call. cli_main then uses
        # os._exit so CPython's thread-executor atexit hook cannot hang Ctrl+C.
        loop.close()
        asyncio.set_event_loop(None)


def main() -> None:
    """Start the application on the local loopback interface."""
    prepare_data_directories()
    write_pid_file()
    try:
        run_server()
    finally:
        remove_own_pid_file()


def cli_main() -> None:
    """Run the server and never wait for abandoned worker threads on shutdown."""
    main()
    # Cancelling an asyncio.to_thread task cannot stop its underlying provider
    # worker. Uvicorn has already completed application shutdown at this point,
    # so exit the desktop process instead of waiting indefinitely for that thread.
    os._exit(0)


if __name__ == "__main__":
    cli_main()
