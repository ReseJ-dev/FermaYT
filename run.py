"""Run the local FermaYT web application."""

import os
from pathlib import Path

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


def main() -> None:
    """Start the application on the local loopback interface."""
    prepare_data_directories()
    write_pid_file()
    try:
        uvicorn.run(
            "app.main:app",
            host=HOST,
            port=PORT,
            reload=False,
        )
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
