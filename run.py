"""Run the local FermaYT web application."""

from pathlib import Path

import uvicorn


HOST = "127.0.0.1"
PORT = 8000
DATA_DIR = Path("data")


def prepare_data_directories() -> None:
    """Create local directories required before application startup."""
    (DATA_DIR / "projects").mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Start the application on the local loopback interface."""
    prepare_data_directories()
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
