#!/usr/bin/env bash

set -u

APP_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
VENV_DIR="$APP_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

show_error() {
  echo
  echo "FermaYT could not start: $1"
  echo
  read -r -p "Press Enter to close this window..." _
  exit 1
}

open_browser() {
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://127.0.0.1:8000" >/dev/null 2>&1 || true
  fi
}

cd "$APP_DIR" || show_error "cannot open the application directory"

if ! command -v python3 >/dev/null 2>&1; then
  show_error "Python 3 is not installed"
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Preparing FermaYT for the first launch..."
  python3 -m venv "$VENV_DIR" \
    || show_error "cannot create the Python virtual environment"
fi

if ! "$PYTHON_BIN" -c \
  "import fastapi, httpx, jinja2, keyring, pydantic, sqlalchemy, uvicorn" \
  >/dev/null 2>&1; then
  echo "Installing FermaYT dependencies..."
  "$PYTHON_BIN" -m pip install -r "$APP_DIR/requirements.txt" \
    || show_error "dependency installation failed"
fi

if "$PYTHON_BIN" -c \
  'from urllib.request import urlopen; response = urlopen("http://127.0.0.1:8000/health", timeout=1); assert response.status == 200' \
  >/dev/null 2>&1; then
  echo "FermaYT is already running. Opening it in the browser..."
  open_browser
  sleep 2
  exit 0
fi

if command -v xdg-open >/dev/null 2>&1; then
  (
    sleep 2
    open_browser
  ) &
fi

echo "Starting FermaYT at http://127.0.0.1:8000"
echo "Close this window or press Ctrl+C to stop the application."
"$PYTHON_BIN" "$APP_DIR/run.py"
exit_code=$?

if [ "$exit_code" -eq 0 ] || [ "$exit_code" -eq 130 ]; then
  exit 0
fi

show_error "the server stopped unexpectedly (exit code $exit_code)"
