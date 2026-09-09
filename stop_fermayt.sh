#!/usr/bin/env bash

set -u

APP_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON_BIN="$APP_DIR/.venv/bin/python"
PID_FILE="$APP_DIR/data/fermayt.pid"

is_fermayt_process() {
  process_id="$1"
  [ -r "/proc/$process_id/cmdline" ] || return 1
  process_cwd="$(readlink -f "/proc/$process_id/cwd" 2>/dev/null)"
  [ "$process_cwd" = "$APP_DIR" ] || return 1
  process_executable="$(readlink -f "/proc/$process_id/exe" 2>/dev/null)"
  expected_executable="$(readlink -f "$PYTHON_BIN" 2>/dev/null)"
  [ "$process_executable" = "$expected_executable" ] || return 1
  command_line="$(tr '\0' ' ' < "/proc/$process_id/cmdline")"
  case "$command_line" in
    *" $APP_DIR/run.py "*|*" run.py "*) return 0 ;;
    *) return 1 ;;
  esac
}

server_pid=""
if [ -f "$PID_FILE" ]; then
  server_pid="$(sed -n '1p' "$PID_FILE" 2>/dev/null)"
fi

if [ -z "$server_pid" ] || ! is_fermayt_process "$server_pid"; then
  server_pid=""
  for process_dir in /proc/[0-9]*; do
    candidate_pid="${process_dir##*/}"
    if is_fermayt_process "$candidate_pid"; then
      server_pid="$candidate_pid"
      break
    fi
  done
fi

if [ -z "$server_pid" ] || ! is_fermayt_process "$server_pid"; then
  rm -f -- "$PID_FILE"
  echo "FermaYT is not running."
  exit 0
fi

kill -TERM "$server_pid" || exit 1
for _ in $(seq 1 50); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    rm -f -- "$PID_FILE"
    echo "FermaYT stopped."
    exit 0
  fi
  sleep 0.1
done

echo "FermaYT did not stop within 5 seconds (PID $server_pid)." >&2
exit 1
