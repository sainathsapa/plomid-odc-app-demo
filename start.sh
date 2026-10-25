#!/bin/sh
# PLOMID ODC — one-command startup.
#
#   ./start.sh [--auto-start] [--port 8080]
#
# Starts the host PLOMID server (only if nothing answers), migrates the
# schema, then serves the console. Everything else keeps working as before.
# Env knobs (all optional): ODC_DSN, PLOMID_BIN, PLOMID_DATA.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"

if [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="python3"
fi

: "${ODC_DSN:=postgresql://plomid:plomid@127.0.0.1:5432/plomid}"
export ODC_DSN

# host/port out of the DSN for the readiness probe.
_rest=${ODC_DSN#*@}
ODC_HOST=${_rest%%:*}
_rest=${ODC_DSN##*:}
ODC_PORT=${_rest%%/*}

is_up() {
  "$PY" -c "import socket,sys; s=socket.socket(); s.settimeout(2)
try: s.connect(('$ODC_HOST', $ODC_PORT)); print('up')
except Exception: sys.exit(1)" 2>/dev/null
}

STARTED_DB=""
if [ "$ODC_HOST" = "127.0.0.1" ] || [ "$ODC_HOST" = "localhost" ] || [ "$ODC_HOST" = "::1" ]; then
  if is_up >/dev/null 2>&1; then
    echo "plomid server already up at $ODC_HOST:$ODC_PORT" >&2
  else
    : "${PLOMID_BIN:=$(CDPATH= cd -- "$ROOT/../plomid/target/release" 2>/dev/null && pwd)/plomid-server}"
    : "${PLOMID_DATA:=/tmp/plomid-data}"
    if [ ! -x "$PLOMID_BIN" ]; then
      echo "error: plomid server binary not found: $PLOMID_BIN" >&2
      echo "build it in ../plomid or start your database first, then re-run." >&2
      exit 1
    fi
    mkdir -p "$PLOMID_DATA"
    echo "starting plomid server ($PLOMID_DATA) ..." >&2
    "$PLOMID_BIN" --data "$PLOMID_DATA" --host 127.0.0.1 --port "$ODC_PORT" \
      --username plomid --password plomid >>/tmp/plomid-odc-server.log 2>&1 &
    STARTED_DB=$!
  fi
fi

cleanup() {
  if [ -n "$STARTED_DB" ]; then
    kill "$STARTED_DB" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

# wait until the database answers (or fail loudly, never hang forever).
i=0
while ! is_up >/dev/null 2>&1; do
  i=$((i + 1))
  if [ "$i" -ge 30 ]; then
    echo "error: database at $ODC_HOST:$ODC_PORT never became ready." >&2
    echo "tail of /tmp/plomid-odc-server.log may say why." >&2
    exit 1
  fi
  sleep 1
done

"$PY" -m odc.cli bootstrap
exec "$PY" -m odc.cli serve "$@"
