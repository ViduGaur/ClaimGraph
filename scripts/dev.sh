#!/usr/bin/env bash
# Start / stop the ClaimGraph backend and frontend in the background.
# Processes are tracked with PID files in logs/, so only they are touched.
#   scripts/dev.sh start | stop | status
# Defaults: backend 127.0.0.1:8765, frontend :5199. Override with CLAIMGRAPH_BACKEND_HOST,
# CLAIMGRAPH_BACKEND_PORT and CLAIMGRAPH_FRONTEND_PORT (CORS_ORIGINS must then include the frontend origin).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/logs"; mkdir -p "$LOGS"
HOST="${CLAIMGRAPH_BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${CLAIMGRAPH_BACKEND_PORT:-8765}"
FRONTEND_PORT="${CLAIMGRAPH_FRONTEND_PORT:-5199}"

start() {
  if [ ! -x "$ROOT/backend/.venv/bin/uvicorn" ]; then echo "backend venv missing — see README 'Setup'"; exit 1; fi
  if [ ! -d "$ROOT/frontend/node_modules" ]; then echo "frontend deps missing — run: (cd frontend && npm install)"; exit 1; fi
  # each server gets its own process group (setsid) so `stop` ends it and its children, and nothing else
  cd "$ROOT/backend"
  setsid nohup .venv/bin/uvicorn app.main:app --host "$HOST" --port "$BACKEND_PORT" > "$LOGS/backend.log" 2>&1 &
  echo $! > "$LOGS/backend.pid"
  cd "$ROOT/frontend"
  CLAIMGRAPH_API="http://$HOST:$BACKEND_PORT" CLAIMGRAPH_FRONTEND_PORT="$FRONTEND_PORT" \
    setsid nohup npm run dev > "$LOGS/frontend.log" 2>&1 &
  echo $! > "$LOGS/frontend.pid"
  cd "$ROOT"
  sleep 3
  echo "backend  → http://$HOST:$BACKEND_PORT/api/docs   (log: logs/backend.log)"
  echo "frontend → http://localhost:$FRONTEND_PORT        (log: logs/frontend.log)"
}

stop() {
  for svc in backend frontend; do
    f="$LOGS/$svc.pid"
    if [ -f "$f" ]; then
      pid="$(cat "$f")"
      # the pid is its own process-group leader (setsid): signal that group only
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      rm -f "$f"; echo "stopped $svc ($pid)"
    fi
  done
}

status() {
  curl -s -o /dev/null -w "backend  %{http_code}\n" "http://$HOST:$BACKEND_PORT/api/v1/health" || echo "backend  down"
  curl -s -o /dev/null -w "frontend %{http_code}\n" "http://localhost:$FRONTEND_PORT/" || echo "frontend down"
}

case "${1:-}" in start) start ;; stop) stop ;; status) status ;; *) echo "usage: $0 start|stop|status"; exit 1 ;; esac
