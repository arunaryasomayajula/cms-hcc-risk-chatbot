#!/usr/bin/env bash
# Linux/macOS launcher. Mirrors start.bat. Also the Docker image's ENTRYPOINT.
#
#   ./start.sh                   # app only, on $PORT (default 8000)
#   ./start.sh --port 9000       # listen on another port
#   ./start.sh --airllm          # app + the local AirLLM sidecar on port 8002
#
# Environment:
#   PORT / HOST     listen address (validated by backend/app.py)
#   SKIP_INSTALL=1  don't pip install (set in the Docker image: deps are baked in)
#   PORT_FILE       if set, the resolved port is written here (the Docker
#                   healthcheck reads it, so `--port` works there too)
set -euo pipefail

cd "$(dirname "$0")"

AIRLLM=0
while [ $# -gt 0 ]; do
  case "$1" in
    --airllm) AIRLLM=1 ;;
    --port)
      [ $# -ge 2 ] || { echo "[ERROR] --port needs a value" >&2; exit 2; }
      PORT="$2"; shift ;;
    --port=*) PORT="${1#--port=}" ;;
    *) echo "[ERROR] Unknown option: $1  (use --port N and/or --airllm)" >&2; exit 2 ;;
  esac
  shift
done
# app.py validates the value and is the single source of the default.
export PORT="${PORT:-8000}"

echo "============================================================"
echo " CMS HCC Risk Scoring Chatbot"
echo "============================================================"
echo

# Harmless elsewhere; needed on conda installs that link two OpenMP runtimes.
export KMP_DUPLICATE_LIB_OK=TRUE

# An API key is only needed for the 'claude' provider. A fully local setup
# (AirLLM / Ollama / vLLM) runs without one, so warn rather than refuse.
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "[WARN] ANTHROPIC_API_KEY is not set - the 'claude' provider will be unavailable."
  echo "       export ANTHROPIC_API_KEY=sk-ant-..."
  echo "       Or pick a local provider (Ollama / AirLLM) in the sidebar dropdowns."
  echo
fi

if [ "${SKIP_INSTALL:-0}" = 1 ]; then
  echo "[1/2] Skipping dependency install (SKIP_INSTALL=1)."
else
  echo "[1/2] Installing Python dependencies..."
  pip install -r backend/requirements.txt --quiet
fi

if [ "$AIRLLM" = 1 ] && [ ! -f backend/airllm_server.py ]; then
  echo "[ERROR] --airllm: backend/airllm_server.py is not present (it is excluded from the Docker image)." >&2
  exit 2
fi

if [ "$AIRLLM" = 1 ]; then
  echo
  echo "[+] Installing AirLLM dependencies..."
  echo "    NOTE: install a torch wheel matching your GPU FIRST - see backend/requirements-local.txt"
  pip install -r backend/requirements-local.txt --quiet
  echo "[+] Starting AirLLM sidecar on port 8002 (logs: airllm_server.log)..."
  ( cd backend && python airllm_server.py > ../airllm_server.log 2>&1 & )
  echo "    First load downloads and re-splits the model."
  echo "    Poll http://localhost:8002/health for progress."
fi

echo
echo "[2/2] Starting server at http://localhost:${PORT}"
echo
echo "NOTE: On first run, the ICD-10 index will be built (~2-5 minutes)."
echo "      The status badge in the UI will turn green when ready."
echo
echo "Press Ctrl+C to stop."
echo

if [ -n "${PORT_FILE:-}" ]; then
  printf '%s\n' "$PORT" > "$PORT_FILE"
fi

cd backend
# exec: python replaces this shell, so it is PID 1 in a container and receives
# SIGTERM from `docker stop` directly (bash would not forward it).
exec python app.py
