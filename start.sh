#!/usr/bin/env bash
# Linux/macOS launcher. Mirrors start.bat.
#
#   ./start.sh              # app only
#   ./start.sh --airllm     # app + the local AirLLM sidecar on port 8002
set -euo pipefail

cd "$(dirname "$0")"

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

echo "[1/2] Installing Python dependencies..."
pip install -r backend/requirements.txt --quiet

if [ "${1:-}" = "--airllm" ]; then
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
echo "[2/2] Starting server at http://localhost:8000"
echo
echo "NOTE: On first run, the ICD-10 index will be built (~2-5 minutes)."
echo "      The status badge in the UI will turn green when ready."
echo
echo "Press Ctrl+C to stop."
echo

cd backend
python app.py
