#!/usr/bin/env bash
# Start the HF HTTP bridge on RunPod so your *local* backend can call it via MODEL_BASE_URL.
#
# 1) RunPod dashboard → your pod → Ports / Connect / HTTP:
#    expose the same port as PORT (default 8000). Copy the https://…proxy.runpod.net URL.
# 2) Set model paths below (adjust if your dirs are named differently), then run this script.
#
# One-time install (venv active):
#   pip install "uvicorn[standard]==0.34.0" "fastapi==0.115.6" "pydantic==2.10.4"
#
# Local backend .env (on your Mac):
#   MODEL_BASE_URL=https://…proxy.runpod.net

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8000}"

export RUNPOD_PATH_QWEN="${RUNPOD_PATH_QWEN:-/workspace/models/qwen25-math-7b}"
export RUNPOD_PATH_DEEPSEEK="${RUNPOD_PATH_DEEPSEEK:-/workspace/models/deepseek-math-7b}"
export RUNPOD_PATH_INTERNLM2="${RUNPOD_PATH_INTERNLM2:-/workspace/models/internlm2-math-plus-7b}"
export RUNPOD_PATH_WIZARDMATH="${RUNPOD_PATH_WIZARDMATH:-/workspace/models/wizardmath-7b-v1.1}"

# Set to 1 for any model that requires trust_remote_code
export RUNPOD_TRUST_QWEN="${RUNPOD_TRUST_QWEN:-0}"
export RUNPOD_TRUST_DEEPSEEK="${RUNPOD_TRUST_DEEPSEEK:-0}"
export RUNPOD_TRUST_INTERNLM2="${RUNPOD_TRUST_INTERNLM2:-1}"
export RUNPOD_TRUST_WIZARDMATH="${RUNPOD_TRUST_WIZARDMATH:-0}"

cd "$SCRIPT_DIR"
exec uvicorn runpod_hf_server:app --host 0.0.0.0 --port "$PORT"
