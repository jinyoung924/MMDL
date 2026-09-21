#!/usr/bin/env bash
# Environment setup for RunPod (template: runpod/pytorch:1.3.2-cu1290-torch2130-ubuntu2404).
# Idempotent: safe to re-run after a pod restart.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${HF_HOME:-}" && -d /workspace ]]; then export HF_HOME=/workspace/hf; fi   # /workspace survives pod stop/start
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"
grep -q 'HF_HOME=' ~/.bashrc 2>/dev/null || echo "export HF_HOME=$HF_HOME" >> ~/.bashrc

echo "== system =="; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python --version

echo "== python deps =="
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "== versions =="
python scripts/check_env.py
python -m pip freeze > results/requirements.lock.txt
echo "wrote results/requirements.lock.txt"
