#!/usr/bin/env bash
# ============================================================================
# Single SSH entry point.  On a fresh RunPod pod:
#
#   git clone https://github.com/jinyoung924/MMDL.git /workspace/MMDL && bash /workspace/MMDL/runpod.sh
#
# Re-running after a disconnect resumes where it stopped (per-subject checkpoints).
# Knobs (env vars):  SKIP_SMOKE=1  SKIP_FULL=1  RUN_ABLATIONS=1  OUT_ROOT=/workspace/results
# ============================================================================
set -euo pipefail
REPO_DIR="${REPO_DIR:-/workspace/MMDL}"
OUT_ROOT="${OUT_ROOT:-/workspace/results}"
cd "$REPO_DIR"
git pull --ff-only || true

bash scripts/setup_runpod.sh

if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  echo "== smoke test (Art, 3 questions) =="
  OUT_DIR="$OUT_ROOT/smoke" bash scripts/run_mmmu_eval.sh --subjects Art --limit 3 --no_resume
fi

if [[ "${SKIP_FULL:-0}" != "1" ]]; then
  echo "== full run: 30 subjects x 30 questions (config defaults: mmmu_pro_cot prompt, 8192 tokens) =="
  OUT_DIR="$OUT_ROOT/mmmu_baseline" bash scripts/run_mmmu_eval.sh
  rm -rf results/mmmu_baseline && cp -r "$OUT_ROOT/mmmu_baseline" results/mmmu_baseline
fi

if [[ "${RUN_ABLATIONS:-0}" == "1" ]]; then
  echo "== ablations: official MMMU direct-answer prompt at 1024 and 8192 tokens (same recipe/seed/parser) =="
  OUT_DIR="$OUT_ROOT/ablation_direct_1k" bash scripts/run_mmmu_eval.sh --prompt_style mmmu_direct --max_new_tokens 1024
  OUT_DIR="$OUT_ROOT/ablation_direct_8k" bash scripts/run_mmmu_eval.sh --prompt_style mmmu_direct --max_new_tokens 8192
  for d in ablation_direct_1k ablation_direct_8k; do rm -rf "results/$d" && cp -r "$OUT_ROOT/$d" "results/$d"; done
fi

echo "== done. results copied into $REPO_DIR/results — commit & push them: =="
echo "   cd $REPO_DIR && git add results && git commit -m 'results: mmmu baseline' && git push"
