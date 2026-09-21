#!/usr/bin/env bash
# ============================================================================
# Single SSH entry point.  On a fresh RunPod pod:
#
#   git clone https://github.com/jinyoung924/MMDL.git /workspace/MMDL && bash /workspace/MMDL/runpod.sh
#
# Re-running after a disconnect resumes where it stopped (per-subject checkpoints).
# Knobs (env vars):  SKIP_SMOKE=1  SKIP_FULL=1  RUN_COT=1  OUT_ROOT=/workspace/results
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
  echo "== full run: 30 subjects x 30 questions, mmmu_direct prompt =="
  OUT_DIR="$OUT_ROOT/mmmu_baseline" bash scripts/run_mmmu_eval.sh
  rm -rf results/mmmu_baseline && cp -r "$OUT_ROOT/mmmu_baseline" results/mmmu_baseline
fi

if [[ "${RUN_COT:-0}" == "1" ]]; then
  echo "== ablation: MMMU-Pro CoT prompt, same recipe/parser =="
  OUT_DIR="$OUT_ROOT/ablation_cot" bash scripts/run_mmmu_eval.sh --prompt_style mmmu_pro_cot
  rm -rf results/ablation_cot && cp -r "$OUT_ROOT/ablation_cot" results/ablation_cot
fi

echo "== done. results copied into $REPO_DIR/results — commit & push them: =="
echo "   cd $REPO_DIR && git add results && git commit -m 'results: mmmu baseline' && git push"
