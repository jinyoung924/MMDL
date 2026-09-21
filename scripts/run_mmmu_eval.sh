#!/usr/bin/env bash
# One-command reproduction of the MMMU-val baseline.
#
#   bash scripts/run_mmmu_eval.sh                      # full 900-question run
#   bash scripts/run_mmmu_eval.sh --subjects Art --limit 3   # smoke test (extra args go to run_eval.py)
#
# Paths are env vars so a grader only swaps these:
#   MODEL_PATH     HF repo id or local checkpoint dir   (default Qwen/Qwen3-VL-4B-Instruct)
#   MODEL_REVISION commit sha (ignored for local dirs)   (default ebb281ec70b05090aa6165b016eac8ec08e71b17)
#   DATA_ROOT      MMMU hub id or local snapshot dir     (default MMMU/MMMU)
#   DATA_REVISION  commit sha                            (default 98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68)
#   OUT_DIR        where predictions/scores/meta land    (default results/mmmu_baseline)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-ebb281ec70b05090aa6165b016eac8ec08e71b17}"
DATA_ROOT="${DATA_ROOT:-MMMU/MMMU}"
DATA_REVISION="${DATA_REVISION:-98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68}"
OUT_DIR="${OUT_DIR:-results/mmmu_baseline}"
if [[ -z "${HF_HOME:-}" && -d /workspace ]]; then export HF_HOME=/workspace/hf; fi
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export TOKENIZERS_PARALLELISM=false

python scripts/run_eval.py \
  --model_path "$MODEL_PATH" --model_revision "$MODEL_REVISION" \
  --data_root "$DATA_ROOT" --data_revision "$DATA_REVISION" \
  --out_dir "$OUT_DIR" "$@"

python scripts/score.py --out_dir "$OUT_DIR"
