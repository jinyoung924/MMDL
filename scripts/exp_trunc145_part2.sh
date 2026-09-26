#!/usr/bin/env bash
# Second half of exp_trunc145: the budget control at 16384 tokens (32768 needs ~5 h on 24 GB because
# only ~2 sequences fit in the KV cache at that length), then summary + copy into results/.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT_ROOT="${OUT_ROOT:-/workspace/results}"
IDS=configs/ids_truncated_cot8k.txt
EXP="$OUT_ROOT/exp_trunc145"
if [[ "${SKIP_BUDGET:-0}" != "1" ]]; then
  echo "== exp_trunc145/03_budget16k =="
  OUT_DIR="$EXP/03_budget16k" bash scripts/run_mmmu_eval.sh --ids_file "$IDS" --no_resume --batch_all \
    --max_new_tokens 16384 --max_model_len 24576
fi
python scripts/summarize_exp.py "$EXP"
mkdir -p results && rm -rf results/exp_trunc145 && cp -r "$EXP" results/exp_trunc145
