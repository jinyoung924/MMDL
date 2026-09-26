#!/usr/bin/env bash
# Second half of exp_trunc145: forced answer with presence_penalty=1.0 (0 and 1.5 are covered by part 1),
# then summary + copy into results/. The 32768-token budget control was dropped (~5 h on 24 GB; the
# 1024->8192 ablation already showed diminishing returns).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT_ROOT="${OUT_ROOT:-/workspace/results}"
IDS=configs/ids_truncated_cot8k.txt
EXP="$OUT_ROOT/exp_trunc145"
echo "== exp_trunc145/03_forced_pp1 =="
OUT_DIR="$EXP/03_forced_pp1" bash scripts/run_mmmu_eval.sh --ids_file "$IDS" --no_resume --batch_all \
  --force_answer --presence_penalty 1.0
python scripts/summarize_exp.py "$EXP"
mkdir -p results && rm -rf results/exp_trunc145 && cp -r "$EXP" results/exp_trunc145
