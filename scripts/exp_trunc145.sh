#!/usr/bin/env bash
# Experiment: the 145 questions whose CoT response hit max_new_tokens=8192 (results/mmmu_baseline).
# Four conditions on exactly those ids, same recipe/seed/parser otherwise. ~1 h on one RTX 4090.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT_ROOT="${OUT_ROOT:-/workspace/results}"
IDS=configs/ids_truncated_cot8k.txt
EXP="$OUT_ROOT/exp_trunc145"
run() { local name=$1; shift; echo "== exp_trunc145/$name =="; OUT_DIR="$EXP/$name" bash scripts/run_mmmu_eval.sh --ids_file "$IDS" --no_resume "$@"; }
run 00_repro                                                   # same settings again: reproducibility of truncation
run 01_forced        --force_answer                            # 2nd pass: append "Answer:" to the partial response
run 02_forced_pp0    --force_answer --presence_penalty 0       # does presence_penalty=1.5 cause the drift?
run 03_budget32k     --max_new_tokens 32768 --max_model_len 40960   # official out_seq_length as a control
python scripts/summarize_exp.py "$EXP"
mkdir -p results && rm -rf results/exp_trunc145 && cp -r "$EXP" results/exp_trunc145
