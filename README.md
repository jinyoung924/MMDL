# MMDL — Qwen3-VL-4B-Instruct MMMU-val Baseline

Reproducible evaluation pipeline (assignment 01). Report: [reports/mmmu_baseline.md](reports/mmmu_baseline.md).

## Reproduce (one command)

```bash
bash scripts/run_mmmu_eval.sh            # 30 subjects x 30 questions -> results/mmmu_baseline/
```

Paths are env vars (defaults = the pinned assignment values):

```bash
MODEL_PATH=/path/to/finetuned_ckpt DATA_ROOT=/path/to/mmmu_snapshot OUT_DIR=results/finetuned \
  bash scripts/run_mmmu_eval.sh
```

RunPod one-shot (setup + smoke test + full run): see [runpod.sh](runpod.sh).

## Environment

| | |
|---|---|
| RunPod template | `runpod/pytorch:1.3.2-cu1290-torch2130-ubuntu2404` |
| CUDA / PyTorch / Python | 12.9 / 2.13.0 / 3.12 |
| vLLM | 0.29.0 (pins torch 2.13.0; pulls transformers>=5.10.4) |
| Exact lock | `results/requirements.lock.txt` (written by `scripts/setup_runpod.sh`) |

## Layout

```
configs/mmmu_baseline.yaml   every pipeline choice (model/data revision, prompt, sampling, image, engine)
src/mmmu_eval/               data.py (load/resize/interleave) · prompts.py · parser.py (MMMU official port)
                             model.py (vLLM) · runner.py (batched gen + resume) · scorer.py · meta.py
scripts/run_mmmu_eval.sh     one-command reproduction  ->  run_eval.py + score.py
scripts/setup_runpod.sh      deps + version check        runpod.sh: SSH entry point
results/<run>/               predictions/<subject>.jsonl, scores.{md,csv,json}, run_meta.json, run.log
reports/mmmu_baseline.md     submission (from docs/submit-template.md)
docs/                        assignment text, template, references
```

Outputs are checkpointed per subject; re-running the same `OUT_DIR` resumes (`--no_resume` to restart).
