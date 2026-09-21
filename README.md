# MMDL — Qwen3-VL-4B-Instruct MMMU-val Baseline

Reproducible evaluation pipeline (assignment 01). Report: [reports/mmmu_baseline.md](reports/mmmu_baseline.md).

| Run | Prompt | max_new_tokens | Overall (macro avg) | Dir |
|---|---|---|---|---|
| **Baseline** | MMMU-Pro CoT | 8192 | **64.56** | `results/mmmu_baseline/` |
| Ablation | MMMU direct-answer | 8192 | 60.11 | `results/ablation_direct_8k/` |
| Ablation | MMMU direct-answer | 1024 | 56.44 | `results/ablation_direct_1k/` |

Official (Qwen3-VL Technical Report): 67.4.

## Reproduce (one command)

```bash
bash scripts/run_mmmu_eval.sh            # 30 subjects x 30 questions -> results/mmmu_baseline/  (~64 min on one RTX 4090)
```

Paths are env vars (defaults = the pinned assignment values):

```bash
MODEL_PATH=/path/to/finetuned_ckpt DATA_ROOT=/path/to/mmmu_snapshot OUT_DIR=results/finetuned \
  bash scripts/run_mmmu_eval.sh
```

RunPod one-shot (setup + smoke test + full run): see [runpod.sh](runpod.sh). `RUN_ABLATIONS=1` also runs the two
direct-answer ablations. Re-grade stored responses without a GPU: `python scripts/regrade.py --out_dir results/<run>`.

## Environment

| | |
|---|---|
| RunPod template | `runpod/pytorch:1.3.2-cu1290-torch2130-ubuntu2404` |
| CUDA / PyTorch / Python | 12.9 / 2.13.0 / 3.12 |
| vLLM | 0.29.0 **+cu129 wheel** from the GitHub release (PyPI default is a CUDA 13.0 build); pins torch 2.13.0 |
| Python env | venv with `--system-site-packages` at `/workspace/venv` (Ubuntu 24.04 system python is PEP 668 managed) |
| Exact lock | `results/requirements.lock.txt` (written by `scripts/setup_runpod.sh`) |

## Layout

```
configs/mmmu_baseline.yaml   every pipeline choice (model/data revision, prompt, sampling, image, engine)
src/mmmu_eval/               data.py (load/resize/interleave) · prompts.py · parser.py (MMMU official port)
                             model.py (vLLM) · runner.py (batched gen + resume) · scorer.py · meta.py
scripts/run_mmmu_eval.sh     one-command reproduction  ->  run_eval.py + score.py
scripts/setup_runpod.sh      CUDA preflight + venv + deps      runpod.sh: SSH entry point
scripts/regrade.py           re-parse stored responses with the current parser (no GPU)
scripts/finish_and_terminate.sh   optional: push results from the pod, then self-terminate it
results/<run>/               predictions/<subject>.jsonl, scores.{md,csv,json}, run_meta.json, run.log
reports/mmmu_baseline.md     submission (from docs/submit-template.md)
docs/                        assignment text, template, references
```

Outputs are checkpointed per subject; re-running the same `OUT_DIR` resumes (`--no_resume` to restart).
