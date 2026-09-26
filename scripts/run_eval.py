#!/usr/bin/env python
"""Run MMMU-val evaluation.

Every pipeline choice comes from configs/mmmu_baseline.yaml; CLI flags override single
values (and the fully-resolved config is written to <out_dir>/config_resolved.yaml).

Example:
  python scripts/run_eval.py --model_path Qwen/Qwen3-VL-4B-Instruct \
      --model_revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
      --data_root MMMU/MMMU --out_dir results/mmmu_baseline
"""
import argparse
import copy
import os
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mmmu_eval.data import SUBJECTS  # noqa: E402
from mmmu_eval.meta import VramSampler, environment_info, write_json  # noqa: E402
from mmmu_eval.runner import run  # noqa: E402
from mmmu_eval.scorer import score, to_markdown, write_outputs  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "mmmu_baseline.yaml"))
    p.add_argument("--model_path", help="HF repo id or local checkpoint dir (overrides config model.path)")
    p.add_argument("--model_revision", help="commit sha for hub ids; ignored for local dirs")
    p.add_argument("--data_root", help="HF dataset id (MMMU/MMMU) or local snapshot dir")
    p.add_argument("--data_revision")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--subjects", default="all", help="comma-separated subject names or 'all'")
    p.add_argument("--limit", type=int, default=None, help="questions per subject (smoke test only)")
    p.add_argument("--prompt_style", choices=["mmmu_direct", "mmmu_pro_cot"])
    p.add_argument("--max_new_tokens", type=int)
    p.add_argument("--max_pixels", type=int)
    p.add_argument("--min_pixels", type=int)
    p.add_argument("--max_model_len", type=int)
    p.add_argument("--gpu_memory_utilization", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--presence_penalty", type=float)
    p.add_argument("--force_answer", action="store_true", help="2nd pass: force 'Answer:' on truncated responses")
    p.add_argument("--ids_file", help="only evaluate question ids listed in this file (one per line, # comments)")
    p.add_argument("--no_resume", action="store_true", help="discard existing predictions in out_dir")
    return p.parse_args()


def resolve_config(args) -> dict:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    ov = [
        ("model_path", ("model", "path")), ("model_revision", ("model", "revision")),
        ("data_root", ("data", "root")), ("data_revision", ("data", "revision")),
        ("prompt_style", ("prompt", "style")), ("max_new_tokens", ("generation", "max_new_tokens")),
        ("max_pixels", ("image", "max_pixels")), ("min_pixels", ("image", "min_pixels")),
        ("max_model_len", ("engine", "max_model_len")),
        ("gpu_memory_utilization", ("engine", "gpu_memory_utilization")), ("seed", ("sampling", "seed")),
        ("presence_penalty", ("sampling", "presence_penalty")),
    ]
    for arg, (sec, key) in ov:
        val = getattr(args, arg)
        if val is not None:
            cfg[sec][key] = val
    if args.force_answer:
        cfg["generation"]["force_answer_on_truncation"] = True
    if os.path.isdir(cfg["model"]["path"]):
        cfg["model"]["revision"] = None
    return cfg


def main():
    args = parse_args()
    cfg = resolve_config(args)
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "run.log")

    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    subjects = SUBJECTS if args.subjects == "all" else [s.strip() for s in args.subjects.split(",") if s.strip()]
    unknown = [s for s in subjects if s not in SUBJECTS]
    if unknown:
        sys.exit(f"unknown subjects: {unknown}")

    with open(os.path.join(args.out_dir, "config_resolved.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    env = environment_info(ROOT)
    log(f"env: {env}")
    log(f"config: {cfg}")

    sampler = VramSampler().start()
    t0 = time.time()
    ids = None
    if args.ids_file:
        ids = {l.strip() for l in open(args.ids_file) if l.strip() and not l.startswith("#")}
        log(f"restricting to {len(ids)} ids from {args.ids_file}")
    stats = run(cfg, args.out_dir, subjects, args.limit, resume=not args.no_resume, log=log, ids=ids)
    elapsed = time.time() - t0
    peak = sampler.stop()

    res = score(os.path.join(args.out_dir, "predictions"))
    write_outputs(res, args.out_dir)
    meta_path = os.path.join(args.out_dir, "run_meta.json")
    prev = {}
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            prev = json.load(f)
    runs = prev.get("runs", [])
    runs.append({"started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                 "elapsed_sec": round(elapsed, 1), "generated": stats["generated"],
                 "peak_vram_mib": peak, "subjects": subjects, "limit": args.limit, "ids_file": args.ids_file})
    write_json(meta_path, {"env": env, "config": cfg, "runs": runs,
                           "total_elapsed_sec": round(sum(r["elapsed_sec"] for r in runs), 1),
                           "peak_vram_mib": max(r["peak_vram_mib"] for r in runs),
                           "macro_avg_acc": res["macro_avg_acc"], "complete": res["complete"]})
    log(f"elapsed={elapsed:.1f}s peak_vram={peak} MiB")
    print(to_markdown(res))


if __name__ == "__main__":
    main()
