#!/usr/bin/env python
"""Re-grade stored responses with the CURRENT parser (no GPU). Keeps predictions/*.jsonl
consistent with src/mmmu_eval/parser.py after a parser change, then rewrites the score files."""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mmmu_eval.parser import grade  # noqa: E402
from mmmu_eval.scorer import score, to_markdown, write_outputs  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    pred_dir = os.path.join(a.out_dir, "predictions")
    changed = 0
    for name in sorted(os.listdir(pred_dir)):
        path = os.path.join(pred_dir, name)
        rows = [json.loads(line) for line in open(path) if line.strip()]
        for r in rows:
            all_choices = [chr(ord("A") + i) for i in range(len(r["options"]))]
            res = grade(r["response"], r["question_type"], all_choices, dict(zip(all_choices, r["options"])),
                        r["gold"], random.Random(r["seed"]), r.get("prompt_style") == "mmmu_pro_cot")
            if (res["pred"], res["correct"], res["parse_fallback"]) != (r["pred"], r["correct"], r["parse_fallback"]):
                # open-question preds are unordered sets -> compare on correctness/fallback only
                if r["question_type"] != "multiple-choice" and res["correct"] == r["correct"]:
                    continue
                changed += 1
                print(f"changed {r['id']}: pred {r['pred']}->{res['pred']} correct {r['correct']}->{res['correct']} "
                      f"fallback {r['parse_fallback']}->{res['parse_fallback']}")
            r.update(res)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    res = score(pred_dir)
    write_outputs(res, a.out_dir)
    print(f"rows changed: {changed}")
    print(to_markdown(res).splitlines()[-3])
