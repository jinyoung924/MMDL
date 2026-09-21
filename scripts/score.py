#!/usr/bin/env python
"""Re-score an existing <out_dir>/predictions directory (no GPU needed)."""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mmmu_eval.scorer import score, to_markdown, write_outputs  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()
    res = score(os.path.join(a.out_dir, "predictions"))
    write_outputs(res, a.out_dir)
    print(to_markdown(res))
