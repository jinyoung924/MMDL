#!/usr/bin/env python
"""Summarize the condition sub-directories of an experiment directory into summary.md."""
import glob
import json
import os
import sys

exp = sys.argv[1]
lines = ["| condition | n | acc | finished (acc) | truncated (acc) | forced | fallback | sec |", "|---|---|---|---|---|---|---|---|"]
for d in sorted(p for p in glob.glob(os.path.join(exp, "*")) if os.path.isdir(p)):
    rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(d, "predictions", "*.jsonl"))) for l in open(f) if l.strip()]
    if not rows:
        continue
    acc = lambda rs: f"{sum(r['correct'] for r in rs) / len(rs) * 100:.1f}" if rs else "-"
    fin = [r for r in rows if r["finish_reason"] != "length"]
    tr = [r for r in rows if r["finish_reason"] == "length"]
    meta = json.load(open(os.path.join(d, "run_meta.json"))) if os.path.exists(os.path.join(d, "run_meta.json")) else {}
    lines.append(f"| {os.path.basename(d)} | {len(rows)} | {acc(rows)} | {len(fin)} ({acc(fin)}) | {len(tr)} ({acc(tr)}) | "
                 f"{sum(1 for r in rows if r.get('forced_answer'))} | {sum(1 for r in rows if r.get('parse_fallback'))} | {meta.get('total_elapsed_sec', '-')} |")
out = "\n".join(lines)
open(os.path.join(exp, "summary.md"), "w").write(out + "\n")
print(out)
