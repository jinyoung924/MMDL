"""Aggregate predictions/*.jsonl into per-subject accuracy + macro average."""
import csv
import json
import os

from .data import SUBJECTS


def read_jsonl(path: str) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score(pred_dir: str) -> dict:
    per_subject = []
    for subj in SUBJECTS:
        rows = read_jsonl(os.path.join(pred_dir, f"{subj}.jsonl"))
        n = len(rows)
        correct = sum(1 for r in rows if r["correct"])
        per_subject.append({
            "subject": subj,
            "n": n,
            "correct": correct,
            "acc": (correct / n * 100.0) if n else None,
            "n_multiple_choice": sum(1 for r in rows if r["question_type"] == "multiple-choice"),
            "n_open": sum(1 for r in rows if r["question_type"] != "multiple-choice"),
            "n_parse_fallback": sum(1 for r in rows if r.get("parse_fallback")),
            "n_truncated": sum(1 for r in rows if r.get("finish_reason") == "length"),
            "n_forced_answer": sum(1 for r in rows if r.get("forced_answer")),
        })
    evaluated = [s for s in per_subject if s["n"]]
    total_n = sum(s["n"] for s in evaluated)
    total_c = sum(s["correct"] for s in evaluated)
    return {
        "per_subject": per_subject,
        "n_subjects_evaluated": len(evaluated),
        "n_questions": total_n,
        "n_correct": total_c,
        "macro_avg_acc": (sum(s["acc"] for s in evaluated) / len(evaluated)) if evaluated else None,
        "micro_avg_acc": (total_c / total_n * 100.0) if total_n else None,
        "n_parse_fallback": sum(s["n_parse_fallback"] for s in evaluated),
        "n_truncated": sum(s["n_truncated"] for s in evaluated),
        "n_forced_answer": sum(s["n_forced_answer"] for s in evaluated),
        "complete": len(evaluated) == len(SUBJECTS) and all(s["n"] == 30 for s in evaluated),
    }


def fmt(x):
    return "" if x is None else f"{x:.2f}"


def to_markdown(res: dict) -> str:
    lines = ["| No. | Subject | Data Num | Acc |", "|---|---|---|---|"]
    for i, s in enumerate(res["per_subject"], 1):
        lines.append(f"| {i} | {s['subject']} | {s['n']} | {fmt(s['acc'])} |")
    lines.append(f"| | **Overall (macro avg)** | **{res['n_questions']}** | **{fmt(res['macro_avg_acc'])}** |")
    lines.append("")
    lines.append(f"계산식: `Overall = mean(30개 과목 accuracy)` = {fmt(res['macro_avg_acc'])}  "
                 f"(micro: {res['n_correct']}/{res['n_questions']} = {fmt(res['micro_avg_acc'])})")
    lines.append(f"파싱 fallback(무작위 선택) 건수: {res['n_parse_fallback']}, "
                 f"max_new_tokens 도달(잘림) 건수: {res['n_truncated']}, 답 강제(2단계) 건수: {res['n_forced_answer']}, 완주 여부: {res['complete']}")
    return "\n".join(lines)


def write_outputs(res: dict, out_dir: str):
    with open(os.path.join(out_dir, "scores.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "scores.md"), "w") as f:
        f.write(to_markdown(res) + "\n")
    with open(os.path.join(out_dir, "scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res["per_subject"][0].keys()))
        w.writeheader()
        w.writerows(res["per_subject"])
