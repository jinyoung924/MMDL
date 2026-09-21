"""Evaluation loop: per-subject batched generation, JSONL checkpoint, resume."""
import json
import os
import random
import time

from tqdm import tqdm

from .data import SUBJECTS, build_example, load_subject
from .model import build_engine, build_sampling_params
from .parser import grade
from .scorer import read_jsonl


def _global_index(subject: str, row_idx: int) -> int:
    """Stable per-question index (subject slot * 1000 + row) -> per-request seed is resume-order independent."""
    return SUBJECTS.index(subject) * 1000 + row_idx


def run(cfg: dict, out_dir: str, subjects: list[str], limit: int | None, resume: bool, log) -> dict:
    pred_dir = os.path.join(out_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    m, d, s, g, im, en = (cfg["model"], cfg["data"], cfg["sampling"], cfg["generation"], cfg["image"], cfg["engine"])
    style = cfg["prompt"]["style"]
    answer_line_first = style == "mmmu_pro_cot"
    base_seed = int(s["seed"])

    # 1) Load + build all examples first (cheap; 71 GB RAM is plenty), skipping finished ones.
    todo: dict[str, list[tuple[int, object]]] = {}
    for subj in subjects:
        path = os.path.join(pred_dir, f"{subj}.jsonl")
        done = {r["id"] for r in read_jsonl(path)} if resume else set()
        if not resume and os.path.exists(path):
            os.remove(path)
        ds = load_subject(d["root"], subj, d["split"], d.get("revision"))
        n = min(limit, len(ds)) if limit else len(ds)
        items = []
        for i in range(n):
            sample = ds[i]
            if sample["id"] in done:
                continue
            ex = build_example(sample, subj, style, int(im["min_pixels"]), int(im["max_pixels"]),
                               int(im["token_budget_per_question"]))
            items.append((_global_index(subj, i), ex))
        log(f"[{subj}] total={n} done={len(done)} todo={len(items)}")
        if items:
            todo[subj] = items
    if not todo:
        log("nothing to do (all subjects complete)")
        return {"generated": 0}

    # 2) Engine
    t0 = time.time()
    llm = build_engine(m["path"], m.get("revision"), m["dtype"], int(en["max_model_len"]),
                       float(en["gpu_memory_utilization"]), int(en["limit_images_per_prompt"]), base_seed)
    log(f"engine ready in {time.time() - t0:.1f}s")

    # 3) Generate per subject, checkpoint after each subject
    generated = 0
    for subj, items in tqdm(todo.items(), desc="subjects"):
        conversations = [[{"role": "user", "content": ex.content}] for _, ex in items]
        sps = [build_sampling_params(s, int(g["max_new_tokens"]), base_seed + gi) for gi, _ in items]
        ts = time.time()
        outputs = llm.chat(conversations, sampling_params=sps, use_tqdm=False)
        elapsed = time.time() - ts
        path = os.path.join(pred_dir, f"{subj}.jsonl")
        with open(path, "a") as f:
            for (gi, ex), out in zip(items, outputs):
                comp = out.outputs[0]
                rng = random.Random(base_seed + gi)  # deterministic fallback per question
                res = grade(comp.text, ex.question_type, ex.all_choices, ex.index2ans, ex.gold, rng, answer_line_first)
                row = {
                    "id": ex.id, "subject": subj, "question_type": ex.question_type,
                    "gold": ex.gold, "pred": res["pred"], "correct": res["correct"],
                    "parse_fallback": res["parse_fallback"], "response": comp.text,
                    "finish_reason": comp.finish_reason, "n_prompt_tokens": len(out.prompt_token_ids or []),
                    "n_output_tokens": len(comp.token_ids), "n_visual_tokens": ex.n_visual_tokens,
                    "image_sizes": ex.image_sizes, "seed": base_seed + gi, "prompt_text": ex.prompt_text,
                    "options": ex.options, "prompt_style": style,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        generated += len(items)
        acc = sum(1 for (gi, ex), out in zip(items, outputs)
                  if grade(out.outputs[0].text, ex.question_type, ex.all_choices, ex.index2ans, ex.gold,
                           random.Random(base_seed + gi), answer_line_first)["correct"]) / len(items) * 100
        log(f"[{subj}] {len(items)} q in {elapsed:.1f}s ({elapsed / len(items):.2f}s/q) batch-acc={acc:.1f}")
    return {"generated": generated}
