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


FORCE_SUFFIX = "\n\nAnswer:"


def _force_answers(llm, cfg, conversations, sps, outputs, log) -> dict[int, str]:
    """Second pass for responses that hit max_new_tokens: continue the partial assistant message
    with ``FORCE_SUFFIX`` for a few tokens so a final answer can be parsed. Returns {index: continuation}."""
    from vllm import SamplingParams

    idx = [k for k, out in enumerate(outputs) if out.outputs[0].finish_reason == "length"]
    if not idx:
        return {}
    max_tok = int(cfg["generation"].get("force_answer_max_tokens", 64))
    conv2, sp2 = [], []
    for k in idx:
        partial = outputs[k].outputs[0].text + FORCE_SUFFIX
        conv2.append(conversations[k] + [{"role": "assistant", "content": partial}])
        base = sps[k]
        sp2.append(SamplingParams(temperature=base.temperature, top_p=base.top_p, top_k=base.top_k,
                                  repetition_penalty=base.repetition_penalty, presence_penalty=base.presence_penalty,
                                  max_tokens=max_tok, seed=base.seed, stop=["\n\n"]))
    outs2 = llm.chat(conv2, sampling_params=sp2, use_tqdm=False, add_generation_prompt=False, continue_final_message=True)
    log(f"  forced answers for {len(idx)} truncated responses")
    return {k: o.outputs[0].text for k, o in zip(idx, outs2)}


def _global_index(subject: str, row_idx: int) -> int:
    """Stable per-question index (subject slot * 1000 + row) -> per-request seed is resume-order independent."""
    return SUBJECTS.index(subject) * 1000 + row_idx


def run(cfg: dict, out_dir: str, subjects: list[str], limit: int | None, resume: bool, log, ids: set[str] | None = None) -> dict:
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
        if ids is not None and not any(f"validation_{subj}_" in x for x in ids):
            continue
        for i in range(n):
            sample = ds[i]
            if sample["id"] in done or (ids is not None and sample["id"] not in ids):
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
        forced = _force_answers(llm, cfg, conversations, sps, outputs, log) if g.get("force_answer_on_truncation") else {}
        elapsed = time.time() - ts
        path = os.path.join(pred_dir, f"{subj}.jsonl")
        with open(path, "a") as f:
            for k, ((gi, ex), out) in enumerate(zip(items, outputs)):
                comp = out.outputs[0]
                text = comp.text
                cont = forced.get(k)
                if cont is not None:
                    text = text + FORCE_SUFFIX + cont
                rng = random.Random(base_seed + gi)  # deterministic fallback per question
                res = grade(text, ex.question_type, ex.all_choices, ex.index2ans, ex.gold, rng, answer_line_first)
                row = {
                    "id": ex.id, "subject": subj, "question_type": ex.question_type,
                    "gold": ex.gold, "pred": res["pred"], "correct": res["correct"],
                    "parse_fallback": res["parse_fallback"], "response": text,
                    "finish_reason": comp.finish_reason, "n_prompt_tokens": len(out.prompt_token_ids or []),
                    "n_output_tokens": len(comp.token_ids), "n_visual_tokens": ex.n_visual_tokens,
                    "image_sizes": ex.image_sizes, "seed": base_seed + gi, "prompt_text": ex.prompt_text,
                    "options": ex.options, "prompt_style": style,
                    "forced_answer": cont is not None, "forced_continuation": cont,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        generated += len(items)
        n_tr = sum(1 for out in outputs if out.outputs[0].finish_reason == "length")
        log(f"[{subj}] {len(items)} q in {elapsed:.1f}s ({elapsed / len(items):.2f}s/q) truncated={n_tr} forced={len(forced)}")
    return {"generated": generated}
