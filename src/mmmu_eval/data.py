"""MMMU loading, image resizing and chat-message construction."""
import ast
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from .prompts import build_prompt

SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
    "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
    "Energy_and_Power", "Finance", "Geography", "History", "Literature", "Manage",
    "Marketing", "Materials", "Math", "Mechanical_Engineering", "Music", "Pharmacy",
    "Physics", "Psychology", "Public_Health", "Sociology",
]

IMAGE_TOKEN_RE = re.compile(r"<image (\d+)>")
IMAGE_FACTOR = 32  # Qwen3-VL: patch_size 16 * spatial_merge_size 2 -> one visual token per 32x32 px


def load_subject(data_root: str, subject: str, split: str, revision: str | None):
    """``data_root`` is either a HF hub id (revision applied) or a local snapshot directory."""
    from datasets import load_dataset  # lazy: scoring/regrading must work without `datasets` installed

    if os.path.isdir(data_root):
        return load_dataset(data_root, subject, split=split)
    return load_dataset(data_root, subject, split=split, revision=revision)


# --- image sizing -----------------------------------------------------------------------
def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Same rounding rule as Qwen's processor (qwen_vl_utils.smart_resize), so the
    processor's own resize becomes a no-op after we pre-resize here."""
    if max(height, width) / max(1, min(height, width)) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {width}x{height}")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def resize_image(img: Image.Image, min_pixels: int, max_pixels: int) -> Image.Image:
    img = img.convert("RGB")
    h, w = smart_resize(img.height, img.width, IMAGE_FACTOR, min_pixels, max_pixels)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.BICUBIC)
    return img


def visual_tokens(img: Image.Image) -> int:
    return (img.height // IMAGE_FACTOR) * (img.width // IMAGE_FACTOR)


# --- sample -> example ------------------------------------------------------------------
def parse_options(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(o) for o in raw]
    try:
        val = ast.literal_eval(raw) if raw else []
    except (ValueError, SyntaxError):
        val = []
    return [str(o) for o in val] if isinstance(val, list) else []


def parse_gold(answer: str, question_type: str):
    """Open answers may be stored as a python-list string (multiple accepted answers)."""
    if question_type != "multiple-choice" and isinstance(answer, str) and answer.strip().startswith("["):
        try:
            val = ast.literal_eval(answer)
            if isinstance(val, list):
                return [str(v) for v in val]
        except (ValueError, SyntaxError):
            pass
    return answer


@dataclass
class Example:
    id: str
    subject: str
    question_type: str
    question: str
    options: list[str]
    all_choices: list[str]
    index2ans: dict[str, str]
    gold: Any
    prompt_text: str
    content: list[dict] = field(default_factory=list)
    image_sizes: dict[int, list[int]] = field(default_factory=dict)
    n_visual_tokens: int = 0


def build_content(prompt_text: str, images: dict[int, Image.Image]) -> list[dict]:
    """Interleave images at their ``<image N>`` positions (first occurrence). Images the
    text never references are prepended. Repeated references stay as literal text."""
    parts: list[dict] = []
    used: set[int] = set()
    pos = 0
    for m in IMAGE_TOKEN_RE.finditer(prompt_text):
        k = int(m.group(1))
        if k in images and k not in used:
            chunk = prompt_text[pos:m.start()]
            if chunk:
                parts.append({"type": "text", "text": chunk})
            parts.append({"type": "image_pil", "image_pil": images[k]})
            used.add(k)
            pos = m.end()
    tail = prompt_text[pos:]
    prefix = [{"type": "image_pil", "image_pil": images[k]} for k in sorted(images) if k not in used]
    parts = prefix + parts
    if tail:
        parts.append({"type": "text", "text": tail})
    return parts


def build_example(sample: dict, subject: str, prompt_style: str, min_pixels: int, max_pixels: int,
                  token_budget: int) -> Example:
    qtype = sample["question_type"]
    options = parse_options(sample["options"]) if qtype == "multiple-choice" else []
    all_choices = [chr(ord("A") + i) for i in range(len(options))]
    index2ans = dict(zip(all_choices, options))
    prompt_text = build_prompt(prompt_style, qtype, sample["question"], options)

    images: dict[int, Image.Image] = {}
    for k in range(1, 8):
        img = sample.get(f"image_{k}")
        if img is not None:
            images[k] = resize_image(img, min_pixels, max_pixels)
    total = sum(visual_tokens(im) for im in images.values())
    if total > token_budget:  # shrink every image proportionally so the prompt fits max_model_len
        scale = token_budget / total
        for k, im in images.items():
            per_img = max(min_pixels, int(im.width * im.height * scale))
            images[k] = resize_image(im, min_pixels, per_img)
        total = sum(visual_tokens(im) for im in images.values())

    return Example(
        id=sample["id"], subject=subject, question_type=qtype, question=sample["question"],
        options=options, all_choices=all_choices, index2ans=index2ans,
        gold=parse_gold(sample["answer"], qtype), prompt_text=prompt_text,
        content=build_content(prompt_text, images),
        image_sizes={k: [im.width, im.height] for k, im in images.items()},
        n_visual_tokens=total,
    )
