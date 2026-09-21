"""Prompt templates.

Baseline (``mmmu_pro_cot``) is the official MMMU-Pro CoT instruction
(https://github.com/MMMU-Benchmark/MMMU/tree/main/mmmu-pro, prompts.yaml), appended to the question
and the options rendered exactly like ``construct_prompt`` in
https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/data_utils.py (``(A) text\\n(B) text\\n...``).

``mmmu_direct`` is the official MMMU direct-answer template, copied verbatim from
https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/configs/llava1.5.yaml. It is kept as an
ablation: with it Qwen3-VL-4B-Instruct still writes a long solution for ~1/3 of the questions, so the
model's behaviour is mixed and the score depends on how many of those get cut off (see the report).
"""

TEMPLATES = {
    "mmmu_direct": {
        "multiple-choice": "{question}\n\n{options}\n\nAnswer with the option's letter from the given choices directly.",
        "open": "{question}\n\nAnswer the question using a single word or phrase.",
    },
    "mmmu_pro_cot": {
        "multiple-choice": (
            "{question}\n\n{options}\n\nAnswer the preceding multiple choice question. "
            "The last line of your response should be of the following format: "
            "'Answer: $LETTER' (without quotes) where LETTER is one of options. "
            "Think step by step before answering."
        ),
        "open": (
            "{question}\n\nAnswer the preceding question. The last line of your response "
            "should be of the following format: 'Answer: $ANSWER' (without quotes) "
            "where ANSWER is your final answer. Think step by step before answering."
        ),
    },
}


def format_options(options: list[str]) -> str:
    """Official MMMU rendering: ``(A) opt\\n(B) opt\\n`` (trailing newline kept as in the original code)."""
    out = ""
    for i, opt in enumerate(options):
        out += f"({chr(ord('A') + i)}) {opt}\n"
    return out


def build_prompt(style: str, question_type: str, question: str, options: list[str]) -> str:
    tpl = TEMPLATES[style]
    if question_type == "multiple-choice":
        return tpl["multiple-choice"].format(question=question, options=format_options(options))
    return tpl["open"].format(question=question)
