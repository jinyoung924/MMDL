"""Answer parsing / scoring.

``parse_multi_choice_response``, ``parse_open_response``, ``eval_multi_choice``,
``eval_open`` and helpers are ported verbatim from the official MMMU evaluation code:
https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/eval_utils.py

Two deliberate changes, both documented in the report:
1. The random fallback (no candidate found) uses an injected ``random.Random``
   instance so results are deterministic, and the caller is told a fallback happened.
2. ``extract_answer_line`` is an optional pre-step used only for the CoT ablation
   (``mmmu_pro_cot``), following MMMU-Pro's ``Answer: X`` convention.
"""
import random
import re

import numpy as np


def parse_multi_choice_response(response, all_choices, index2ans, rng: random.Random):
    """Return (predicted index e.g. 'A', used_random_fallback)."""
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f" {choice} " in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        return rng.choice(all_choices), True
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    start_indexes.append(response.rfind(f"({can})"))
            else:
                for can in candidates:
                    start_indexes.append(response.rfind(f" {can} "))
        else:
            for can in candidates:
                start_indexes.append(response.lower().rfind(index2ans[can].lower()))
        # get the last one
        pred_index = candidates[int(np.argmax(start_indexes))]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]
    return pred_index, False


def check_is_number(string):
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        return False


def normalize_str(string):
    string = string.strip()
    is_number = check_is_number(string)
    if is_number:
        string = string.replace(",", "")
        string = float(string)
        string = round(string, 2)
        return [string]
    else:
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "]  # avoid trivial matches
        return [string]


def extract_numbers(string):
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"
    numbers_with_commas = re.findall(pattern_commas, string)
    numbers_scientific = re.findall(pattern_scientific, string)
    numbers_simple = re.findall(pattern_simple, string)
    return numbers_with_commas + numbers_scientific + numbers_simple


def parse_open_response(response):
    def get_key_subresponses(response):
        key_responses = []
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", response)
        indicators_of_keys = ["could be ", "so ", "is ", "thus ", "therefore ", "final ", "answer ", "result "]
        key_responses = []
        for index, resp in enumerate(sub_responses):
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(["="])
            shortest_key_response = None
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                            shortest_key_response = resp.split(indicator)[-1].strip()
            if shortest_key_response:
                if shortest_key_response.strip() not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:
            return [response]
        return key_responses

    key_responses = get_key_subresponses(response)
    pred_list = key_responses.copy()
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))
    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))
    pred_list = tmp_pred_list
    pred_list = list(set(pred_list))
    return pred_list


def eval_multi_choice(gold_i, pred_i):
    if isinstance(gold_i, list):
        return any(answer == pred_i for answer in gold_i)
    return gold_i == pred_i


def eval_open(gold_i, pred_i):
    correct = False
    if isinstance(gold_i, list):
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)
    for pred in pred_i:
        if isinstance(pred, str):
            for norm_ans in norm_answers:
                if isinstance(norm_ans, str) and norm_ans in pred:
                    correct = True
                    break
        else:
            if pred in norm_answers:
                correct = True
                break
    return correct


_ANSWER_LINE_RE = re.compile(r"answer\s*:\s*([^\n]+?)\s*$", re.IGNORECASE | re.MULTILINE)
_MARKUP_RE = re.compile(r"\\boxed|\\text|[*$`{}\\]")


def extract_answer_line(response: str) -> str | None:
    """MMMU-Pro style: take the LAST ``Answer: X`` line if present (CoT prompt only).
    Markdown/LaTeX wrappers (``**B**``, ``$C``, ``\\boxed{B}``) are stripped so the official
    parser below sees ``B`` / ``(B) text``; parentheses are kept on purpose."""
    matches = _ANSWER_LINE_RE.findall(response)
    if not matches:
        return None
    return _MARKUP_RE.sub(" ", matches[-1]).strip()


def grade(response: str, question_type: str, all_choices, index2ans, gold, rng: random.Random, answer_line_first: bool):
    """Return dict(pred, correct, parse_fallback)."""
    text = response
    if answer_line_first:
        line = extract_answer_line(response)
        if line is not None:
            text = line
    if question_type == "multiple-choice":
        pred, fallback = parse_multi_choice_response(text, all_choices, index2ans, rng)
        return {"pred": pred, "correct": bool(eval_multi_choice(gold, pred)), "parse_fallback": fallback}
    pred_list = parse_open_response(text)
    return {"pred": pred_list, "correct": bool(eval_open(gold, pred_list)), "parse_fallback": False}
