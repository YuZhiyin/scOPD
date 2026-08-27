"""Shared parsing and scoring helpers for CellPuzzles-style annotation."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


STRICT_RESPONSE_RE = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*"
    r"<answer>(?P<answer>.*?)</answer>\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)
STRICT_REASONING_RESPONSE_RE = re.compile(
    r"^\s*<reasoning>(?P<reasoning>.*?)</reasoning>\s*"
    r"<answer>(?P<answer>.*?)</answer>\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)
STRICT_ANSWER_ONLY_RE = re.compile(
    r"^\s*<answer>(?P<answer>.*?)</answer>\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)
ANSWER_TAG_RE = re.compile(
    r"<answer>(?P<answer>.*?)</answer>",
    flags=re.DOTALL | re.IGNORECASE,
)


def normalize_label(value: str) -> str:
    """Normalize surface-only differences without merging biological labels."""
    return " ".join(str(value).strip().split()).casefold()


def split_assignments(answer: str) -> List[str]:
    if not answer or not str(answer).strip():
        return []
    return [item.strip() for item in str(answer).strip().split("|")]


def parse_prediction(text: str) -> Tuple[List[str], bool, Optional[str]]:
    """Return assignments, strict-format validity, and extracted answer text."""
    text = "" if text is None else str(text)
    strict_match = STRICT_RESPONSE_RE.fullmatch(text)
    if strict_match:
        answer = strict_match.group("answer").strip()
        return split_assignments(answer), True, answer
    reasoning_match = STRICT_REASONING_RESPONSE_RE.fullmatch(text)
    if reasoning_match and reasoning_match.group("reasoning").strip():
        answer = reasoning_match.group("answer").strip()
        return split_assignments(answer), True, answer
    # Qwen3's enable_thinking=False template pre-fills an empty
    # <think></think> block in the assistant prefix. vLLM returns only the
    # generated continuation, so answer-only is the strict continuation format.
    answer_only_match = STRICT_ANSWER_ONLY_RE.fullmatch(text)
    if answer_only_match:
        answer = answer_only_match.group("answer").strip()
        return split_assignments(answer), True, answer

    matches = list(ANSWER_TAG_RE.finditer(text))
    if matches:
        answer = matches[-1].group("answer").strip()
        return split_assignments(answer), False, answer

    # A relaxed fallback is useful for diagnosing biologically correct but
    # improperly formatted generations. It never counts as strict format.
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    pipe_lines = [line for line in nonempty_lines if "|" in line]
    if pipe_lines:
        answer = pipe_lines[-1]
        return split_assignments(answer), False, answer
    return [], False, None


def score_prediction(prediction: str, ground_truth: str) -> Dict[str, Any]:
    """Compute strict-format, positional, permutation, and exact-match metrics."""
    gold = split_assignments(ground_truth)
    pred, strict_format, extracted = parse_prediction(prediction)
    gold_norm = [normalize_label(item) for item in gold]
    pred_norm = [normalize_label(item) for item in pred]

    num_gold = len(gold_norm)
    num_correct = sum(
        pred_norm[index] == label
        for index, label in enumerate(gold_norm)
        if index < len(pred_norm)
    )
    partial_accuracy = num_correct / num_gold if num_gold else 0.0
    answer_count_valid = len(pred_norm) == num_gold
    unique_valid = len(pred_norm) == len(set(pred_norm))
    candidate_set_valid = (
        answer_count_valid and Counter(pred_norm) == Counter(gold_norm)
    )
    exact_match = bool(
        answer_count_valid and num_correct == num_gold and num_gold > 0
    )
    valid_output_format = bool(strict_format and candidate_set_valid and unique_valid)

    return {
        "strict_format": strict_format,
        "extracted_answer": extracted,
        "predicted_assignments": pred,
        "gold_assignments": gold,
        "num_cells": num_gold,
        "num_correct": num_correct,
        "partial_accuracy": partial_accuracy,
        "answer_count_valid": answer_count_valid,
        "unique_valid": unique_valid,
        "candidate_set_valid": candidate_set_valid,
        "valid_output_format": valid_output_format,
        "exact_match": exact_match,
        "strict_exact_match": bool(strict_format and exact_match),
    }


def read_records(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
