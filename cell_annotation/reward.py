"""Rule reward and ground-truth privileged feedback for SDPO."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    from cell_annotation.common import score_prediction
except ImportError:
    from common import score_prediction


def _weight(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return scalar reward plus feedback consumed only by the SDPO teacher.

    Default reward:
      0.10 * valid output format and candidate permutation
      0.45 * positional cell accuracy
      0.45 * exact batch match
    """
    del extra_info
    metrics = score_prediction(solution_str, ground_truth)
    format_weight = _weight("CELL_REWARD_FORMAT_WEIGHT", 0.10)
    partial_weight = _weight("CELL_REWARD_PARTIAL_WEIGHT", 0.45)
    exact_weight = _weight("CELL_REWARD_EXACT_WEIGHT", 0.45)
    total_weight = format_weight + partial_weight + exact_weight
    if abs(total_weight - 1.0) > 1e-8:
        raise ValueError(
            "CELL_REWARD_FORMAT_WEIGHT + CELL_REWARD_PARTIAL_WEIGHT + "
            f"CELL_REWARD_EXACT_WEIGHT must equal 1, got {total_weight}"
        )

    score = (
        format_weight * float(metrics["valid_output_format"])
        + partial_weight * float(metrics["partial_accuracy"])
        + exact_weight * float(metrics["exact_match"])
    )
    attempted = metrics["extracted_answer"] or "(no parseable answer)"
    feedback = (
        "Privileged training information (available to the teacher only; it is "
        "not available to the student at inference time):\n"
        f"The verified ground-truth assignment in Cell order is:\n"
        f"<answer>{ground_truth}</answer>\n"
        "Under Qwen3 non-thinking mode, the chat template already supplies an "
        "empty <think></think> prefix, so the generated continuation must be "
        "exactly one <answer>...</answer> block. An explicit "
        "<think>...</think><answer>...</answer> response is also accepted. "
        "The answer must contain one "
        "unique candidate label per cell, in Cell order, separated by ' | '.\n"
        f"Earlier attempted answer: {attempted}\n"
        f"Diagnostics: strict_format={metrics['strict_format']}; "
        f"positionally_correct={metrics['num_correct']}/{metrics['num_cells']}; "
        f"valid_candidate_permutation={metrics['candidate_set_valid']}; "
        f"exact_match={metrics['exact_match']}.\n"
        "Use the verified assignment to retrospectively produce the best "
        "reasoning and correctly solve the original question."
    )

    return {
        "score": float(score),
        "feedback": feedback,
        "format_reward": float(metrics["valid_output_format"]),
        "partial_accuracy": float(metrics["partial_accuracy"]),
        "exact_match": float(metrics["exact_match"]),
        "strict_exact_match": float(metrics["strict_exact_match"]),
        "answer_count_valid": float(metrics["answer_count_valid"]),
        "candidate_set_valid": float(metrics["candidate_set_valid"]),
    }
