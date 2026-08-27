"""Cell-o1-aligned RLVR reward with GT-only privileged teacher feedback."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Optional

try:
    from cell_annotation.common import normalize_label
except ImportError:
    from common import normalize_label


STRICT_RESPONSE_RES = {
    "think": re.compile(
        r"^<think>.*?</think>(?:[ \t]*\r?\n)+[ \t]*<answer>.*?</answer>$",
        flags=re.DOTALL,
    ),
    "reasoning": re.compile(
        r"^<reasoning>.*?</reasoning>(?:[ \t]*\r?\n)+[ \t]*<answer>.*?</answer>$",
        flags=re.DOTALL,
    ),
}
REASONING_CONTENT_RES = {
    tag: re.compile(rf"<{tag}>(.*?)</{tag}>", flags=re.DOTALL)
    for tag in STRICT_RESPONSE_RES
}
PLACEHOLDER_REASONING_RE = re.compile(
    r"(?:see\s+(?:the\s+)?(?:detailed\s+)?(?:reasoning|thought\s+process|"
    r"reasoning\s+outline)|reasoning\s+attached|attached\s+reasoning|"
    r"enclosed\s+files?|article\s+body|derivation\s*&\s*resolution\s+report)",
    flags=re.IGNORECASE,
)
MIN_REASONING_CHARS = 200
MIN_REASONING_UNITS = 64


def _extract_answer(prediction: str) -> str:
    """Reproduce cell-o1's line-based answer extraction."""
    for line in prediction.splitlines():
        if "<answer>" in line:
            candidate = line.split("<answer>", 1)[1].strip()
            if candidate.endswith("</answer>"):
                candidate = candidate[: -len("</answer>")].strip()
            return candidate
    for line in prediction.splitlines():
        if "|" in line:
            return line.strip()
    return ""


def _strict_format(
    prediction: str, candidate: str, reasoning_tag: str = "think"
) -> bool:
    if reasoning_tag not in STRICT_RESPONSE_RES:
        raise ValueError(f"unsupported reasoning tag: {reasoning_tag!r}")
    return bool(
        prediction.count(f"<{reasoning_tag}>") == 1
        and prediction.count(f"</{reasoning_tag}>") == 1
        and prediction.count("<answer>") == 1
        and prediction.count("</answer>") == 1
        and STRICT_RESPONSE_RES[reasoning_tag].fullmatch(prediction)
        and "|" in candidate
    )


def _reasoning_quality(
    prediction: str, reasoning_tag: str = "think"
) -> Dict[str, Any]:
    """Return a conservative, model-independent reasoning sanity check.

    This is deliberately only a collapse guard.  It does not claim that a
    biologically plausible-looking rationale is correct; it prevents short or
    explicit placeholder ``<think>`` blocks from becoming privileged sibling
    demonstrations and from being reinforced by the GRPO branch.
    """
    if reasoning_tag not in REASONING_CONTENT_RES:
        raise ValueError(f"unsupported reasoning tag: {reasoning_tag!r}")
    match = REASONING_CONTENT_RES[reasoning_tag].search(prediction)
    reasoning = match.group(1).strip() if match else ""
    reasoning_chars = len(reasoning)
    reasoning_units = len(re.findall(r"\S+", reasoning))
    placeholder = bool(PLACEHOLDER_REASONING_RE.search(reasoning))
    normal = bool(
        reasoning_chars >= MIN_REASONING_CHARS
        and reasoning_units >= MIN_REASONING_UNITS
        and not placeholder
    )
    return {
        "reasoning_normal": normal,
        "reasoning_chars": reasoning_chars,
        "reasoning_units": reasoning_units,
        "reasoning_placeholder": placeholder,
    }


def _gt_teacher_feedback(ground_truth: str) -> str:
    # Deliberately exclude the student's attempted answer, diagnostics, marker
    # genes, and any offline reasoning trace. The causal teacher forward later
    # appends only the student's prefix y_<t>.
    return (
        "Ground-truth privileged information for training only:\n"
        "The verified cell-type assignment, in Cell order, is:\n"
        f"<answer>{ground_truth}</answer>\n"
        "Use this verified assignment to reason through and correctly solve "
        "the original cell-annotation problem."
    )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the exact cell-o1 reward and a strict rollout-routing flag.

    Cell-o1 assigns -1 to malformed outputs, answer-count mismatches, and
    duplicated labels. Otherwise its reward is:

        (positional cell accuracy + exact cell-batch match) / 2.

    ``route_to_sdpo`` is one only when the complete rollout is not a strict
    exact match. GT feedback is exposed only for those failed rollouts.
    """
    return _compute_score_impl(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        reasoning_tag="think",
    )


def compute_nonthinking_reasoning_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Dense Cell-o1 repo reward for the explicit non-thinking protocol.

    This accepts exactly ``<reasoning>...</reasoning>`` followed by a newline
    and ``<answer>...</answer>``. Native Qwen ``<think>`` output is rejected so
    the RL run cannot silently drift away from its reasoning-SFT contract.
    """
    return _compute_score_impl(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        reasoning_tag="reasoning",
    )


def _compute_score_impl(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]],
    reasoning_tag: str,
) -> Dict[str, Any]:
    del extra_info
    if "cell" not in str(data_source).casefold():
        raise ValueError(f"unexpected data source for cell reward: {data_source!r}")

    prediction = "" if solution_str is None else str(solution_str)
    candidate = _extract_answer(prediction)
    # Keep the same empty-field handling and duplicate test as cell-o1.
    gold_items = [item.strip() for item in ground_truth.split("|") if item.strip()]
    candidate_items = [item.strip() for item in candidate.split("|") if item.strip()]
    strict_format = _strict_format(prediction, candidate, reasoning_tag)

    answer_count_valid = bool(gold_items and len(candidate_items) == len(gold_items))
    unique_valid = len(candidate_items) == len(set(candidate_items))
    structurally_valid = strict_format and answer_count_valid and unique_valid

    num_correct = 0
    partial_accuracy = 0.0
    exact_match = False
    if structurally_valid:
        gold_norm = [normalize_label(item) for item in gold_items]
        candidate_norm = [normalize_label(item) for item in candidate_items]
        num_correct = sum(
            predicted == expected
            for predicted, expected in zip(candidate_norm, gold_norm)
        )
        partial_accuracy = num_correct / len(gold_norm)
        exact_match = num_correct == len(gold_norm)
        score = (partial_accuracy + float(exact_match)) / 2.0
        candidate_set_valid = Counter(candidate_norm) == Counter(gold_norm)
    else:
        score = -1.0
        candidate_set_valid = False

    strict_exact_match = bool(structurally_valid and exact_match)
    route_to_sdpo = not strict_exact_match
    return {
        "score": float(score),
        "feedback": _gt_teacher_feedback(ground_truth) if route_to_sdpo else None,
        "route_to_sdpo": float(route_to_sdpo),
        "route_to_grpo": float(strict_exact_match),
        "strict_format": float(strict_format),
        "format_reward": float(structurally_valid),
        "partial_accuracy": float(partial_accuracy),
        "exact_match": float(exact_match),
        "strict_exact_match": float(strict_exact_match),
        "answer_count_valid": float(answer_count_valid),
        "unique_valid": float(unique_valid),
        "candidate_set_valid": float(candidate_set_valid),
        "num_correct": float(num_correct),
        "num_cells": float(len(gold_items)),
    }


def compute_sparse_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return Cell-o1's paper-main sparse 1/0/-1 reward.

    A structurally valid, fully correct cell batch receives 1. A structurally
    valid but not fully correct batch receives 0. Malformed outputs, answer
    count mismatches, and duplicate assignments receive -1. Routing and GT
    teacher feedback are identical to :func:`compute_score`.
    """
    result = compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    if result["strict_exact_match"]:
        sparse_score = 1.0
    elif result["format_reward"]:
        sparse_score = 0.0
    else:
        sparse_score = -1.0
    result["score"] = sparse_score
    return result


def compute_nonthinking_reasoning_sparse_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Sparse 1/0/-1 reward with strict ``<reasoning>`` output format."""
    result = compute_nonthinking_reasoning_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    if result["strict_exact_match"]:
        result["score"] = 1.0
    elif result["format_reward"]:
        result["score"] = 0.0
    else:
        result["score"] = -1.0
    return result


def _add_sibling_srpo_metadata(
    result: Dict[str, Any], solution_str: str, reasoning_tag: str = "think"
) -> Dict[str, Any]:
    """Attach reasoning-aware sibling eligibility without changing reward."""
    quality = _reasoning_quality(
        "" if solution_str is None else str(solution_str), reasoning_tag
    )
    teacher_eligible = bool(result["strict_exact_match"] and quality["reasoning_normal"])
    result.update(quality)
    result.update(
        {
            "feedback": None,
            "teacher_eligible": float(teacher_eligible),
            "route_to_sdpo": float(not teacher_eligible),
            "route_to_grpo": float(teacher_eligible),
        }
    )
    return result


def compute_sparse_sibling_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Sparse Cell-o1 reward plus reasoning-aware sibling-SRPO metadata.

    The scalar reward remains the paper-aligned 1/0/-1 reward.  A rollout is a
    valid successful sibling only when the complete cell batch is exactly
    correct and its visible reasoning passes the collapse guard.  The trainer
    resolves the final route: requested corrections use SDPO only if such a
    sibling exists in the same rollout group; otherwise they fall back to
    GRPO.  No ground-truth feedback is exposed to the teacher in this mode.
    """
    result = compute_sparse_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    return _add_sibling_srpo_metadata(result, solution_str)


def compute_repo_sibling_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Cell-o1 repo dense reward plus reasoning-aware sibling-SRPO metadata.

    Structurally valid rollouts receive ``(positional accuracy + exact match) /
    2`` and invalid rollouts receive ``-1``.  Sibling eligibility and routing
    are otherwise identical to :func:`compute_sparse_sibling_score`.
    """
    result = compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    return _add_sibling_srpo_metadata(result, solution_str)


def compute_nonthinking_reasoning_repo_sibling_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Dense repo reward + sibling routing for explicit non-thinking reasoning."""
    result = compute_nonthinking_reasoning_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    return _add_sibling_srpo_metadata(
        result, solution_str, reasoning_tag="reasoning"
    )


def compute_nonthinking_reasoning_sparse_sibling_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Sparse 1/0/-1 reward + sibling routing for explicit reasoning.

    A rollout is a qualified sibling only when the complete cell batch is
    exactly correct and its ``<reasoning>`` block passes the collapse guard.
    Answer-wrong rollouts expose no GT feedback: the trainer either uses a
    qualified leave-one-out sibling from the K-rollout group or falls back to
    sparse GRPO.
    """
    result = compute_nonthinking_reasoning_sparse_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    return _add_sibling_srpo_metadata(
        result, solution_str, reasoning_tag="reasoning"
    )


def compute_nonthinking_reasoning_joint_gt_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Dense GRPO reward plus an always-on GT privilege for joint training."""
    result = compute_nonthinking_reasoning_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    # Joint SDPO consumes this field for every rollout. Keep ``feedback`` empty
    # so the GT-only ablation cannot accidentally acquire O1 or sibling context.
    result["teacher_base_feedback"] = _gt_teacher_feedback(ground_truth)
    result["feedback"] = None
    return result


def compute_nonthinking_reasoning_joint_hierarchical_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Dense reward with GT anchor and sibling-first/O1-fallback privilege.

    The trainer always includes ``teacher_base_feedback``. A qualified exact
    leave-one-out sibling is preferred as the process demonstration; when it
    is unavailable, ``feedback`` supplies the converted offline O1 trace.
    """
    result = compute_nonthinking_reasoning_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    result = _add_sibling_srpo_metadata(
        result, solution_str, reasoning_tag="reasoning"
    )
    result["teacher_base_feedback"] = _gt_teacher_feedback(ground_truth)

    o1_reasoning = None
    if isinstance(extra_info, dict):
        raw_trace = extra_info.get("o1_reasoning")
        if isinstance(raw_trace, str) and raw_trace.strip():
            o1_reasoning = raw_trace.strip().replace(
                "<think>", "<reasoning>"
            ).replace("</think>", "</reasoning>")
    result["o1_reasoning_available"] = float(o1_reasoning is not None)
    # Unlike routed correction, all-rollout joint training may also use O1 for
    # a correct rollout when no distinct qualified sibling exists.
    result["feedback"] = o1_reasoning
    return result


def compute_repo_o1_fallback_sibling_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Repo reward with sibling-first and offline-o1 fallback supervision.

    The rollout requests the same reasoning-aware sibling routing as
    :func:`compute_repo_sibling_score`.  For an answer-wrong rollout only, an
    available CellPuzzles o1 trace is exposed as environment feedback.  The
    trainer uses it only when no qualified sibling exists in the K-rollout
    group; samples without either source fall back to GRPO.
    """
    result = compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    result = _add_sibling_srpo_metadata(result, solution_str)
    return _add_o1_fallback_metadata(result, extra_info)


def compute_nonthinking_reasoning_repo_o1_fallback_sibling_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Dense reward + sibling/o1 routing for non-thinking explicit reasoning."""
    result = compute_nonthinking_reasoning_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    result = _add_sibling_srpo_metadata(
        result, solution_str, reasoning_tag="reasoning"
    )
    return _add_o1_fallback_metadata(
        result, extra_info, convert_to_reasoning_tags=True
    )


def compute_nonthinking_reasoning_sparse_o1_fallback_sibling_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Sparse reward + sibling-first/O1-fallback explicit reasoning route."""
    result = compute_nonthinking_reasoning_sparse_sibling_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    return _add_o1_fallback_metadata(
        result, extra_info, convert_to_reasoning_tags=True
    )


def compute_sparse_o1_fallback_sibling_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Sparse reward with sibling-first and offline-o1 fallback supervision.

    This is identical to :func:`compute_repo_o1_fallback_sibling_score` in
    routing and privileged feedback, but preserves the paper-main 1/0/-1
    scalar reward used by :func:`compute_sparse_score`.
    """
    result = compute_sparse_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    result = _add_sibling_srpo_metadata(result, solution_str)
    return _add_o1_fallback_metadata(result, extra_info)


def _add_o1_fallback_metadata(
    result: Dict[str, Any],
    extra_info: Optional[Dict[str, Any]],
    convert_to_reasoning_tags: bool = False,
) -> Dict[str, Any]:
    """Expose an offline o1 trace only for an answer-wrong rollout."""
    o1_reasoning = None
    if isinstance(extra_info, dict):
        raw_trace = extra_info.get("o1_reasoning")
        if isinstance(raw_trace, str) and raw_trace.strip():
            o1_reasoning = raw_trace.strip()
            if convert_to_reasoning_tags:
                o1_reasoning = o1_reasoning.replace(
                    "<think>", "<reasoning>"
                ).replace("</think>", "</reasoning>")
    answer_wrong = not bool(result["strict_exact_match"])
    result["o1_reasoning_available"] = float(o1_reasoning is not None)
    result["o1_fallback_requested"] = float(answer_wrong and o1_reasoning is not None)
    result["feedback"] = o1_reasoning if answer_wrong else None
    return result
