"""Pure helpers for disagreement-gated pairwise refinement (DGPR).

The routing and selection functions in this module deliberately receive only
the candidate labels and model generations.  They never receive test labels;
ground truth is reserved for post-hoc evaluation.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from cell_annotation.common import (
        STRICT_REASONING_RESPONSE_RE,
        normalize_label,
        split_assignments,
    )
except ImportError:
    from common import (  # type: ignore
        STRICT_REASONING_RESPONSE_RE,
        normalize_label,
        split_assignments,
    )


CANDIDATE_HEADER = "Match the cells above to one of the following cell types:"
MIN_REASONING_CHARS = 200
MIN_REASONING_UNITS = 64
PLACEHOLDER_REASONING_RE = re.compile(
    r"(?:see\s+(?:the\s+)?(?:detailed\s+)?(?:reasoning|thought\s+process|"
    r"reasoning\s+outline)|reasoning\s+attached|attached\s+reasoning|"
    r"enclosed\s+files?|article\s+body|derivation\s*&\s*resolution\s+report)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedRollout:
    index: int
    text: str
    assignments: Tuple[str, ...]
    normalized_assignments: Tuple[str, ...]
    reasoning: str
    valid: bool
    invalid_reason: Optional[str]
    reasoning_chars: int
    reasoning_units: int


@dataclass(frozen=True)
class DGPRPlan:
    anchor_index: Optional[int]
    challenger_index: Optional[int]
    valid_indices: Tuple[int, ...]
    stable_indices: Tuple[int, ...]
    uncertain_indices: Tuple[int, ...]
    vote_counts: Tuple[Dict[str, int], ...]
    trigger_refinement: bool
    route_reason: str


def parse_candidate_labels(user_message: str) -> List[str]:
    """Extract the candidate list from a CellPuzzles prompt."""
    if CANDIDATE_HEADER not in user_message:
        raise ValueError("CellPuzzles candidate header is missing from user prompt")
    candidate_block = user_message.rsplit(CANDIDATE_HEADER, 1)[1]
    candidates = [line.strip() for line in candidate_block.splitlines() if line.strip()]
    if not candidates:
        raise ValueError("CellPuzzles prompt has an empty candidate list")
    normalized = [normalize_label(label) for label in candidates]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"candidate list contains duplicate labels: {candidates!r}")
    return candidates


def _reasoning_is_normal(reasoning: str) -> Tuple[bool, int, int]:
    reasoning = reasoning.strip()
    chars = len(reasoning)
    units = len(re.findall(r"\S+", reasoning))
    normal = bool(
        chars >= MIN_REASONING_CHARS
        and units >= MIN_REASONING_UNITS
        and not PLACEHOLDER_REASONING_RE.search(reasoning)
    )
    return normal, chars, units


def parse_rollout(index: int, text: str, candidates: Sequence[str]) -> ParsedRollout:
    """Parse one rollout and apply the format/permutation/collapse guards."""
    text = "" if text is None else str(text)
    match = STRICT_REASONING_RESPONSE_RE.fullmatch(text)
    if not match:
        return ParsedRollout(index, text, (), (), "", False, "strict_format", 0, 0)

    reasoning = match.group("reasoning").strip()
    assignments = tuple(split_assignments(match.group("answer").strip()))
    normalized_assignments = tuple(normalize_label(value) for value in assignments)
    normalized_candidates = tuple(normalize_label(value) for value in candidates)
    normal, chars, units = _reasoning_is_normal(reasoning)

    invalid_reason: Optional[str] = None
    if len(assignments) != len(candidates):
        invalid_reason = "answer_count"
    elif len(normalized_assignments) != len(set(normalized_assignments)):
        invalid_reason = "duplicate_labels"
    elif Counter(normalized_assignments) != Counter(normalized_candidates):
        invalid_reason = "candidate_set"
    elif not normal:
        invalid_reason = "reasoning_collapse"

    return ParsedRollout(
        index=index,
        text=text,
        assignments=assignments,
        normalized_assignments=normalized_assignments,
        reasoning=reasoning,
        valid=invalid_reason is None,
        invalid_reason=invalid_reason,
        reasoning_chars=chars,
        reasoning_units=units,
    )


def _hamming(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) != len(right):
        raise ValueError("cannot compare assignments with different lengths")
    return sum(a != b for a, b in zip(left, right))


def build_dgpr_plan(
    rollouts: Sequence[ParsedRollout],
    num_cells: int,
    consensus_threshold: float = 0.75,
) -> DGPRPlan:
    """Select an answer medoid and a supported competing reasoning trace."""
    if not 0.5 < consensus_threshold <= 1.0:
        raise ValueError("consensus_threshold must be in (0.5, 1.0]")
    valid = [rollout for rollout in rollouts if rollout.valid]
    if not valid:
        return DGPRPlan(None, None, (), (), tuple(range(num_cells)), (), False, "no_valid_rollout")

    medoid_costs = {
        rollout.index: sum(
            _hamming(rollout.normalized_assignments, other.normalized_assignments)
            for other in valid
            if other.index != rollout.index
        )
        for rollout in valid
    }
    anchor = min(valid, key=lambda rollout: (medoid_costs[rollout.index], rollout.index))

    vote_counts: List[Dict[str, int]] = []
    for cell_index in range(num_cells):
        votes = Counter(
            rollout.normalized_assignments[cell_index] for rollout in valid
        )
        vote_counts.append(dict(sorted(votes.items())))

    stable: List[int] = []
    uncertain: List[int] = []
    for cell_index, votes in enumerate(vote_counts):
        anchor_label = anchor.normalized_assignments[cell_index]
        confidence = votes.get(anchor_label, 0) / len(valid)
        max_votes = max(votes.values())
        anchor_is_mode = votes.get(anchor_label, 0) == max_votes
        if anchor_is_mode and confidence >= consensus_threshold:
            stable.append(cell_index)
        else:
            uncertain.append(cell_index)

    if len(valid) < 2:
        return DGPRPlan(
            anchor.index,
            None,
            tuple(item.index for item in valid),
            tuple(stable),
            tuple(uncertain),
            tuple(vote_counts),
            False,
            "fewer_than_two_valid_rollouts",
        )
    if not uncertain:
        return DGPRPlan(
            anchor.index,
            None,
            tuple(item.index for item in valid),
            tuple(stable),
            (),
            tuple(vote_counts),
            False,
            "all_cells_stable",
        )

    challengers = [
        rollout
        for rollout in valid
        if rollout.index != anchor.index
        and any(
            rollout.normalized_assignments[cell_index]
            != anchor.normalized_assignments[cell_index]
            for cell_index in uncertain
        )
    ]
    if not challengers:
        return DGPRPlan(
            anchor.index,
            None,
            tuple(item.index for item in valid),
            tuple(stable),
            tuple(uncertain),
            tuple(vote_counts),
            False,
            "no_supported_challenger",
        )

    def challenger_key(rollout: ParsedRollout) -> Tuple[int, int, int]:
        support = sum(
            vote_counts[cell_index].get(
                rollout.normalized_assignments[cell_index], 0
            )
            for cell_index in uncertain
        )
        disagreement = sum(
            rollout.normalized_assignments[cell_index]
            != anchor.normalized_assignments[cell_index]
            for cell_index in uncertain
        )
        return support, disagreement, -rollout.index

    challenger = max(challengers, key=challenger_key)
    return DGPRPlan(
        anchor.index,
        challenger.index,
        tuple(item.index for item in valid),
        tuple(stable),
        tuple(uncertain),
        tuple(vote_counts),
        True,
        "disagreement",
    )


def compact_trace(text: str, max_reasoning_chars: int) -> str:
    """Shorten only the middle of a trace while preserving its XML structure."""
    match = STRICT_REASONING_RESPONSE_RE.fullmatch(text)
    if not match:
        return text[:max_reasoning_chars]
    reasoning = match.group("reasoning").strip()
    answer = match.group("answer").strip()
    if len(reasoning) > max_reasoning_chars:
        head = int(max_reasoning_chars * 0.6)
        tail = max_reasoning_chars - head
        reasoning = (
            reasoning[:head].rstrip()
            + "\n[... middle of reasoning omitted only for context length ...]\n"
            + reasoning[-tail:].lstrip()
        )
    return f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>{answer}</answer>"


def build_refinement_instruction(
    anchor: ParsedRollout,
    challenger: ParsedRollout,
    plan: DGPRPlan,
    candidate_surfaces: Sequence[str],
) -> str:
    """Build the test-time peer-refinement instruction without any GT data."""
    uncertain_lines = []
    for cell_index in plan.uncertain_indices:
        labels = [
            candidate_surfaces[index]
            for index, candidate in enumerate(candidate_surfaces)
            if normalize_label(candidate) in plan.vote_counts[cell_index]
        ]
        uncertain_lines.append(
            f"- Cell {cell_index + 1}: competing labels observed: " + " / ".join(labels)
        )
    stable_cells = ", ".join(str(index + 1) for index in plan.stable_indices) or "none"
    return (
        "\n\nTest-time sibling reasoning refinement:\n"
        "Multiple independent attempts were generated for this same cell batch. "
        "The two representative attempts below disagree on some cells. Re-solve "
        "the disputed assignments from the biological evidence in the original "
        "input; do not choose an attempt merely because it appears first.\n\n"
        "Disputed cells:\n"
        + "\n".join(uncertain_lines)
        + f"\nStable cells that must keep the Anchor assignments: {stable_cells}.\n\n"
        "Anchor attempt:\n"
        f"{anchor.text}\n\n"
        "Alternative attempt:\n"
        f"{challenger.text}\n\n"
        "Requirements:\n"
        "1. Verify marker-gene evidence directly against the original input.\n"
        "2. Reconsider the disputed cells while preserving every listed stable cell.\n"
        "3. Use every candidate cell type exactly once.\n"
        "4. Write a new self-contained reasoning process, not a critique or vote.\n"
        "5. Output exactly <reasoning>...</reasoning> followed by "
        "<answer>cell type 1 | cell type 2 | ...</answer>."
    )


def validate_refinement(
    text: str,
    candidates: Sequence[str],
    anchor: ParsedRollout,
    plan: DGPRPlan,
) -> Tuple[bool, str, ParsedRollout]:
    """Apply conservative, GT-free acceptance checks to a refined response."""
    parsed = parse_rollout(-1, text, candidates)
    if not parsed.valid:
        return False, parsed.invalid_reason or "invalid_refinement", parsed
    for cell_index in plan.stable_indices:
        if (
            parsed.normalized_assignments[cell_index]
            != anchor.normalized_assignments[cell_index]
        ):
            return False, "changed_stable_cell", parsed
    for cell_index in plan.uncertain_indices:
        if parsed.normalized_assignments[cell_index] not in plan.vote_counts[cell_index]:
            return False, "unsupported_new_label", parsed
    return True, "accepted", parsed


def plan_to_dict(plan: DGPRPlan) -> Dict[str, Any]:
    return {
        "anchor_index": plan.anchor_index,
        "challenger_index": plan.challenger_index,
        "valid_indices": list(plan.valid_indices),
        "stable_cells": [index + 1 for index in plan.stable_indices],
        "uncertain_cells": [index + 1 for index in plan.uncertain_indices],
        "vote_counts": [dict(votes) for votes in plan.vote_counts],
        "trigger_refinement": plan.trigger_refinement,
        "route_reason": plan.route_reason,
    }

