"""GT-free helpers for one-sample, one-refinement cell annotation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from cell_annotation.common import normalize_label
    from cell_annotation.dgpr import ParsedRollout, parse_rollout
except ImportError:
    from common import normalize_label  # type: ignore
    from dgpr import ParsedRollout, parse_rollout  # type: ignore


CELL_BLOCK_RE = re.compile(
    r"(?:^|\n)Cell\s+(?P<cell>\d+):\s*(?P<genes>.*?)(?=\nCell\s+\d+:|\Z)",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class VerifiedIssue:
    cell_index: int
    swap_index: int
    current_label: str
    alternative_label: str
    supporting_genes: Tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class CriticDecision:
    verdict: str
    issue: Optional[VerifiedIssue]
    valid: bool
    status: str
    raw: Dict[str, Any]


def parse_cell_genes(user_message: str, candidate_header: str) -> List[set[str]]:
    """Parse the supplied top genes for each cell, without using test labels."""
    cell_text = user_message.split(candidate_header, 1)[0]
    matches = list(CELL_BLOCK_RE.finditer(cell_text))
    if not matches:
        raise ValueError("no Cell N gene blocks found in user prompt")
    result: List[set[str]] = []
    expected = 1
    for match in matches:
        cell_number = int(match.group("cell"))
        if cell_number != expected:
            raise ValueError(
                f"cell blocks are not consecutive: expected {expected}, got {cell_number}"
            )
        genes = {
            gene.strip().casefold()
            for gene in match.group("genes").replace("\n", " ").split(",")
            if gene.strip()
        }
        if not genes:
            raise ValueError(f"Cell {cell_number} has no parsed genes")
        result.append(genes)
        expected += 1
    return result


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Tolerate a fenced JSON object but reject non-object critic outputs."""
    text = "" if text is None else str(text).strip()
    decoder = json.JSONDecoder()
    for start in [index for index, char in enumerate(text) if char == "{"]:
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def verify_critic_output(
    text: str,
    initial: ParsedRollout,
    candidates: Sequence[str],
    cell_genes: Sequence[set[str]],
    min_supporting_genes: int = 2,
) -> CriticDecision:
    """Accept REVISE only when its proposed swap has input-grounded evidence."""
    raw = extract_json_object(text)
    if raw is None:
        return CriticDecision("INVALID", None, False, "invalid_json", {})
    verdict = str(raw.get("verdict", "")).strip().upper()
    if verdict == "ACCEPT":
        return CriticDecision("ACCEPT", None, True, "critic_accept", raw)
    if verdict != "REVISE":
        return CriticDecision(verdict or "INVALID", None, False, "invalid_verdict", raw)
    issue = raw.get("issue")
    if not isinstance(issue, dict):
        return CriticDecision(verdict, None, False, "missing_issue", raw)
    try:
        cell_index = int(issue["cell"]) - 1
        swap_index = int(issue["swap_partner"]) - 1
    except (KeyError, TypeError, ValueError):
        return CriticDecision(verdict, None, False, "invalid_cell_indices", raw)
    if not (0 <= cell_index < len(candidates)) or not (
        0 <= swap_index < len(candidates)
    ) or cell_index == swap_index:
        return CriticDecision(verdict, None, False, "invalid_cell_indices", raw)

    current = str(issue.get("current_label", "")).strip()
    alternative = str(issue.get("alternative_label", "")).strip()
    current_norm = normalize_label(current)
    alternative_norm = normalize_label(alternative)
    candidate_norms = {normalize_label(label) for label in candidates}
    if current_norm != initial.normalized_assignments[cell_index]:
        return CriticDecision(verdict, None, False, "current_label_mismatch", raw)
    if alternative_norm not in candidate_norms or alternative_norm == current_norm:
        return CriticDecision(verdict, None, False, "invalid_alternative", raw)
    if initial.normalized_assignments[swap_index] != alternative_norm:
        return CriticDecision(verdict, None, False, "invalid_swap_partner", raw)

    supporting = issue.get("supporting_genes", [])
    if not isinstance(supporting, list):
        return CriticDecision(verdict, None, False, "invalid_supporting_genes", raw)
    supporting_clean = tuple(
        dict.fromkeys(str(gene).strip() for gene in supporting if str(gene).strip())
    )
    grounded = [
        gene
        for gene in supporting_clean
        if gene.casefold() in cell_genes[cell_index]
    ]
    if len(grounded) < min_supporting_genes or len(grounded) != len(supporting_clean):
        return CriticDecision(verdict, None, False, "ungrounded_supporting_genes", raw)
    reason = str(issue.get("reason", "")).strip()
    if not reason:
        return CriticDecision(verdict, None, False, "missing_reason", raw)
    verified = VerifiedIssue(
        cell_index=cell_index,
        swap_index=swap_index,
        current_label=current,
        alternative_label=alternative,
        supporting_genes=tuple(grounded),
        reason=reason,
    )
    return CriticDecision(verdict, verified, True, "verified_revision", raw)


def build_critic_instruction(initial_text: str) -> str:
    return (
        "\n\nCounterfactual verification step:\n"
        "Review the annotation attempt below against the original cell genes. "
        "Look for one concrete error that can be repaired by swapping two candidate "
        "labels. Do not request revision based on vague uncertainty. A REVISE verdict "
        "must cite at least two marker genes that literally occur in the disputed "
        "cell's original gene list.\n\n"
        f"Annotation attempt:\n{initial_text}\n\n"
        "Return JSON only in one of these forms:\n"
        '{"verdict":"ACCEPT","issue":null}\n'
        "or\n"
        '{"verdict":"REVISE","issue":{"cell":3,"current_label":"...",'
        '"alternative_label":"...","swap_partner":6,"supporting_genes":'
        '["GENE1","GENE2"],"reason":"..."}}'
    )


def build_refinement_instruction(
    initial_text: str,
    issue: Optional[VerifiedIssue],
    structural_reason: Optional[str] = None,
) -> str:
    if issue is not None:
        evidence = ", ".join(issue.supporting_genes)
        task = (
            f"A verified counterfactual check found a potential conflict for Cell "
            f"{issue.cell_index + 1}. It currently has {issue.current_label}, while "
            f"{evidence} support {issue.alternative_label}. Re-evaluate only Cell "
            f"{issue.cell_index + 1} and its label-swap partner Cell "
            f"{issue.swap_index + 1}. Preserve every other cell assignment. "
            f"The critic's reason was: {issue.reason}"
        )
    else:
        task = (
            "The previous response failed a deterministic structural check "
            f"({structural_reason}). Repair the response while solving the original "
            "problem, using every candidate exactly once."
        )
    return (
        "\n\nSingle refinement step:\n"
        f"{task}\n\nPrevious response:\n{initial_text}\n\n"
        "Generate a new self-contained biological reasoning process. Verify marker "
        "genes against the original input and output exactly "
        "<reasoning>...</reasoning> followed by "
        "<answer>cell type 1 | cell type 2 | ...</answer>."
    )


def validate_refinement(
    text: str,
    candidates: Sequence[str],
    initial: ParsedRollout,
    issue: Optional[VerifiedIssue],
) -> Tuple[bool, str, ParsedRollout]:
    parsed = parse_rollout(-1, text, candidates)
    if not parsed.valid:
        return False, parsed.invalid_reason or "invalid_refinement", parsed
    if issue is None or not initial.valid:
        return True, "accepted_structural_repair", parsed
    changed = {
        index
        for index, (before, after) in enumerate(
            zip(initial.normalized_assignments, parsed.normalized_assignments)
        )
        if before != after
    }
    allowed = {issue.cell_index, issue.swap_index}
    if not changed:
        return False, "no_assignment_change", parsed
    if not changed.issubset(allowed):
        return False, "changed_unrelated_cell", parsed
    if parsed.normalized_assignments[issue.cell_index] != normalize_label(
        issue.alternative_label
    ):
        return False, "did_not_apply_verified_alternative", parsed
    if parsed.normalized_assignments[issue.swap_index] != normalize_label(
        issue.current_label
    ):
        return False, "did_not_complete_label_swap", parsed
    return True, "accepted_verified_swap", parsed


def decision_to_dict(decision: Optional[CriticDecision]) -> Dict[str, Any]:
    if decision is None:
        return {}
    issue = decision.issue
    return {
        "verdict": decision.verdict,
        "valid": decision.valid,
        "status": decision.status,
        "issue": None
        if issue is None
        else {
            "cell": issue.cell_index + 1,
            "swap_partner": issue.swap_index + 1,
            "current_label": issue.current_label,
            "alternative_label": issue.alternative_label,
            "supporting_genes": list(issue.supporting_genes),
            "reason": issue.reason,
        },
    }

