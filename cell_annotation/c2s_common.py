"""Shared Cell2Sentence formatting and CellPuzzles expansion helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

try:
    from cell_annotation.common import normalize_label, split_assignments
    from cell_annotation.infer_chatcell import parse_cellpuzzles_prompt
except ImportError:
    from common import normalize_label, split_assignments
    from infer_chatcell import parse_cellpuzzles_prompt


C2S_PROMPT_VERSION = "c2s_cellpuzzles_candidates_v1"


def build_c2s_prompt(
    genes: Sequence[str],
    candidates: Sequence[str],
    organism: str = "Homo sapiens",
) -> str:
    """Build a deterministic C2S cell-type-prediction prompt.

    The first two lines and final response cue follow the canonical official
    Cell2Sentence prompt. CellPuzzles' candidate range is inserted immediately
    before the response cue so every cell is evaluated as the same closed-set
    classification problem used for the ChatCell cell-level baseline.
    """
    clean_genes = [str(gene).strip() for gene in genes if str(gene).strip()]
    clean_candidates = [
        str(candidate).strip()
        for candidate in candidates
        if str(candidate).strip()
    ]
    if not clean_genes:
        raise ValueError("C2S prompt requires at least one ranked gene")
    if not clean_candidates:
        raise ValueError("C2S prompt requires at least one candidate label")
    if len({normalize_label(label) for label in clean_candidates}) != len(
        clean_candidates
    ):
        raise ValueError("C2S candidate labels must be unique")

    cell_sentence = " ".join(clean_genes)
    candidate_sentence = "; ".join(clean_candidates)
    return (
        f"The following is a list of {len(clean_genes)} gene names ordered by "
        f"descending expression level in a {organism} cell. Your task is to "
        "give the cell type which this cell belongs to based on its gene "
        "expression.\n"
        f"Cell sentence: {cell_sentence}.\n"
        "Choose exactly one label from the following candidate cell types: "
        f"{candidate_sentence}.\n"
        "The cell type corresponding to these genes is:"
    )


def c2s_target(label: str) -> str:
    """Return the official C2S response style with an explicit token boundary."""
    clean_label = " ".join(str(label).strip().split())
    if not clean_label:
        raise ValueError("C2S target label cannot be empty")
    return f" {clean_label}."


def expand_cellpuzzles_batch(
    row: Dict[str, Any],
    organism: str = "Homo sapiens",
) -> List[Dict[str, Any]]:
    """Expand one CellPuzzles batch into independent C2S training examples."""
    batch_id = str(row.get("id", "")).strip()
    if not batch_id:
        raise ValueError("CellPuzzles row has no id")
    _, cells, candidates = parse_cellpuzzles_prompt(row["user_msg"])
    gold = split_assignments(row.get("answer") or row.get("assistant_msg") or "")
    if len(cells) != len(gold):
        raise ValueError(
            f"{batch_id}: number of cells ({len(cells)}) != labels ({len(gold)})"
        )
    candidate_lookup = {normalize_label(label) for label in candidates}

    expanded: List[Dict[str, Any]] = []
    for cell_index, (genes, label) in enumerate(zip(cells, gold)):
        if normalize_label(label) not in candidate_lookup:
            raise ValueError(
                f"{batch_id}: gold label {label!r} is absent from candidates"
            )
        expanded.append(
            {
                "id": f"{batch_id}_c2s_{cell_index + 1:02d}",
                "batch_id": batch_id,
                "cell_index": cell_index,
                "split": row.get("split", "train"),
                "genes": list(genes),
                "candidates": list(candidates),
                "cell_type": label,
                "prompt": build_c2s_prompt(
                    genes=genes,
                    candidates=candidates,
                    organism=organism,
                ),
                "response": c2s_target(label),
            }
        )
    return expanded


def select_independent(
    score_matrix: Any, candidates: Sequence[str]
) -> tuple[List[str], List[int]]:
    """Choose the highest-scoring candidate independently for every cell."""
    import numpy as np

    scores = np.asarray(score_matrix)
    if scores.ndim != 2 or scores.shape[1] != len(candidates):
        raise ValueError(
            "score matrix must have one column per candidate; "
            f"got {scores.shape} for {len(candidates)} candidates"
        )
    candidate_indices = np.argmax(scores, axis=1).astype(int).tolist()
    return [candidates[index] for index in candidate_indices], candidate_indices
