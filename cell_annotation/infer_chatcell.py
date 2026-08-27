#!/usr/bin/env python3
"""Closed-set CellPuzzles evaluation for the ChatCell T5 model.

ChatCell annotates one cell at a time. For the primary cell-level baseline, each
cell independently chooses the highest-likelihood label from the candidate cell
types in its CellPuzzles batch. An optional Hungarian mode is retained for the
separate batch-assignment baseline.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    from cell_annotation.common import read_records, write_json
except ImportError:
    from common import read_records, write_json


CELL_LINE_RE = re.compile(r"^Cell\s+(\d+):\s*(.+?)\s*$", flags=re.MULTILINE)
CANDIDATE_MARKER = "Match the cells above to one of the following cell types:"


def parse_cellpuzzles_prompt(user_msg: str) -> Tuple[str, List[List[str]], List[str]]:
    """Extract donor context, ranked genes, and candidate labels."""
    if CANDIDATE_MARKER not in user_msg:
        raise ValueError(f"missing candidate marker: {CANDIDATE_MARKER!r}")
    cell_section, candidate_section = user_msg.split(CANDIDATE_MARKER, 1)
    matches = list(CELL_LINE_RE.finditer(cell_section))
    if not matches:
        raise ValueError("no 'Cell N:' lines found")

    expected_numbers = list(range(1, len(matches) + 1))
    actual_numbers = [int(match.group(1)) for match in matches]
    if actual_numbers != expected_numbers:
        raise ValueError(
            f"cell numbering must be consecutive; got {actual_numbers}"
        )

    first_cell_start = matches[0].start()
    context = cell_section[:first_cell_start].strip()
    genes = [
        [gene.strip() for gene in match.group(2).split(",") if gene.strip()]
        for match in matches
    ]
    candidates = [
        line.strip() for line in candidate_section.splitlines() if line.strip()
    ]
    if len(genes) != len(candidates):
        raise ValueError(
            f"number of cells ({len(genes)}) != candidates ({len(candidates)})"
        )
    if len(set(label.casefold() for label in candidates)) != len(candidates):
        raise ValueError("candidate labels must be unique")
    return context, genes, candidates


def build_chatcell_prompt(
    genes: Sequence[str],
    context: str,
    candidates: Sequence[str],
    include_context: bool,
    include_candidates: bool,
) -> str:
    """Use an annotation instruction represented in ChatCell's official data."""
    parts: List[str] = []
    if include_context and context:
        parts.append(context)
    parts.append(
        "Provide the likely cell type based on these "
        f"{len(genes)} genes with high expression levels. {' '.join(genes)}"
    )
    if include_candidates:
        parts.append(
            "Choose from the following candidate cell types: "
            + "; ".join(candidates)
            + "."
        )
    parts.append("The cell type best represented by these genes is: ")
    return " ".join(parts)


def chatcell_target(label: str) -> str:
    """Match the capitalized target style in ChatCell's official instructions."""
    return label[:1].upper() + label[1:]


def conditional_log_likelihoods(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    candidates: Sequence[str],
    batch_size: int,
    max_input_length: int,
    max_target_length: int,
    length_normalize: bool,
) -> np.ndarray:
    """Return a [num_cells, num_candidates] conditional score matrix.

    The encoder is evaluated once per cell; its hidden states are then reused
    for all candidate decoder targets.
    """
    import torch
    import torch.nn.functional as functional

    device = next(model.parameters()).device
    encoded = tokenizer(
        list(prompts),
        padding=True,
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        encoder_hidden = model.get_encoder()(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            return_dict=True,
        ).last_hidden_state

    candidate_pairs: List[Tuple[int, int, str]] = []
    for cell_index in range(len(prompts)):
        for candidate_index, candidate in enumerate(candidates):
            candidate_pairs.append(
                (cell_index, candidate_index, chatcell_target(candidate))
            )

    scores = np.full((len(prompts), len(candidates)), -np.inf, dtype=np.float64)
    for start in range(0, len(candidate_pairs), batch_size):
        batch = candidate_pairs[start : start + batch_size]
        cell_indices = torch.tensor(
            [item[0] for item in batch], dtype=torch.long, device=device
        )
        batch_targets = [item[2] for item in batch]
        targets = tokenizer(
            text_target=batch_targets,
            padding=True,
            truncation=True,
            max_length=max_target_length,
            return_tensors="pt",
        )
        target_ids = targets["input_ids"].to(device)
        labels = target_ids.masked_fill(
            target_ids.eq(tokenizer.pad_token_id), -100
        )
        with torch.inference_mode():
            logits = model(
                encoder_outputs=(
                    encoder_hidden.index_select(0, cell_indices),
                ),
                attention_mask=encoded["attention_mask"].index_select(
                    0, cell_indices
                ),
                labels=labels,
            ).logits
        token_losses = functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(labels.shape)
        valid_tokens = labels.ne(-100)
        sequence_log_probs = -(token_losses * valid_tokens).sum(dim=1)
        if length_normalize:
            sequence_log_probs = sequence_log_probs / valid_tokens.sum(
                dim=1
            ).clamp_min(1)

        for item, value in zip(batch, sequence_log_probs.cpu().tolist()):
            cell_index, candidate_index = item[:2]
            scores[cell_index, candidate_index] = float(value)
    return scores


def solve_assignment(
    score_matrix: np.ndarray, candidates: Sequence[str]
) -> Tuple[List[str], List[int]]:
    from scipy.optimize import linear_sum_assignment

    if score_matrix.shape != (len(candidates), len(candidates)):
        raise ValueError(
            f"assignment matrix must be square, got {score_matrix.shape}"
        )
    row_indices, column_indices = linear_sum_assignment(-score_matrix)
    assignment = [-1] * len(candidates)
    for row_index, column_index in zip(row_indices, column_indices):
        assignment[int(row_index)] = int(column_index)
    if any(index < 0 for index in assignment):
        raise RuntimeError("Hungarian assignment did not cover every cell")
    return [candidates[index] for index in assignment], assignment


def solve_independent(
    score_matrix: np.ndarray, candidates: Sequence[str]
) -> Tuple[List[str], List[int]]:
    """Select the best candidate for every cell without a uniqueness constraint."""
    if score_matrix.ndim != 2 or score_matrix.shape[1] != len(candidates):
        raise ValueError(
            "score matrix must have one column per candidate; "
            f"got {score_matrix.shape} for {len(candidates)} candidates"
        )
    candidate_indices = np.argmax(score_matrix, axis=1).astype(int).tolist()
    return [candidates[index] for index in candidate_indices], candidate_indices


def select_predictions(
    score_matrix: np.ndarray,
    candidates: Sequence[str],
    assignment_mode: str,
) -> Tuple[List[str], List[int]]:
    if assignment_mode == "independent":
        return solve_independent(score_matrix, candidates)
    if assignment_mode == "hungarian":
        return solve_assignment(score_matrix, candidates)
    raise ValueError(f"unsupported assignment mode: {assignment_mode!r}")


def resolve_dtype(torch_module: Any, name: str, device: str) -> Any:
    if name == "auto":
        return (
            torch_module.bfloat16
            if device == "cuda" and torch_module.cuda.is_bf16_supported()
            else torch_module.float32
        )
    return getattr(torch_module, name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=32)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--include-context", action="store_true")
    parser.add_argument("--include-candidates", action="store_true")
    parser.add_argument(
        "--assignment-mode",
        choices=("independent", "hungarian"),
        default="independent",
        help=(
            "independent: classify each cell separately (cell-level baseline); "
            "hungarian: enforce a one-to-one batch assignment"
        ),
    )
    parser.add_argument("--sum-log-probs", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device == "auto":
        device = "cpu"
    dtype = resolve_dtype(torch, args.dtype, device)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    records = read_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    results: List[Dict[str, Any]] = []
    for record_index, row in enumerate(records):
        context, cell_genes, candidates = parse_cellpuzzles_prompt(row["user_msg"])
        prompts = [
            build_chatcell_prompt(
                genes,
                context=context,
                candidates=candidates,
                include_context=args.include_context,
                include_candidates=args.include_candidates,
            )
            for genes in cell_genes
        ]
        score_matrix = conditional_log_likelihoods(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            candidates=candidates,
            batch_size=args.batch_size,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
            length_normalize=not args.sum_log_probs,
        )
        labels, candidate_indices = select_predictions(
            score_matrix, candidates, args.assignment_mode
        )
        updated = dict(row)
        updated["prediction"] = f"<answer>{' | '.join(labels)}</answer>"
        updated["chatcell_assignment_mode"] = args.assignment_mode
        updated["chatcell_candidate_indices"] = candidate_indices
        # Keep these legacy field names so existing result-analysis notebooks
        # can read new independent runs as well as prior Hungarian runs.
        updated["chatcell_assignment"] = candidate_indices
        updated["chatcell_assignment_scores"] = [
            round(float(score_matrix[cell_index, candidate_index]), 6)
            for cell_index, candidate_index in enumerate(candidate_indices)
        ]
        updated["chatcell_score_matrix"] = score_matrix.round(6).tolist()
        results.append(updated)
        if (record_index + 1) % 25 == 0 or record_index + 1 == len(records):
            print(
                f"[{record_index + 1}/{len(records)}] "
                f"processed {args.input.name}",
                flush=True,
            )

    write_json(args.output, results)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "rows": len(results),
                "device": device,
                "dtype": str(dtype),
                "include_context": args.include_context,
                "include_candidates": args.include_candidates,
                "assignment_mode": args.assignment_mode,
                "length_normalize": not args.sum_log_probs,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
