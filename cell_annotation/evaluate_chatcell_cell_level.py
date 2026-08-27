#!/usr/bin/env python3
"""Cell-level metrics for closed-set ChatCell evaluation.

This evaluator intentionally does not report batch exact-match or candidate
permutation validity. Each cell is an independent classification example whose
label space is the candidate list supplied by its CellPuzzles batch.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

try:
    from cell_annotation.common import (
        normalize_label,
        parse_prediction,
        read_records,
        split_assignments,
        write_json,
    )
    from cell_annotation.infer_chatcell import parse_cellpuzzles_prompt
except ImportError:
    from common import (
        normalize_label,
        parse_prediction,
        read_records,
        split_assignments,
        write_json,
    )
    from infer_chatcell import parse_cellpuzzles_prompt


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _parse_top_k(values: str) -> List[int]:
    top_k = sorted({int(value.strip()) for value in values.split(",") if value.strip()})
    if not top_k or any(value <= 0 for value in top_k):
        raise ValueError("--top-k must contain positive comma-separated integers")
    return top_k


def summarize_cell_level(
    rows: Sequence[Dict[str, Any]], top_k: Sequence[int] = (1, 3, 5)
) -> Dict[str, Any]:
    num_correct = 0
    reciprocal_rank_sum = 0.0
    top_k_correct = {int(k): 0 for k in top_k}
    label_total: Dict[str, int] = defaultdict(int)
    label_correct: Dict[str, int] = defaultdict(int)
    by_candidate_count: Dict[int, Dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    modes = set()
    total_cells = 0

    for row_index, row in enumerate(rows):
        ground_truth = row.get("answer") or row.get("assistant_msg")
        if not ground_truth:
            raise ValueError(f"row {row_index} has no answer/assistant_msg field")
        gold = split_assignments(ground_truth)
        predicted, _, _ = parse_prediction(row.get("prediction", ""))
        _, genes, candidates = parse_cellpuzzles_prompt(row["user_msg"])
        scores = np.asarray(row.get("chatcell_score_matrix"), dtype=np.float64)

        expected_shape = (len(genes), len(candidates))
        if scores.shape != expected_shape:
            raise ValueError(
                f"row {row_index}: score matrix {scores.shape} != {expected_shape}"
            )
        if len(gold) != len(genes) or len(predicted) != len(genes):
            raise ValueError(
                f"row {row_index}: expected {len(genes)} gold/predicted labels, "
                f"got {len(gold)}/{len(predicted)}"
            )

        mode = row.get("chatcell_assignment_mode", "hungarian_legacy")
        modes.add(str(mode))
        candidate_lookup = {
            normalize_label(label): index for index, label in enumerate(candidates)
        }
        bucket = by_candidate_count[len(candidates)]
        bucket["num_batches"] += 1

        for cell_index, (gold_label, predicted_label) in enumerate(
            zip(gold, predicted)
        ):
            gold_norm = normalize_label(gold_label)
            predicted_norm = normalize_label(predicted_label)
            if gold_norm not in candidate_lookup:
                raise ValueError(
                    f"row {row_index}: gold label {gold_label!r} is not a candidate"
                )

            is_correct = predicted_norm == gold_norm
            num_correct += int(is_correct)
            total_cells += 1
            label_total[gold_norm] += 1
            label_correct[gold_norm] += int(is_correct)
            bucket["num_cells"] += 1
            bucket["num_correct"] += int(is_correct)

            # Stable ordering makes exact score ties deterministic and agrees
            # with np.argmax's first-candidate tie breaking.
            ranking = np.argsort(-scores[cell_index], kind="stable")
            gold_index = candidate_lookup[gold_norm]
            rank = int(np.flatnonzero(ranking == gold_index)[0]) + 1
            reciprocal_rank_sum += 1.0 / rank
            bucket["reciprocal_rank_sum"] += 1.0 / rank
            for k in top_k:
                hit = int(rank <= min(k, len(candidates)))
                top_k_correct[int(k)] += hit
                bucket[f"top_{k}_correct"] += hit

    per_cell_type = {
        label: {
            "support": label_total[label],
            "correct": label_correct[label],
            "recall": _safe_divide(label_correct[label], label_total[label]),
        }
        for label in sorted(label_total)
    }
    macro_recall = _safe_divide(
        sum(item["recall"] for item in per_cell_type.values()),
        len(per_cell_type),
    )

    result: Dict[str, Any] = {
        "evaluation_protocol": "closed_set_cell_level",
        "assignment_modes": sorted(modes),
        "num_batches": len(rows),
        "num_cells": total_cells,
        "num_cell_types": len(per_cell_type),
        "num_correct": num_correct,
        "cell_accuracy": _safe_divide(num_correct, total_cells),
        "macro_cell_type_recall": macro_recall,
        "mean_reciprocal_rank": _safe_divide(reciprocal_rank_sum, total_cells),
    }
    for k in top_k:
        result[f"top_{k}_accuracy"] = _safe_divide(
            top_k_correct[int(k)], total_cells
        )

    result["by_num_candidates"] = {}
    for candidate_count, bucket in sorted(by_candidate_count.items()):
        count = int(bucket["num_cells"])
        item: Dict[str, Any] = {
            "num_batches": int(bucket["num_batches"]),
            "num_cells": count,
            "cell_accuracy": _safe_divide(bucket["num_correct"], count),
            "mean_reciprocal_rank": _safe_divide(
                bucket["reciprocal_rank_sum"], count
            ),
        }
        for k in top_k:
            item[f"top_{k}_accuracy"] = _safe_divide(
                bucket[f"top_{k}_correct"], count
            )
        result["by_num_candidates"][str(candidate_count)] = item
    result["per_cell_type"] = per_cell_type
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", default="1,3,5")
    args = parser.parse_args()

    summary = summarize_cell_level(
        read_records(args.predictions), top_k=_parse_top_k(args.top_k)
    )
    printable = {key: value for key, value in summary.items() if key != "per_cell_type"}
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    if args.output:
        write_json(args.output, summary)


if __name__ == "__main__":
    main()
