#!/usr/bin/env python3
"""Cell-level metrics and train-label generalization for C2S predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

try:
    from cell_annotation.common import (
        normalize_label,
        parse_prediction,
        read_records,
        split_assignments,
        write_json,
    )
    from cell_annotation.evaluate_chatcell_cell_level import (
        _parse_top_k,
        summarize_cell_level,
    )
except ImportError:
    from common import (
        normalize_label,
        parse_prediction,
        read_records,
        split_assignments,
        write_json,
    )
    from evaluate_chatcell_cell_level import _parse_top_k, summarize_cell_level


def training_label_set(rows: Sequence[Dict[str, Any]]) -> Set[str]:
    return {
        normalize_label(label)
        for row in rows
        for label in split_assignments(
            row.get("answer") or row.get("assistant_msg") or ""
        )
    }


def summarize_c2s(
    rows: Sequence[Dict[str, Any]],
    known_labels: Set[str],
    top_k: Sequence[int] = (1, 3, 5),
) -> Dict[str, Any]:
    adapted_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        if "c2s_score_matrix" not in row:
            raise ValueError(f"row {index} has no c2s_score_matrix")
        adapted = dict(row)
        adapted["chatcell_score_matrix"] = row["c2s_score_matrix"]
        adapted["chatcell_assignment_mode"] = "independent"
        adapted_rows.append(adapted)
    result = summarize_cell_level(adapted_rows, top_k=top_k)
    result["model_family"] = "Cell2Sentence"
    result["evaluation_protocol"] = "closed_set_cell_level_independent"

    buckets: Dict[str, Dict[str, Any]] = {
        "train_seen_labels": {
            "num_cells": 0,
            "num_correct": 0,
            "cell_types": set(),
        },
        "train_unseen_labels": {
            "num_cells": 0,
            "num_correct": 0,
            "cell_types": set(),
        },
    }
    errors: List[Dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        gold = split_assignments(
            row.get("answer") or row.get("assistant_msg") or ""
        )
        predicted, _, _ = parse_prediction(row.get("prediction", ""))
        if len(predicted) != len(gold):
            raise ValueError(
                f"row {row_index}: predicted/gold lengths differ "
                f"({len(predicted)}/{len(gold)})"
            )
        for cell_index, (gold_label, predicted_label) in enumerate(
            zip(gold, predicted)
        ):
            gold_norm = normalize_label(gold_label)
            pred_norm = normalize_label(predicted_label)
            bucket_name = (
                "train_seen_labels"
                if gold_norm in known_labels
                else "train_unseen_labels"
            )
            bucket = buckets[bucket_name]
            bucket["num_cells"] += 1
            bucket["num_correct"] += int(pred_norm == gold_norm)
            bucket["cell_types"].add(gold_norm)
            if pred_norm != gold_norm and len(errors) < 50:
                errors.append(
                    {
                        "id": row.get("id", row_index),
                        "cell_index": cell_index,
                        "gold": gold_label,
                        "prediction": predicted_label,
                        "train_label_status": bucket_name,
                    }
                )

    result["by_train_label_status"] = {}
    for name, bucket in buckets.items():
        count = int(bucket["num_cells"])
        correct = int(bucket["num_correct"])
        result["by_train_label_status"][name] = {
            "num_cells": count,
            "num_cell_types": len(bucket["cell_types"]),
            "num_correct": correct,
            "cell_accuracy": correct / count if count else 0.0,
            "cell_types": sorted(bucket["cell_types"]),
        }
    result["first_50_errors"] = errors
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--train-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", default="1,3,5")
    args = parser.parse_args()

    known_labels = training_label_set(read_records(args.train_reference))
    summary = summarize_c2s(
        read_records(args.predictions),
        known_labels=known_labels,
        top_k=_parse_top_k(args.top_k),
    )
    printable = {
        key: value
        for key, value in summary.items()
        if key not in ("per_cell_type", "first_50_errors")
    }
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    if args.output:
        write_json(args.output, summary)


if __name__ == "__main__":
    main()
