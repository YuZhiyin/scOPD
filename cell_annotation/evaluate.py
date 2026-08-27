#!/usr/bin/env python3
"""Strict and diagnostic evaluation for generated cell annotations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

try:
    from cell_annotation.common import read_records, score_prediction, write_json
except ImportError:
    from common import read_records, score_prediction, write_json


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals = defaultdict(float)
    by_n: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    errors: List[Dict[str, Any]] = []
    total_cells = 0
    correct_cells = 0

    for index, row in enumerate(rows):
        if "prediction" not in row:
            raise ValueError(f"row {index} has no 'prediction' field")
        ground_truth = row.get("answer") or row.get("assistant_msg")
        if not ground_truth:
            raise ValueError(f"row {index} has no answer/assistant_msg field")
        metrics = score_prediction(row["prediction"], ground_truth)
        n = metrics["num_cells"]
        total_cells += n
        correct_cells += metrics["num_correct"]
        totals["batches"] += 1
        by_n[n]["batches"] += 1
        for key in (
            "strict_format",
            "answer_count_valid",
            "unique_valid",
            "candidate_set_valid",
            "valid_output_format",
            "exact_match",
            "strict_exact_match",
            "partial_accuracy",
        ):
            totals[key] += float(metrics[key])
            by_n[n][key] += float(metrics[key])
        if not metrics["strict_exact_match"] and len(errors) < 50:
            errors.append(
                {
                    "id": row.get("id", index),
                    "gold": ground_truth,
                    "prediction": row["prediction"],
                    "metrics": metrics,
                }
            )

    batches = int(totals["batches"])
    result: Dict[str, Any] = {
        "num_batches": batches,
        "num_cells": total_cells,
        "micro_cell_accuracy": correct_cells / total_cells if total_cells else 0.0,
    }
    for key in (
        "partial_accuracy",
        "strict_format",
        "answer_count_valid",
        "unique_valid",
        "candidate_set_valid",
        "valid_output_format",
        "exact_match",
        "strict_exact_match",
    ):
        result[f"mean_{key}"] = totals[key] / batches if batches else 0.0

    result["by_num_cells"] = {}
    for n, bucket in sorted(by_n.items()):
        count = bucket["batches"]
        result["by_num_cells"][str(n)] = {
            "num_batches": int(count),
            **{
                f"mean_{key}": bucket[key] / count
                for key in (
                    "partial_accuracy",
                    "strict_format",
                    "candidate_set_valid",
                    "valid_output_format",
                    "exact_match",
                    "strict_exact_match",
                )
            },
        }
    result["first_50_errors"] = errors
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(read_records(args.predictions))
    printable = {key: value for key, value in summary.items() if key != "first_50_errors"}
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    if args.output:
        write_json(args.output, summary)


if __name__ == "__main__":
    main()
