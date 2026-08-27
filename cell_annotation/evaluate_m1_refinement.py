#!/usr/bin/env python3
"""Measure the net effect of one-sample conditional refinement."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

try:
    from cell_annotation.common import read_records, score_prediction, write_json
except ImportError:
    from common import read_records, score_prediction, write_json  # type: ignore


def compare(initial_rows: List[Dict[str, Any]], final_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(initial_rows) != len(final_rows):
        raise ValueError("initial and final files have different lengths")
    totals = Counter()
    statuses = Counter()
    critic_statuses = Counter()
    for initial, final in zip(initial_rows, final_rows):
        if initial.get("id") != final.get("id"):
            raise ValueError("initial and final ids are misaligned")
        gold = final.get("answer") or final.get("assistant_msg")
        before = score_prediction(initial["prediction"], gold)
        after = score_prediction(final["prediction"], gold)
        diagnostic = final.get("m1_refinement", {})
        triggered = bool(diagnostic.get("trigger_refinement", False))
        accepted = bool(diagnostic.get("refinement_accepted", False))
        totals["batches"] += 1
        totals["cells"] += after["num_cells"]
        totals["initial_correct_cells"] += before["num_correct"]
        totals["final_correct_cells"] += after["num_correct"]
        totals["initial_exact"] += int(before["strict_exact_match"])
        totals["final_exact"] += int(after["strict_exact_match"])
        totals["triggered"] += int(triggered)
        totals["accepted"] += int(accepted)
        totals["changed"] += int(before["predicted_assignments"] != after["predicted_assignments"])
        if after["num_correct"] > before["num_correct"]:
            totals["improved_batches"] += 1
        elif after["num_correct"] < before["num_correct"]:
            totals["harmed_batches"] += 1
        if after["strict_exact_match"] and not before["strict_exact_match"]:
            totals["batch_corrected"] += 1
        if before["strict_exact_match"] and not after["strict_exact_match"]:
            totals["batch_harmed"] += 1
        statuses[str(diagnostic.get("refinement_status", "missing"))] += 1
        critic_statuses[str(diagnostic.get("critic", {}).get("status", "not_run"))] += 1
    batches, cells = totals["batches"], totals["cells"]
    return {
        "num_batches": batches,
        "num_cells": cells,
        "initial_micro_cell_accuracy": totals["initial_correct_cells"] / cells if cells else 0.0,
        "final_micro_cell_accuracy": totals["final_correct_cells"] / cells if cells else 0.0,
        "micro_cell_accuracy_delta": (totals["final_correct_cells"] - totals["initial_correct_cells"]) / cells if cells else 0.0,
        "initial_batch_accuracy": totals["initial_exact"] / batches if batches else 0.0,
        "final_batch_accuracy": totals["final_exact"] / batches if batches else 0.0,
        "batch_accuracy_delta": (totals["final_exact"] - totals["initial_exact"]) / batches if batches else 0.0,
        "trigger_rate": totals["triggered"] / batches if batches else 0.0,
        "accept_rate": totals["accepted"] / batches if batches else 0.0,
        "accept_rate_given_trigger": totals["accepted"] / totals["triggered"] if totals["triggered"] else 0.0,
        "changed_answer_rate": totals["changed"] / batches if batches else 0.0,
        "cell_improved_batches": totals["improved_batches"],
        "cell_harmed_batches": totals["harmed_batches"],
        "batch_corrected": totals["batch_corrected"],
        "batch_harmed": totals["batch_harmed"],
        "critic_status_counts": dict(sorted(critic_statuses.items())),
        "refinement_status_counts": dict(sorted(statuses.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(read_records(args.initial), read_records(args.final))
    write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

