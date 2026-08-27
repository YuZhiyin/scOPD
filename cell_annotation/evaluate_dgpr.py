#!/usr/bin/env python3
"""Compare DGPR final predictions against their M=8 answer-medoid anchors."""

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


def compare(anchor_rows: List[Dict[str, Any]], final_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(anchor_rows) != len(final_rows):
        raise ValueError("anchor and final prediction files have different lengths")
    totals = Counter()
    rejection_reasons = Counter()
    for anchor, final in zip(anchor_rows, final_rows):
        if anchor.get("id") != final.get("id"):
            raise ValueError("anchor and final prediction ids are misaligned")
        gold = final.get("answer") or final.get("assistant_msg")
        anchor_score = score_prediction(anchor["prediction"], gold)
        final_score = score_prediction(final["prediction"], gold)
        diagnostic = final.get("dgpr", {})
        triggered = bool(diagnostic.get("trigger_refinement", False))
        accepted = bool(diagnostic.get("refinement_accepted", False))
        totals["batches"] += 1
        totals["anchor_correct_cells"] += anchor_score["num_correct"]
        totals["final_correct_cells"] += final_score["num_correct"]
        totals["cells"] += final_score["num_cells"]
        totals["anchor_exact"] += int(anchor_score["strict_exact_match"])
        totals["final_exact"] += int(final_score["strict_exact_match"])
        totals["triggered"] += int(triggered)
        totals["accepted"] += int(accepted)
        totals["changed_answer"] += int(
            anchor_score["predicted_assignments"]
            != final_score["predicted_assignments"]
        )
        if triggered:
            totals["triggered_anchor_correct_cells"] += anchor_score["num_correct"]
            totals["triggered_final_correct_cells"] += final_score["num_correct"]
            totals["triggered_cells"] += final_score["num_cells"]
        if final_score["num_correct"] > anchor_score["num_correct"]:
            totals["cell_improved_batches"] += 1
        elif final_score["num_correct"] < anchor_score["num_correct"]:
            totals["cell_harmed_batches"] += 1
        if final_score["strict_exact_match"] and not anchor_score["strict_exact_match"]:
            totals["batch_corrected"] += 1
        if anchor_score["strict_exact_match"] and not final_score["strict_exact_match"]:
            totals["batch_harmed"] += 1
        rejection_reasons[str(diagnostic.get("refinement_status", "missing"))] += 1

    batches = totals["batches"]
    cells = totals["cells"]
    triggered_cells = totals["triggered_cells"]
    return {
        "num_batches": batches,
        "num_cells": cells,
        "anchor_micro_cell_accuracy": totals["anchor_correct_cells"] / cells if cells else 0.0,
        "final_micro_cell_accuracy": totals["final_correct_cells"] / cells if cells else 0.0,
        "micro_cell_accuracy_delta": (
            totals["final_correct_cells"] - totals["anchor_correct_cells"]
        ) / cells if cells else 0.0,
        "anchor_batch_accuracy": totals["anchor_exact"] / batches if batches else 0.0,
        "final_batch_accuracy": totals["final_exact"] / batches if batches else 0.0,
        "batch_accuracy_delta": (
            totals["final_exact"] - totals["anchor_exact"]
        ) / batches if batches else 0.0,
        "trigger_rate": totals["triggered"] / batches if batches else 0.0,
        "accept_rate": totals["accepted"] / batches if batches else 0.0,
        "accept_rate_given_trigger": totals["accepted"] / totals["triggered"] if totals["triggered"] else 0.0,
        "changed_answer_rate": totals["changed_answer"] / batches if batches else 0.0,
        "triggered_anchor_cell_accuracy": totals["triggered_anchor_correct_cells"] / triggered_cells if triggered_cells else 0.0,
        "triggered_final_cell_accuracy": totals["triggered_final_correct_cells"] / triggered_cells if triggered_cells else 0.0,
        "cell_improved_batches": totals["cell_improved_batches"],
        "cell_harmed_batches": totals["cell_harmed_batches"],
        "batch_corrected": totals["batch_corrected"],
        "batch_harmed": totals["batch_harmed"],
        "refinement_status_counts": dict(sorted(rejection_reasons.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(read_records(args.anchor), read_records(args.final))
    write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

