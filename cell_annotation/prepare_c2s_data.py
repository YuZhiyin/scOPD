#!/usr/bin/env python3
"""Expand fixed CellPuzzles train/dev batches into C2S single-cell examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from cell_annotation.c2s_common import (
        C2S_PROMPT_VERSION,
        expand_cellpuzzles_batch,
    )
    from cell_annotation.common import read_records, write_json, write_jsonl
except ImportError:
    from c2s_common import C2S_PROMPT_VERSION, expand_cellpuzzles_batch
    from common import read_records, write_json, write_jsonl


def expand_records(
    rows: List[Dict[str, Any]], organism: str
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for row in rows:
        examples.extend(expand_cellpuzzles_batch(row, organism=organism))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-batches", type=Path, required=True)
    parser.add_argument("--dev-batches", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--organism", default="Homo sapiens")
    args = parser.parse_args()

    fit_batches = read_records(args.fit_batches)
    dev_batches = read_records(args.dev_batches)
    fit_batch_ids = {str(row.get("id")) for row in fit_batches}
    dev_batch_ids = {str(row.get("id")) for row in dev_batches}
    overlap = fit_batch_ids & dev_batch_ids
    if overlap:
        raise ValueError(
            f"fit/dev batch overlap detected: {sorted(overlap)[:5]}"
        )

    fit_examples = expand_records(fit_batches, organism=args.organism)
    dev_examples = expand_records(dev_batches, organism=args.organism)
    fit_ids = {row["id"] for row in fit_examples}
    dev_ids = {row["id"] for row in dev_examples}
    if fit_ids & dev_ids:
        raise ValueError("expanded fit/dev example IDs overlap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fit_path = args.output_dir / "fit.jsonl"
    dev_path = args.output_dir / "dev.jsonl"
    write_jsonl(fit_path, fit_examples)
    write_jsonl(dev_path, dev_examples)

    all_examples = fit_examples + dev_examples
    gene_counts: Dict[str, int] = {}
    for row in all_examples:
        key = str(len(row["genes"]))
        gene_counts[key] = gene_counts.get(key, 0) + 1
    manifest = {
        "prompt_version": C2S_PROMPT_VERSION,
        "organism": args.organism,
        "sources": {
            "fit_batches": str(args.fit_batches),
            "dev_batches": str(args.dev_batches),
        },
        "counts": {
            "fit_batches": len(fit_batches),
            "dev_batches": len(dev_batches),
            "fit_cells": len(fit_examples),
            "dev_cells": len(dev_examples),
        },
        "gene_count_distribution": gene_counts,
        "notes": [
            "The existing CellPuzzles batch-level fit/dev split is preserved.",
            "Every expanded example contains one cell and its source batch candidates.",
            "No test, test_clean, or unseen record is used for fine-tuning.",
        ],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
