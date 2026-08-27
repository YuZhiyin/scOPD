#!/usr/bin/env python3
"""Prepare deduplicated GenePT-s texts from CellPuzzles JSON files."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from cell_annotation.genept import (
    ARTIFACT_VERSION,
    file_sha256,
    parse_named_path,
    parse_split,
    write_json_atomic,
    write_jsonl,
)


def prepare(inputs: List[str], output_dir: Path) -> Dict[str, Any]:
    named_paths = [parse_named_path(value) for value in inputs]
    split_names = [name for name, _ in named_paths]
    if len(split_names) != len(set(split_names)):
        raise ValueError(f"split names must be unique: {split_names}")

    all_rows: List[Dict[str, Any]] = []
    split_summary: Dict[str, Any] = {}
    input_summary: Dict[str, Any] = {}
    for split_name, path in named_paths:
        rows = parse_split(path, split_name)
        all_rows.extend(rows)
        split_summary[split_name] = {
            "num_cells": len(rows),
            "num_batches": len({row["batch_index"] for row in rows}),
            "num_labels": len({row["label_normalized"] for row in rows}),
            "gene_count_distribution": dict(
                sorted(Counter(row["num_genes"] for row in rows).items())
            ),
        }
        input_summary[split_name] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }

    unique_by_hash: Dict[str, Dict[str, Any]] = {}
    for row in all_rows:
        digest = row["text_sha256"]
        previous = unique_by_hash.get(digest)
        if previous is not None and previous["text"] != row["text"]:
            raise RuntimeError(f"SHA256 collision for {digest}")
        if previous is None:
            unique_by_hash[digest] = {"text_sha256": digest, "text": row["text"]}

    unique_inputs: List[Dict[str, Any]] = []
    hash_to_index: Dict[str, int] = {}
    for index, item in enumerate(unique_by_hash.values()):
        item = {"embedding_index": index, **item}
        unique_inputs.append(item)
        hash_to_index[item["text_sha256"]] = index

    records: List[Dict[str, Any]] = []
    for row in all_rows:
        record = dict(row)
        record.pop("text")
        record["embedding_index"] = hash_to_index[record["text_sha256"]]
        records.append(record)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    unique_path = output_dir / "unique_inputs.jsonl"
    write_jsonl(records_path, records)
    write_jsonl(unique_path, unique_inputs)

    for split_name in split_names:
        split_summary[split_name]["num_unique_inputs"] = len(
            {row["embedding_index"] for row in records if row["split"] == split_name}
        )

    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "GenePT-s",
        "prompt_prefix": "A cell with genes ranked by expression: ",
        "num_cells": len(records),
        "num_unique_inputs": len(unique_inputs),
        "splits": split_summary,
        "inputs": input_summary,
        "records_sha256": file_sha256(records_path),
        "unique_inputs_sha256": file_sha256(unique_path),
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Repeatable NAME=PATH CellPuzzles JSON input",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.input, args.output_dir)
    print(
        f"Prepared {manifest['num_cells']} cells and "
        f"{manifest['num_unique_inputs']} unique GenePT-s texts in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
