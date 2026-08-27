#!/usr/bin/env python3
"""Migrate CellPuzzles RL data to an explicit ``<reasoning>`` protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from cell_annotation.prepare_data import ORIGINAL_REASONING_INSTRUCTION
from cell_annotation.prepare_reasoning_sft import (
    EXPLICIT_REASONING_INSTRUCTION,
    NONTHINKING_REASONING_INSTRUCTION,
)


EXPECTED_COUNTS = {"train": 6912, "test": 1095}


def _convert_trace(trace: Any) -> Any:
    if not isinstance(trace, str) or not trace.strip():
        return trace
    converted = trace.strip().replace("<think>", "<reasoning>").replace(
        "</think>", "</reasoning>"
    )
    if "<reasoning>" not in converted or "</reasoning>" not in converted:
        raise ValueError("non-empty o1 trace does not contain a reasoning block")
    return converted


def convert_split(
    source: Path,
    output: Path,
    split: str,
    response_mode: str = "nonthinking_reasoning",
) -> dict[str, int]:
    if response_mode not in {"nonthinking_reasoning", "explicit_reasoning"}:
        raise ValueError(f"unsupported response mode: {response_mode}")
    replacement_instruction = (
        NONTHINKING_REASONING_INSTRUCTION
        if response_mode == "nonthinking_reasoning"
        else EXPLICIT_REASONING_INSTRUCTION
    )
    rows = pq.read_table(source).to_pylist()
    if len(rows) != EXPECTED_COUNTS[split]:
        raise ValueError(
            f"{split}: expected {EXPECTED_COUNTS[split]} rows, got {len(rows)}"
        )

    converted_rows: list[dict[str, Any]] = []
    o1_available = 0
    for index, source_row in enumerate(rows):
        row = dict(source_row)
        messages = [dict(message) for message in row.get("prompt") or []]
        system_messages = [m for m in messages if m.get("role") == "system"]
        if len(system_messages) != 1:
            raise ValueError(f"{split} row {index}: expected exactly one system message")
        system = str(system_messages[0].get("content") or "")
        if ORIGINAL_REASONING_INSTRUCTION not in system:
            raise ValueError(
                f"{split} row {index}: source prompt lacks CellPuzzles reasoning instruction"
            )
        system_messages[0]["content"] = system.replace(
            ORIGINAL_REASONING_INSTRUCTION, replacement_instruction
        )
        row["prompt"] = messages

        extra_info = dict(row.get("extra_info") or {})
        if "o1_reasoning" in extra_info:
            extra_info["o1_reasoning"] = _convert_trace(extra_info["o1_reasoning"])
            o1_available += bool(extra_info["o1_reasoning"])
        row["extra_info"] = extra_info
        converted_rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(converted_rows), output, compression="zstd")
    return {"rows": len(converted_rows), "o1_reasoning_available": o1_available}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--response-mode",
        choices=("nonthinking_reasoning", "explicit_reasoning"),
        default="nonthinking_reasoning",
    )
    args = parser.parse_args()

    counts = {
        split: convert_split(
            args.source_dir / f"{split}.parquet",
            args.output_dir / f"{split}.parquet",
            split,
            args.response_mode,
        )
        for split in ("train", "test")
    }
    if counts["train"]["o1_reasoning_available"] != 3912:
        raise ValueError(
            "train: expected 3,912 o1 traces, got "
            f"{counts['train']['o1_reasoning_available']}"
        )
    manifest = {
        "source_dir": str(args.source_dir),
        "response_mode": args.response_mode,
        "chat_template_mode": (
            "enable_thinking=False"
            if args.response_mode == "nonthinking_reasoning"
            else "standard_chat_template"
        ),
        "target_format": "<reasoning>...</reasoning>\\n<answer>...</answer>",
        "counts": counts,
        "notes": [
            "All 6,912 official CellPuzzles train examples are retained.",
            "The 3,912 offline o1 traces are converted from <think> to <reasoning>.",
            "Ground-truth answers and user prompts are unchanged.",
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
