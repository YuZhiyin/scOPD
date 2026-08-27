#!/usr/bin/env python3
"""Attach the 3,912 official CellPuzzles o1 traces to the full RL train set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


EXPECTED_RL_ROWS = 6912
EXPECTED_O1_ROWS = 3912


def normalized_text(value: str) -> str:
    return " ".join(value.strip().split()).casefold()


def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    row = json.loads(line)
                    if not row.get("user_msg") or not row.get("assistant_msg"):
                        raise ValueError(f"{path}:{line_number}: missing prompt or o1 trace")
                    rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-train", type=Path, required=True)
    parser.add_argument("--reasoning-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    rl_rows = pq.read_table(args.rl_train).to_pylist()
    if len(rl_rows) != EXPECTED_RL_ROWS:
        raise ValueError(f"expected {EXPECTED_RL_ROWS} RL rows, got {len(rl_rows)}")
    o1_rows = read_jsonl(args.reasoning_jsonl)
    if len(o1_rows) != EXPECTED_O1_ROWS:
        raise ValueError(f"expected {EXPECTED_O1_ROWS} o1 rows, got {len(o1_rows)}")

    traces_by_prompt: dict[str, str] = {}
    for row in o1_rows:
        key = normalized_text(row["user_msg"])
        if key in traces_by_prompt:
            raise ValueError("duplicate o1 prompt")
        traces_by_prompt[key] = row["assistant_msg"].strip()

    matched = 0
    enriched: list[dict[str, Any]] = []
    for row in rl_rows:
        prompt_messages = row.get("prompt") or []
        user_messages = [m.get("content", "") for m in prompt_messages if m.get("role") == "user"]
        if len(user_messages) != 1:
            raise ValueError("each RL row must contain exactly one user prompt")
        trace = traces_by_prompt.get(normalized_text(user_messages[0]))
        extra_info = dict(row.get("extra_info") or {})
        extra_info["o1_reasoning"] = trace
        row = dict(row)
        row["extra_info"] = extra_info
        enriched.append(row)
        matched += trace is not None

    if matched != EXPECTED_O1_ROWS:
        raise ValueError(f"expected {EXPECTED_O1_ROWS} matched o1 traces, got {matched}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(enriched), args.output, compression="zstd")
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = {
        "source_rl_train": str(args.rl_train),
        "source_reasoning_jsonl": [str(path) for path in args.reasoning_jsonl],
        "rows": len(enriched),
        "o1_reasoning_available": matched,
        "o1_reasoning_unavailable": len(enriched) - matched,
        "match_key": "normalized user prompt",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[done] {args.output}")


if __name__ == "__main__":
    main()
