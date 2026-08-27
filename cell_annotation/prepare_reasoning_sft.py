#!/usr/bin/env python3
"""Prepare CellPuzzles' o1-distilled split for explicit-reasoning SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pyarrow.parquet as pq

try:
    from cell_annotation.common import normalize_label, write_json, write_jsonl
except ImportError:
    from common import normalize_label, write_json, write_jsonl


HF_BASE = "https://huggingface.co/datasets/ncbi/CellPuzzles/resolve/main/data"
SOURCE_URLS = {
    "reasoning": f"{HF_BASE}/reasoning-00000-of-00001.parquet",
    "train": f"{HF_BASE}/train-00000-of-00001.parquet",
    "test": f"{HF_BASE}/test-00000-of-00001.parquet",
}
EXPECTED_COUNTS = {"reasoning": 3912, "train": 6912, "test": 1095}
STRICT_REASONING_RE = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*"
    r"<answer>(?P<answer>.*?)</answer>\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)
ORIGINAL_REASONING_INSTRUCTION = (
    "Include your detailed reasoning within <think> and </think> tags, and "
    "provide your final answer within <answer> and </answer> tags."
)
NONTHINKING_REASONING_INSTRUCTION = (
    "The model operates in non-thinking mode. Provide explicit, detailed "
    "biological reasoning within <reasoning> and </reasoning> tags. Ground "
    "each assignment in marker genes, cell and donor context, and the global "
    "one-to-one matching constraint. Then provide your final answer within "
    "<answer> and </answer> tags."
)
EXPLICIT_REASONING_INSTRUCTION = (
    "Provide explicit, detailed biological reasoning within <reasoning> and "
    "</reasoning> tags. Ground each assignment in marker genes, cell and "
    "donor context, and the global one-to-one matching constraint. Then "
    "provide your final answer within <answer> and </answer> tags."
)


def download(url: str, output: Path, force: bool = False) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0 and not force:
        print(f"[reuse] {output}")
        return
    partial = output.with_suffix(output.suffix + ".part")
    print(f"[download] {url} -> {output}")
    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    partial.replace(output)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_parquet(path: Path, split: str) -> List[Dict[str, str]]:
    rows = pq.read_table(path).to_pylist()
    if len(rows) != EXPECTED_COUNTS[split]:
        raise ValueError(
            f"{split}: expected {EXPECTED_COUNTS[split]} rows, got {len(rows)}"
        )
    required = {"system_msg", "user_msg", "assistant_msg"}
    for index, row in enumerate(rows):
        if not required.issubset(row):
            raise ValueError(f"{split} row {index}: missing fields {required - set(row)}")
        if not all(
            isinstance(row[field], str) and row[field].strip() for field in required
        ):
            raise ValueError(f"{split} row {index}: empty required field")
    return rows


def normalized_text(value: str) -> str:
    return " ".join(value.strip().split()).casefold()


def validate_and_normalize(
    reasoning: Sequence[Dict[str, str]],
    train: Sequence[Dict[str, str]],
    test: Sequence[Dict[str, str]],
    response_mode: str = "thinking",
) -> List[Dict[str, Any]]:
    if response_mode not in {
        "thinking",
        "nonthinking_reasoning",
        "explicit_reasoning",
    }:
        raise ValueError(f"unsupported response mode: {response_mode}")
    train_by_prompt = {
        normalized_text(row["user_msg"]): row["assistant_msg"] for row in train
    }
    test_prompts = {normalized_text(row["user_msg"]) for row in test}
    seen_prompts = set()
    normalized: List[Dict[str, Any]] = []

    for index, row in enumerate(reasoning):
        prompt_key = normalized_text(row["user_msg"])
        if prompt_key in seen_prompts:
            raise ValueError(f"reasoning row {index}: duplicate user prompt")
        seen_prompts.add(prompt_key)
        if prompt_key not in train_by_prompt:
            raise ValueError(f"reasoning row {index}: prompt is not in official train")
        if prompt_key in test_prompts:
            raise ValueError(f"reasoning row {index}: prompt overlaps official test")

        source_assistant = row["assistant_msg"].strip()
        match = STRICT_REASONING_RE.fullmatch(source_assistant)
        if not match or not match.group("think").strip():
            raise ValueError(
                f"reasoning row {index}: expected non-empty <think> and <answer>"
            )
        answer = match.group("answer").strip()
        reasoning_text = match.group("think").strip()
        gold = train_by_prompt[prompt_key].strip()
        if normalize_label(answer) != normalize_label(gold):
            raise ValueError(
                f"reasoning row {index}: distilled answer does not match train gold"
            )

        system_msg = row["system_msg"].strip()
        if response_mode in {"nonthinking_reasoning", "explicit_reasoning"}:
            if ORIGINAL_REASONING_INSTRUCTION not in system_msg:
                raise ValueError(
                    f"reasoning row {index}: missing expected output instruction"
                )
            replacement_instruction = (
                NONTHINKING_REASONING_INSTRUCTION
                if response_mode == "nonthinking_reasoning"
                else EXPLICIT_REASONING_INSTRUCTION
            )
            system_msg = system_msg.replace(
                ORIGINAL_REASONING_INSTRUCTION, replacement_instruction
            )
            assistant = (
                f"<reasoning>{reasoning_text}</reasoning>\n"
                f"<answer>{answer}</answer>"
            )
        else:
            assistant = source_assistant

        normalized.append(
            {
                "id": f"reasoning_{index:06d}",
                "split": "reasoning",
                "system_msg": system_msg,
                "user_msg": row["user_msg"].strip(),
                "assistant_msg": assistant,
                "answer": answer,
                "source": "ncbi/CellPuzzles:reasoning",
            }
        )
    return normalized


def split_records(
    records: Sequence[Dict[str, Any]], dev_ratio: float, seed: int
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    dev_size = max(1, round(len(records) * dev_ratio))
    dev_indices = set(indices[:dev_size])
    train = [row for index, row in enumerate(records) if index not in dev_indices]
    dev = [row for index, row in enumerate(records) if index in dev_indices]
    return train, dev


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--source-raw-dir",
        type=Path,
        help=(
            "Reuse already-downloaded official parquet files from this directory. "
            "When omitted, files are downloaded into DATA_ROOT/raw."
        ),
    )
    parser.add_argument("--dev-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--response-mode",
        choices=("thinking", "nonthinking_reasoning", "explicit_reasoning"),
        default="thinking",
        help=(
            "thinking preserves the source <think> trace; "
            "nonthinking_reasoning converts it to an explicit <reasoning> "
            "response for Qwen3 enable_thinking=False; explicit_reasoning "
            "uses the same visible format with a standard chat template that "
            "has no thinking toggle (for example Llama 3.1)"
        ),
    )
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.dev_ratio < 0.5:
        raise ValueError("--dev-ratio must be between 0 and 0.5")

    raw_dir = args.source_raw_dir or args.data_root / "raw"
    processed_dir = args.data_root / "processed"
    paths = {
        split: raw_dir / f"cellpuzzles_{split}.parquet"
        for split in ("reasoning", "train", "test")
    }
    if args.source_raw_dir is not None:
        for split, path in paths.items():
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(
                    f"{split}: missing reusable official parquet file: {path}"
                )
            print(f"[reuse source] {path}")
    else:
        for split, path in paths.items():
            download(SOURCE_URLS[split], path, force=args.force_download)

    reasoning = load_parquet(paths["reasoning"], "reasoning")
    train = load_parquet(paths["train"], "train")
    test = load_parquet(paths["test"], "test")
    records = validate_and_normalize(
        reasoning, train, test, response_mode=args.response_mode
    )
    train_records, dev_records = split_records(
        records, dev_ratio=args.dev_ratio, seed=args.seed
    )

    train_path = processed_dir / "sft_reasoning_train.jsonl"
    dev_path = processed_dir / "sft_reasoning_dev.jsonl"
    write_jsonl(train_path, train_records)
    write_jsonl(dev_path, dev_records)
    manifest = {
        "source": "ncbi/CellPuzzles",
        "source_split": "reasoning",
        "description": "Expert-like reasoning traces distilled from o1.",
        "source_urls": SOURCE_URLS,
        "source_raw_dir": str(raw_dir),
        "sha256": {split: sha256_file(path) for split, path in paths.items()},
        "counts": {
            "reasoning": len(records),
            "train": len(train_records),
            "dev": len(dev_records),
            "official_test_overlap": 0,
            "answer_mismatch_with_official_train": 0,
        },
        "split_seed": args.seed,
        "dev_ratio": args.dev_ratio,
        "target_format": (
            "<reasoning>...</reasoning><answer>...</answer>"
            if args.response_mode in {"nonthinking_reasoning", "explicit_reasoning"}
            else "<think>...</think><answer>...</answer>"
        ),
        "chat_template_mode": (
            "enable_thinking=False"
            if args.response_mode == "nonthinking_reasoning"
            else (
                "standard_chat_template"
                if args.response_mode == "explicit_reasoning"
                else "enable_thinking=True"
            )
        ),
        "response_mode": args.response_mode,
    }
    write_json(processed_dir / "reasoning_sft_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[done] reasoning SFT train: {train_path}")
    print(f"[done] reasoning SFT dev:   {dev_path}")


if __name__ == "__main__":
    main()
