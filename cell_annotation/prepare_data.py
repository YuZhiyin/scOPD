#!/usr/bin/env python3
"""Download and normalize CellPuzzles data for inference, SFT, and SDPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from cell_annotation.common import read_records, write_json, write_jsonl
except ImportError:
    from common import read_records, write_json, write_jsonl


HF_BASE = "https://huggingface.co/datasets/ncbi/CellPuzzles/resolve/main/data"
OFFICIAL_FILES = {
    "train": f"{HF_BASE}/train-00000-of-00001.parquet",
    "test": f"{HF_BASE}/test-00000-of-00001.parquet",
}
EXPECTED_COUNTS = {"train": 6912, "test": 1095}
CELL_LINE_RE = re.compile(r"^Cell\s+\d+\s*:\s*(.+)$", flags=re.MULTILINE)
ORIGINAL_REASONING_INSTRUCTION = (
    "Include your detailed reasoning within <think> and </think> tags, and "
    "provide your final answer within <answer> and </answer> tags."
)
NONTHINKING_INSTRUCTION = (
    "The model is evaluated in non-thinking mode, so do not provide reasoning. "
    "Provide only the final answer within <answer> and </answer> tags."
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


def normalize_row(
    row: Dict[str, Any], split: str, index: int, nonthinking_prompt: bool
) -> Dict[str, Any]:
    answer = row.get("answer") or row.get("assistant_msg")
    system = row.get("system_msg")
    user = row.get("user_msg")
    if not all(isinstance(value, str) and value.strip() for value in (system, user, answer)):
        raise ValueError(f"invalid {split} row {index}: missing system/user/answer")
    normalized_system = system.strip()
    if nonthinking_prompt:
        if ORIGINAL_REASONING_INSTRUCTION in normalized_system:
            normalized_system = normalized_system.replace(
                ORIGINAL_REASONING_INSTRUCTION, NONTHINKING_INSTRUCTION
            )
        else:
            normalized_system = f"{normalized_system}\n\n{NONTHINKING_INSTRUCTION}"
    return {
        "id": f"{split}_{index:06d}",
        "split": split,
        "system_msg": normalized_system,
        "user_msg": user.strip(),
        "answer": answer.strip(),
    }


def load_official(
    path: Path, split: str, nonthinking_prompt: bool
) -> List[Dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    expected = EXPECTED_COUNTS[split]
    if len(rows) != expected:
        raise ValueError(f"{split}: expected {expected} rows, got {len(rows)}")
    return [
        normalize_row(row, split, index, nonthinking_prompt)
        for index, row in enumerate(rows)
    ]


def load_unseen(
    unseen_dir: Path, nonthinking_prompt: bool
) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(unseen_dir.glob("*_test.json")):
        stem = path.stem
        split = f"unseen_{stem[:-5] if stem.endswith('_test') else stem}"
        result[split] = [
            normalize_row(row, split, index, nonthinking_prompt)
            for index, row in enumerate(read_records(path))
        ]
    if not result:
        raise FileNotFoundError(f"no *_test.json files found in {unseen_dir}")
    return result


def ranked_gene_signatures(user_msg: str) -> List[str]:
    """Return one exact ranked-gene signature per cell in a batch."""
    return [" ".join(cell.split()).casefold() for cell in CELL_LINE_RE.findall(user_msg)]


def to_verl_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "data_source": "cell_annotation",
        "prompt": [
            {"role": "system", "content": record["system_msg"]},
            {"role": "user", "content": record["user_msg"]},
        ],
        "ability": "cell_type_annotation",
        "reward_model": {
            "style": "rule",
            "ground_truth": record["answer"],
        },
        "extra_info": {
            "id": record["id"],
            "split": record["split"],
            "answer": record["answer"],
        },
    }


def write_parquet(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(records)), path, compression="zstd")


def build_sft_split(
    records: Sequence[Dict[str, Any]], dev_ratio: float, seed: int
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    dev_size = max(1, round(len(records) * dev_ratio))
    dev_indices = set(indices[:dev_size])
    train = [record for index, record in enumerate(records) if index not in dev_indices]
    dev = [record for index, record in enumerate(records) if index in dev_indices]
    return train, dev


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--unseen-dir",
        type=Path,
        default=Path("/mnt/shared-storage-user/yuzhiyin/cell-o1/unseen_data"),
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--keep-original-system-prompt",
        action="store_true",
        help="Keep CellPuzzles' reasoning instruction instead of the non-thinking protocol.",
    )
    parser.add_argument("--sft-dev-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.sft_dev_ratio < 0.5:
        raise ValueError("--sft-dev-ratio must be between 0 and 0.5")

    raw_dir = args.output_dir / "raw"
    processed_dir = args.output_dir / "processed"
    raw_paths: Dict[str, Path] = {}
    for split, url in OFFICIAL_FILES.items():
        raw_path = raw_dir / f"cellpuzzles_{split}.parquet"
        download(url, raw_path, force=args.force_download)
        raw_paths[split] = raw_path

    nonthinking_prompt = not args.keep_original_system_prompt
    train = load_official(raw_paths["train"], "train", nonthinking_prompt)
    test = load_official(raw_paths["test"], "test", nonthinking_prompt)
    unseen = load_unseen(args.unseen_dir, nonthinking_prompt)

    train_signatures = {
        signature
        for row in train
        for signature in ranked_gene_signatures(row["user_msg"])
    }
    test_clean = [
        row
        for row in test
        if train_signatures.isdisjoint(ranked_gene_signatures(row["user_msg"]))
    ]

    write_json(processed_dir / "train.json", train)
    write_json(processed_dir / "test.json", test)
    write_json(processed_dir / "test_clean.json", test_clean)
    write_parquet(
        processed_dir / "train.parquet", [to_verl_record(row) for row in train]
    )
    write_parquet(
        processed_dir / "test.parquet", [to_verl_record(row) for row in test]
    )
    write_parquet(
        processed_dir / "test_clean.parquet",
        [to_verl_record(row) for row in test_clean],
    )

    unseen_all: List[Dict[str, Any]] = []
    for split, records in unseen.items():
        unseen_all.extend(records)
        name = split[len("unseen_") :] if split.startswith("unseen_") else split
        write_json(processed_dir / "unseen" / f"{name}.json", records)
        write_parquet(
            processed_dir / "unseen" / f"{name}.parquet",
            [to_verl_record(row) for row in records],
        )
    write_json(processed_dir / "unseen_all.json", unseen_all)
    write_parquet(
        processed_dir / "unseen_all.parquet",
        [to_verl_record(row) for row in unseen_all],
    )

    train_fit, train_dev = build_sft_split(
        train, dev_ratio=args.sft_dev_ratio, seed=args.seed
    )
    write_jsonl(processed_dir / "sft_train.jsonl", train_fit)
    write_jsonl(processed_dir / "sft_dev.jsonl", train_dev)
    write_parquet(
        processed_dir / "train_fit.parquet",
        [to_verl_record(row) for row in train_fit],
    )
    write_parquet(
        processed_dir / "train_dev.parquet",
        [to_verl_record(row) for row in train_dev],
    )

    manifest = {
        "source": "ncbi/CellPuzzles",
        "source_urls": OFFICIAL_FILES,
        "sha256": {split: sha256_file(path) for split, path in raw_paths.items()},
        "counts": {
            "train": len(train),
            "test": len(test),
            "test_clean_no_cell_level_ranked_gene_overlap": len(test_clean),
            "unseen_total": len(unseen_all),
            **{split: len(records) for split, records in unseen.items()},
            "train_fit": len(train_fit),
            "train_dev": len(train_dev),
        },
        "sft_split_seed": args.seed,
        "sft_dev_ratio": args.sft_dev_ratio,
        "prompt_mode": "nonthinking" if nonthinking_prompt else "original",
        "notes": [
            "SFT and SDPO fit/dev are derived only from the official CellPuzzles train split.",
            "test_clean removes any test batch containing a cell whose exact ranked-gene signature occurs in train.",
            "The local unseen_data files are evaluation-only.",
        ],
    }
    write_json(processed_dir / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], indent=2))
    print(f"[done] processed data: {processed_dir}")


if __name__ == "__main__":
    main()
