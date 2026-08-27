"""Shared helpers for the GenePT-s CellPuzzles baseline."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    from cell_annotation.common import normalize_label, read_records, split_assignments
    from cell_annotation.infer_chatcell import parse_cellpuzzles_prompt
except ImportError:
    from common import normalize_label, read_records, split_assignments
    from infer_chatcell import parse_cellpuzzles_prompt


GENEPT_S_PREFIX = "A cell with genes ranked by expression: "
ARTIFACT_VERSION = 1


def genept_s_text(genes: Sequence[str]) -> str:
    """Construct the exact ranked-gene sentence used by GenePT-s."""
    cleaned = [str(gene).strip() for gene in genes if str(gene).strip()]
    if not cleaned:
        raise ValueError("GenePT-s requires at least one gene")
    return GENEPT_S_PREFIX + " ".join(cleaned)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_split(path: Path, split_name: str) -> List[Dict[str, Any]]:
    """Flatten CellPuzzles batches into independently labelled cells."""
    flattened: List[Dict[str, Any]] = []
    for batch_index, row in enumerate(read_records(path)):
        _, cells, candidates = parse_cellpuzzles_prompt(row["user_msg"])
        labels = split_assignments(row.get("answer") or row.get("assistant_msg") or "")
        if len(cells) != len(labels):
            raise ValueError(
                f"{path} row {batch_index}: {len(cells)} cells but {len(labels)} labels"
            )
        if Counter(normalize_label(x) for x in candidates) != Counter(
            normalize_label(x) for x in labels
        ):
            raise ValueError(
                f"{path} row {batch_index}: candidates are not a permutation of labels"
            )
        for cell_index, (genes, label) in enumerate(zip(cells, labels)):
            text = genept_s_text(genes)
            flattened.append(
                {
                    "split": split_name,
                    "source_split": row.get("split"),
                    "batch_id": row.get("id", f"{split_name}_{batch_index:06d}"),
                    "batch_index": batch_index,
                    "cell_index": cell_index,
                    "num_genes": len(genes),
                    "label": label,
                    "label_normalized": normalize_label(label),
                    "candidate_types": candidates,
                    "text_sha256": text_sha256(text),
                    "text": text,
                }
            )
    return flattened


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json_atomic(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def parse_named_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"empty split name in {value!r}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return name, path
