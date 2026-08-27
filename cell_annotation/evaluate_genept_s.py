#!/usr/bin/env python3
"""Exact cosine kNN evaluation for cached GenePT-s embeddings."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from cell_annotation.genept import read_jsonl, write_json_atomic, write_jsonl


def majority_vote(labels: Sequence[str], similarities: Sequence[float]) -> str:
    """Vote by count, then summed/max similarity, then lexical order."""
    if not labels or len(labels) != len(similarities):
        raise ValueError("labels and similarities must be non-empty and aligned")
    counts = Counter(labels)
    sums: Dict[str, float] = defaultdict(float)
    maxima: Dict[str, float] = defaultdict(lambda: -float("inf"))
    for label, similarity in zip(labels, similarities):
        sums[label] += float(similarity)
        maxima[label] = max(maxima[label], float(similarity))
    return min(
        counts,
        key=lambda label: (-counts[label], -sums[label], -maxima[label], label),
    )


def exact_cosine_knn(
    reference: np.ndarray,
    queries: np.ndarray,
    k: int,
    device: str,
    query_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if reference.ndim != 2 or queries.ndim != 2 or reference.shape[1] != queries.shape[1]:
        raise ValueError("reference and query arrays must be aligned 2-D matrices")
    if not 1 <= k <= len(reference):
        raise ValueError(f"k must be in [1, {len(reference)}], got {k}")
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")
    if device == "cpu":
        reference_cpu = np.asarray(reference, dtype=np.float32)
        queries_cpu = np.asarray(queries, dtype=np.float32)
        reference_norms = np.linalg.norm(reference_cpu, axis=1, keepdims=True)
        query_norms = np.linalg.norm(queries_cpu, axis=1, keepdims=True)
        if np.any(reference_norms == 0) or np.any(query_norms == 0):
            raise ValueError("cosine kNN received a zero-norm embedding")
        reference_cpu = reference_cpu / reference_norms
        queries_cpu = queries_cpu / query_norms
        all_scores: List[np.ndarray] = []
        all_indices: List[np.ndarray] = []
        for start in range(0, len(queries_cpu), query_batch_size):
            scores = queries_cpu[start : start + query_batch_size] @ reference_cpu.T
            if k == len(reference_cpu):
                selected = np.broadcast_to(
                    np.arange(len(reference_cpu), dtype=np.int64), scores.shape
                ).copy()
            else:
                selected = np.argpartition(scores, -k, axis=1)[:, -k:]
            selected_scores = np.take_along_axis(scores, selected, axis=1)
            order = np.argsort(-selected_scores, axis=1, kind="stable")
            all_indices.append(np.take_along_axis(selected, order, axis=1))
            all_scores.append(np.take_along_axis(selected_scores, order, axis=1))
        return np.concatenate(all_scores), np.concatenate(all_indices)

    import torch
    import torch.nn.functional as functional

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if resolved_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    reference_tensor = functional.normalize(
        torch.as_tensor(np.asarray(reference), dtype=torch.float32, device=resolved_device),
        p=2,
        dim=1,
    )
    all_scores: List[np.ndarray] = []
    all_indices: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(queries), query_batch_size):
            query = torch.as_tensor(
                np.asarray(queries[start : start + query_batch_size]),
                dtype=torch.float32,
                device=resolved_device,
            )
            query = functional.normalize(query, p=2, dim=1)
            scores, indices = torch.topk(query @ reference_tensor.T, k=k, dim=1)
            all_scores.append(scores.cpu().numpy())
            all_indices.append(indices.cpu().numpy())
    return np.concatenate(all_scores), np.concatenate(all_indices)


def classification_metrics(rows: Sequence[Dict[str, Any]], train_labels: set[str]) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    gold = [row["label_normalized"] for row in rows]
    predicted = [row["prediction_normalized"] for row in rows]
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, predicted, average="macro", zero_division=0
    )

    def bucket(selected: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        count = len(selected)
        correct = sum(
            row["label_normalized"] == row["prediction_normalized"] for row in selected
        )
        return {
            "num_cells": count,
            "num_correct": correct,
            "accuracy": correct / count if count else None,
            "num_gold_labels": len({row["label_normalized"] for row in selected}),
        }

    seen = [row for row in rows if row["label_normalized"] in train_labels]
    unseen = [row for row in rows if row["label_normalized"] not in train_labels]
    return {
        "num_cells": len(rows),
        "num_gold_labels": len(set(gold)),
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "train_label_cell_coverage": len(seen) / len(rows) if rows else 0.0,
        "train_seen_labels": bucket(seen),
        "train_unseen_labels": bucket(unseen),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--query-split", action="append")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-batch-size", type=int, default=256)
    args = parser.parse_args()

    records = read_jsonl(args.artifact_dir / "records.jsonl")
    # The shared filesystem does not support mmap reliably, so load the roughly
    # 468 MiB float32 matrix into RAM on the offline evaluation worker.
    embeddings = np.load(args.artifact_dir / "embeddings.npy")
    completed = np.load(args.artifact_dir / "embedding_completed.npy")
    if not bool(np.asarray(completed).all()):
        raise RuntimeError(
            f"embedding cache is incomplete: {int(np.asarray(completed).sum())}/{len(completed)}"
        )
    if not np.isfinite(np.asarray(embeddings)).all():
        raise ValueError("embedding cache contains NaN or infinity")

    train_rows = [row for row in records if row["split"] == args.train_split]
    if not train_rows:
        raise ValueError(f"no records found for train split {args.train_split!r}")
    available_splits = sorted({row["split"] for row in records})
    query_splits = args.query_split or [
        split for split in available_splits if split != args.train_split
    ]
    missing = sorted(set(query_splits) - set(available_splits))
    if missing:
        raise ValueError(f"unknown query splits: {missing}; available={available_splits}")

    reference_indices = np.asarray(
        [row["embedding_index"] for row in train_rows], dtype=np.int64
    )
    reference_embeddings = np.asarray(embeddings[reference_indices])
    reference_labels = [row["label_normalized"] for row in train_rows]
    train_labels = set(reference_labels)
    canonical_label: Dict[str, str] = {}
    for row in train_rows:
        canonical_label.setdefault(row["label_normalized"], row["label"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics: Dict[str, Any] = {
        "method": "GenePT-s global cosine kNN",
        "train_split": args.train_split,
        "num_reference_cells": len(train_rows),
        "num_reference_labels": len(train_labels),
        "k": args.k,
        "splits": {},
    }
    all_predictions: List[Dict[str, Any]] = []
    for split in query_splits:
        query_rows = [row for row in records if row["split"] == split]
        query_indices = np.asarray(
            [row["embedding_index"] for row in query_rows], dtype=np.int64
        )
        similarities, neighbor_indices = exact_cosine_knn(
            reference_embeddings,
            np.asarray(embeddings[query_indices]),
            args.k,
            args.device,
            args.query_batch_size,
        )
        split_predictions: List[Dict[str, Any]] = []
        for row, neighbor_row_indices, neighbor_scores in zip(
            query_rows, neighbor_indices, similarities
        ):
            neighbor_labels = [reference_labels[int(i)] for i in neighbor_row_indices]
            prediction_norm = majority_vote(neighbor_labels, neighbor_scores.tolist())
            prediction = {
                **row,
                "prediction": canonical_label[prediction_norm],
                "prediction_normalized": prediction_norm,
                "train_label_seen": row["label_normalized"] in train_labels,
                "neighbor_labels_normalized": neighbor_labels,
                "neighbor_similarities": [float(value) for value in neighbor_scores],
            }
            split_predictions.append(prediction)
        write_jsonl(args.output_dir / f"predictions_{split}.jsonl", split_predictions)
        metrics["splits"][split] = classification_metrics(split_predictions, train_labels)
        all_predictions.extend(split_predictions)
        print(json.dumps({split: metrics["splits"][split]}, indent=2, ensure_ascii=False))

    unseen_predictions = [
        row for row in all_predictions if row["split"].startswith("unseen_")
    ]
    if unseen_predictions:
        metrics["splits"]["unseen_all"] = classification_metrics(
            unseen_predictions, train_labels
        )
        write_jsonl(args.output_dir / "predictions_unseen_all.jsonl", unseen_predictions)
        print(
            json.dumps(
                {"unseen_all": metrics["splits"]["unseen_all"]},
                indent=2,
                ensure_ascii=False,
            )
        )

    write_json_atomic(args.output_dir / "metrics.json", metrics)
    print(f"Saved GenePT-s evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
