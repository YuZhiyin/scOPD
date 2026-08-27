#!/usr/bin/env python3
"""Create resumable OpenAI embeddings for prepared GenePT-s texts."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from cell_annotation.genept import file_sha256, read_jsonl, write_json_atomic


def ordered_embeddings(response: Any, expected_count: int, expected_dim: int) -> np.ndarray:
    data = sorted(response.data, key=lambda item: int(item.index))
    indices = [int(item.index) for item in data]
    if indices != list(range(expected_count)):
        raise ValueError(f"embedding response indices are invalid: {indices}")
    array = np.asarray([item.embedding for item in data], dtype=np.float32)
    if array.shape != (expected_count, expected_dim):
        raise ValueError(
            f"expected embedding shape {(expected_count, expected_dim)}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("embedding response contains NaN or infinity")
    return array


def request_with_retry(
    client: Any,
    texts: Sequence[str],
    model: str,
    expected_dim: int,
    max_attempts: int,
) -> np.ndarray:
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.embeddings.create(model=model, input=list(texts))
            return ordered_embeddings(response, len(texts), expected_dim)
        except Exception as exc:
            if attempt == max_attempts:
                raise
            delay = min(60.0, 2.0 ** (attempt - 1)) + random.random()
            print(
                f"Embedding request attempt {attempt}/{max_attempts} failed "
                f"with {type(exc).__name__}; retrying in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def create_client(base_url: str | None, timeout: float) -> Any:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set in the environment")
    kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout, "max_retries": 0}
    resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    return OpenAI(**kwargs)


def initialize_or_validate_artifact(
    artifact_dir: Path,
    num_inputs: int,
    dimension: int,
    model: str,
    input_sha256: str,
) -> tuple[np.ndarray, Path, Path]:
    embeddings_path = artifact_dir / "embeddings.npy"
    completed_path = artifact_dir / "embedding_completed.npy"
    manifest_path = artifact_dir / "embedding_manifest.json"
    chunks_dir = artifact_dir / "embedding_chunks"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "model": model,
            "dimension": dimension,
            "num_inputs": num_inputs,
            "unique_inputs_sha256": input_sha256,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"existing embedding artifact has {key}={manifest.get(key)!r}, "
                    f"expected {value!r}; use a new output directory"
                )
    else:
        if embeddings_path.exists() or completed_path.exists():
            raise FileExistsError(
                "embedding array exists without a manifest; move it aside or use a new directory"
            )
        write_json_atomic(
            manifest_path,
            {
                "model": model,
                "dimension": dimension,
                "num_inputs": num_inputs,
                "unique_inputs_sha256": input_sha256,
                "num_completed": 0,
                "complete": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    chunks_dir.mkdir(parents=True, exist_ok=True)
    completed = np.zeros(num_inputs, dtype=np.bool_)
    owners: Dict[int, tuple[Path, int]] = {}
    for chunk_path in sorted(chunks_dir.glob("chunk_*.npz")):
        with np.load(chunk_path) as chunk:
            indices = np.asarray(chunk["indices"], dtype=np.int64)
            vectors = np.asarray(chunk["embeddings"], dtype=np.float32)
        if vectors.shape != (len(indices), dimension):
            raise ValueError(f"invalid embedding chunk shape in {chunk_path}: {vectors.shape}")
        if np.any(indices < 0) or np.any(indices >= num_inputs):
            raise ValueError(f"out-of-range indices in {chunk_path}")
        if not np.isfinite(vectors).all():
            raise ValueError(f"non-finite embedding values in {chunk_path}")
        overlaps: Dict[Path, List[tuple[int, int]]] = {}
        for current_position, raw_index in enumerate(indices):
            index = int(raw_index)
            if index in owners:
                previous_path, previous_position = owners[index]
                overlaps.setdefault(previous_path, []).append(
                    (previous_position, current_position)
                )
            else:
                owners[index] = (chunk_path, current_position)
        for previous_path, positions in overlaps.items():
            with np.load(previous_path) as previous_chunk:
                previous_vectors = np.asarray(
                    previous_chunk["embeddings"], dtype=np.float32
                )
            previous_positions = [item[0] for item in positions]
            current_positions = [item[1] for item in positions]
            # The relay can return sub-millithreshold floating-point variation
            # for identical ada-002 inputs across requests. Accept only that
            # narrow numerical variation; semantic/model mismatches remain fatal.
            if not np.allclose(
                previous_vectors[previous_positions],
                vectors[current_positions],
                rtol=1e-4,
                atol=1e-3,
            ):
                raise ValueError(
                    f"overlapping embedding values disagree between "
                    f"{previous_path} and {chunk_path}"
                )
        completed[indices] = True
    return completed, manifest_path, chunks_dir


def write_embedding_chunk(chunks_dir: Path, indices: np.ndarray, vectors: np.ndarray) -> None:
    chunk_path = chunks_dir / f"chunk_{int(indices[0]):06d}_{int(indices[-1]):06d}.npz"
    temporary = chunk_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, indices=indices, embeddings=vectors)
    temporary.replace(chunk_path)


def consolidate_embeddings(
    artifact_dir: Path, chunks_dir: Path, num_inputs: int, dimension: int
) -> None:
    embeddings = np.empty((num_inputs, dimension), dtype=np.float32)
    completed = np.zeros(num_inputs, dtype=np.bool_)
    for chunk_path in sorted(chunks_dir.glob("chunk_*.npz")):
        with np.load(chunk_path) as chunk:
            indices = np.asarray(chunk["indices"], dtype=np.int64)
            vectors = np.asarray(chunk["embeddings"], dtype=np.float32)
        new_mask = ~completed[indices]
        embeddings[indices[new_mask]] = vectors[new_mask]
        completed[indices] = True
    if not completed.all():
        raise RuntimeError(
            f"cannot consolidate incomplete cache: {int(completed.sum())}/{num_inputs}"
        )
    embeddings_path = artifact_dir / "embeddings.npy"
    embeddings_tmp = artifact_dir / "embeddings.npy.tmp"
    with embeddings_tmp.open("wb") as handle:
        np.save(handle, embeddings)
    embeddings_tmp.replace(embeddings_path)
    completed_path = artifact_dir / "embedding_completed.npy"
    completed_tmp = artifact_dir / "embedding_completed.npy.tmp"
    with completed_tmp.open("wb") as handle:
        np.save(handle, completed)
    completed_tmp.replace(completed_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--model", default="text-embedding-ada-002")
    parser.add_argument("--dimension", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--base-url")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.dimension <= 0 or args.workers <= 0:
        raise ValueError("batch size, dimension and workers must be positive")
    input_path = args.artifact_dir / "unique_inputs.jsonl"
    inputs = read_jsonl(input_path)
    for index, row in enumerate(inputs):
        if row.get("embedding_index") != index:
            raise ValueError(f"non-contiguous embedding_index at input row {index}")

    client = create_client(args.base_url, args.timeout)
    if args.probe_only:
        vector = request_with_retry(
            client, [inputs[0]["text"]], args.model, args.dimension, args.max_attempts
        )
        print(
            f"Probe succeeded: model={args.model}, shape={tuple(vector.shape)}, "
            f"finite={bool(np.isfinite(vector).all())}"
        )
        return

    completed, manifest_path, chunks_dir = initialize_or_validate_artifact(
        args.artifact_dir,
        len(inputs),
        args.dimension,
        args.model,
        file_sha256(input_path),
    )
    incomplete = np.flatnonzero(~completed)
    if args.limit is not None:
        incomplete = incomplete[: args.limit]
    print(
        f"Embedding {len(incomplete)} remaining inputs "
        f"({int(completed.sum())}/{len(inputs)} already complete)",
        flush=True,
    )
    batches = [
        incomplete[start : start + args.batch_size]
        for start in range(0, len(incomplete), args.batch_size)
    ]

    def embed_batch(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        texts = [inputs[int(index)]["text"] for index in indices]
        vectors = request_with_retry(
            client, texts, args.model, args.dimension, args.max_attempts
        )
        return indices, vectors

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(embed_batch, indices) for indices in batches]
        for future in concurrent.futures.as_completed(futures):
            indices, vectors = future.result()
            write_embedding_chunk(chunks_dir, indices, vectors)
            completed[indices] = True
            total_completed = int(completed.sum())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update(
                {
                    "num_completed": total_completed,
                    "complete": total_completed == len(inputs),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            write_json_atomic(manifest_path, manifest)
            print(f"Completed {total_completed}/{len(inputs)}", flush=True)

    total_completed = int(completed.sum())
    if total_completed == len(inputs):
        consolidate_embeddings(args.artifact_dir, chunks_dir, len(inputs), args.dimension)
    print(f"Embedding cache status: {total_completed}/{len(inputs)} complete")


if __name__ == "__main__":
    main()
