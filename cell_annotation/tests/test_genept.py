import json
from pathlib import Path

import numpy as np

from cell_annotation.evaluate_genept_s import exact_cosine_knn, majority_vote
from cell_annotation.genept import GENEPT_S_PREFIX, genept_s_text, read_jsonl
from cell_annotation.prepare_genept_s import prepare


def _row(row_id: str, cells, labels):
    cell_lines = "\n".join(
        f"Cell {index + 1}: {', '.join(genes)}" for index, genes in enumerate(cells)
    )
    candidates = "\n".join(reversed(labels))
    return {
        "id": row_id,
        "split": "toy",
        "user_msg": (
            f"Toy context\n{cell_lines}\n"
            "Match the cells above to one of the following cell types:\n"
            f"{candidates}"
        ),
        "answer": " | ".join(labels),
    }


def test_genept_sentence_preserves_rank_order():
    assert genept_s_text(["CD3D", " CD3E ", "TRBC1"]) == (
        GENEPT_S_PREFIX + "CD3D CD3E TRBC1"
    )


def test_prepare_flattens_and_deduplicates(tmp_path: Path):
    input_path = tmp_path / "toy.json"
    rows = [
        _row("a", [["CD3D", "CD3E"], ["MS4A1", "CD79A"]], ["T cell", "B cell"]),
        _row("b", [["CD3D", "CD3E"]], ["T cell"]),
    ]
    input_path.write_text(json.dumps(rows), encoding="utf-8")
    output_dir = tmp_path / "artifact"
    manifest = prepare([f"train={input_path}"], output_dir)
    assert manifest["num_cells"] == 3
    assert manifest["num_unique_inputs"] == 2
    records = read_jsonl(output_dir / "records.jsonl")
    assert records[0]["embedding_index"] == records[2]["embedding_index"]
    assert records[0]["num_genes"] == 2


def test_majority_vote_uses_similarity_for_ties():
    assert majority_vote(["a", "b", "a", "b"], [0.5, 0.8, 0.4, 0.7]) == "b"
    assert majority_vote(["b", "a"], [0.5, 0.5]) == "a"


def test_exact_cosine_knn_cpu():
    reference = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    queries = np.asarray([[0.9, 0.1], [0.0, 2.0]], dtype=np.float32)
    similarities, indices = exact_cosine_knn(
        reference, queries, k=2, device="cpu", query_batch_size=1
    )
    assert indices.shape == (2, 2)
    assert indices[0, 0] == 0
    assert indices[1, 0] == 1
    assert np.allclose(similarities[:, 0], 1.0, atol=0.01)
