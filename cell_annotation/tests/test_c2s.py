import numpy as np
import pytest

from cell_annotation.c2s_common import (
    C2S_PROMPT_VERSION,
    build_c2s_prompt,
    c2s_target,
    expand_cellpuzzles_batch,
    select_independent,
)
from cell_annotation.evaluate_c2s_cell_level import summarize_c2s


EXAMPLE_PROMPT = """Context: lung donor.

Cell 1: EPCAM, KRT19, KRT8
Cell 2: COL1A1, COL3A1, DCN

Match the cells above to one of the following cell types:
epithelial cell
fibroblast"""


def test_c2s_prompt_has_ranked_genes_candidates_and_official_response_cue():
    prompt = build_c2s_prompt(
        genes=["EPCAM", "KRT19", "KRT8"],
        candidates=["epithelial cell", "fibroblast"],
    )
    assert "3 gene names ordered by descending expression level" in prompt
    assert "Cell sentence: EPCAM KRT19 KRT8." in prompt
    assert "epithelial cell; fibroblast" in prompt
    assert prompt.endswith("The cell type corresponding to these genes is:")
    assert "lung donor" not in prompt
    assert C2S_PROMPT_VERSION == "c2s_cellpuzzles_candidates_v1"


def test_c2s_target_matches_official_label_and_period_style():
    assert c2s_target(" epithelial   cell ") == " epithelial cell."


def test_expand_batch_produces_independent_single_cell_examples():
    rows = expand_cellpuzzles_batch(
        {
            "id": "train_000001",
            "split": "train",
            "user_msg": EXAMPLE_PROMPT,
            "answer": "epithelial cell | fibroblast",
        }
    )
    assert len(rows) == 2
    assert rows[0]["id"] == "train_000001_c2s_01"
    assert rows[0]["genes"] == ["EPCAM", "KRT19", "KRT8"]
    assert rows[0]["cell_type"] == "epithelial cell"
    assert rows[0]["response"] == " epithelial cell."
    assert rows[1]["cell_index"] == 1
    assert rows[1]["candidates"] == ["epithelial cell", "fibroblast"]


def test_expand_batch_rejects_gold_outside_candidates():
    with pytest.raises(ValueError, match="absent from candidates"):
        expand_cellpuzzles_batch(
            {
                "id": "train_bad",
                "user_msg": EXAMPLE_PROMPT,
                "answer": "epithelial cell | T cell",
            }
        )


def test_c2s_independent_selection_allows_duplicate_predictions():
    labels, indices = select_independent(
        np.asarray([[5.0, 1.0], [4.0, 3.0]]),
        ["epithelial cell", "fibroblast"],
    )
    assert labels == ["epithelial cell", "epithelial cell"]
    assert indices == [0, 0]


def test_c2s_metrics_include_seen_and_unseen_training_labels():
    rows = [
        {
            "id": "test_1",
            "user_msg": EXAMPLE_PROMPT,
            "answer": "epithelial cell | fibroblast",
            "prediction": "<answer>epithelial cell | epithelial cell</answer>",
            "c2s_score_matrix": [[5.0, 1.0], [4.0, 3.0]],
        }
    ]
    summary = summarize_c2s(
        rows,
        known_labels={"epithelial cell"},
        top_k=(1, 2),
    )
    assert summary["evaluation_protocol"] == "closed_set_cell_level_independent"
    assert summary["cell_accuracy"] == 0.5
    assert summary["top_2_accuracy"] == 1.0
    assert (
        summary["by_train_label_status"]["train_seen_labels"]["cell_accuracy"]
        == 1.0
    )
    assert (
        summary["by_train_label_status"]["train_unseen_labels"]["cell_accuracy"]
        == 0.0
    )
