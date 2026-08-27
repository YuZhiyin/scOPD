import pytest

from cell_annotation.evaluate_chatcell_cell_level import summarize_cell_level


def test_cell_level_metrics_use_independent_predictions_and_candidate_ranks():
    rows = [
        {
            "user_msg": """Context: lung.

Cell 1: EPCAM, KRT19
Cell 2: COL1A1, DCN

Match the cells above to one of the following cell types:
epithelial cell
fibroblast""",
            "answer": "epithelial cell | fibroblast",
            "prediction": "<answer>epithelial cell | epithelial cell</answer>",
            "chatcell_assignment_mode": "independent",
            "chatcell_score_matrix": [[5.0, 1.0], [4.0, 3.0]],
        }
    ]
    result = summarize_cell_level(rows, top_k=(1, 2))
    assert result["evaluation_protocol"] == "closed_set_cell_level"
    assert result["assignment_modes"] == ["independent"]
    assert result["num_cells"] == 2
    assert result["num_correct"] == 1
    assert result["cell_accuracy"] == 0.5
    assert result["top_1_accuracy"] == 0.5
    assert result["top_2_accuracy"] == 1.0
    assert result["mean_reciprocal_rank"] == pytest.approx(0.75)
    assert result["macro_cell_type_recall"] == 0.5
