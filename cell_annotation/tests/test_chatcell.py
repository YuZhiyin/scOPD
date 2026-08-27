import numpy as np

from cell_annotation.infer_chatcell import (
    build_chatcell_prompt,
    parse_cellpuzzles_prompt,
    solve_assignment,
    solve_independent,
)


def test_parse_cellpuzzles_prompt():
    prompt = """Context: lung donor.

Cell 1: EPCAM, KRT19, KRT8
Cell 2: COL1A1, COL3A1, DCN

Match the cells above to one of the following cell types:
epithelial cell
fibroblast"""
    context, genes, candidates = parse_cellpuzzles_prompt(prompt)
    assert context == "Context: lung donor."
    assert genes == [["EPCAM", "KRT19", "KRT8"], ["COL1A1", "COL3A1", "DCN"]]
    assert candidates == ["epithelial cell", "fibroblast"]


def test_prompt_uses_ranked_genes_and_optional_information():
    prompt = build_chatcell_prompt(
        ["EPCAM", "KRT19"],
        context="Context: lung.",
        candidates=["epithelial cell", "fibroblast"],
        include_context=True,
        include_candidates=True,
    )
    assert "Context: lung." in prompt
    assert "EPCAM KRT19" in prompt
    assert "epithelial cell; fibroblast" in prompt


def test_hungarian_assignment_enforces_unique_candidates():
    scores = np.asarray([[5.0, 4.0], [4.9, 0.0]])
    labels, indices = solve_assignment(scores, ["a", "b"])
    assert indices == [1, 0]
    assert labels == ["b", "a"]


def test_independent_predictions_allow_repeated_candidates():
    scores = np.asarray([[5.0, 4.0], [4.9, 0.0]])
    labels, indices = solve_independent(scores, ["a", "b"])
    assert indices == [0, 0]
    assert labels == ["a", "a"]
