from __future__ import annotations

import unittest

from cell_annotation.dgpr import (
    build_dgpr_plan,
    build_refinement_instruction,
    parse_candidate_labels,
    parse_rollout,
    validate_refinement,
)
from cell_annotation.evaluate_dgpr import compare


CANDIDATES = ["T cell", "B cell", "monocyte"]


def response(answer: str, prefix: str = "Evidence") -> str:
    reasoning = " ".join(
        [
            f"{prefix} marker genes support the assignments and exclude competing lineages."
            for _ in range(12)
        ]
    )
    return f"<reasoning>{reasoning}</reasoning>\n<answer>{answer}</answer>"


class DGPRTest(unittest.TestCase):
    def test_candidate_labels_are_parsed_from_prompt(self):
        prompt = (
            "Cell 1: CD3D, IL7R\nCell 2: MS4A1, CD79A\n\n"
            "Match the cells above to one of the following cell types:\n"
            "T cell\nB cell\n"
        )
        self.assertEqual(parse_candidate_labels(prompt), ["T cell", "B cell"])

    def test_rollout_requires_reasoning_and_candidate_permutation(self):
        valid = parse_rollout(0, response("T cell | B cell | monocyte"), CANDIDATES)
        duplicate = parse_rollout(1, response("T cell | T cell | monocyte"), CANDIDATES)
        collapsed = parse_rollout(
            2,
            "<reasoning>too short</reasoning>\n"
            "<answer>T cell | B cell | monocyte</answer>",
            CANDIDATES,
        )
        self.assertTrue(valid.valid)
        self.assertEqual(duplicate.invalid_reason, "duplicate_labels")
        self.assertEqual(collapsed.invalid_reason, "reasoning_collapse")

    def test_medoid_disagreement_selects_supported_challenger(self):
        texts = [
            response("T cell | B cell | monocyte", "Anchor"),
            response("T cell | B cell | monocyte", "Anchor sibling"),
            response("T cell | monocyte | B cell", "Alternative"),
            response("T cell | monocyte | B cell", "Alternative sibling"),
        ]
        rollouts = [parse_rollout(i, text, CANDIDATES) for i, text in enumerate(texts)]
        plan = build_dgpr_plan(rollouts, 3, consensus_threshold=0.75)
        self.assertEqual(plan.anchor_index, 0)
        self.assertEqual(plan.challenger_index, 2)
        self.assertEqual(plan.stable_indices, (0,))
        self.assertEqual(plan.uncertain_indices, (1, 2))
        self.assertTrue(plan.trigger_refinement)

    def test_unanimous_rollouts_do_not_trigger_refinement(self):
        rollouts = [
            parse_rollout(i, response("T cell | B cell | monocyte"), CANDIDATES)
            for i in range(8)
        ]
        plan = build_dgpr_plan(rollouts, 3, consensus_threshold=0.75)
        self.assertFalse(plan.trigger_refinement)
        self.assertEqual(plan.route_reason, "all_cells_stable")

    def test_refinement_preserves_stable_cells_and_observed_alternatives(self):
        texts = [
            response("T cell | B cell | monocyte", "Anchor"),
            response("T cell | B cell | monocyte", "Anchor sibling"),
            response("T cell | monocyte | B cell", "Alternative"),
            response("T cell | monocyte | B cell", "Alternative sibling"),
        ]
        rollouts = [parse_rollout(i, text, CANDIDATES) for i, text in enumerate(texts)]
        plan = build_dgpr_plan(rollouts, 3, consensus_threshold=0.75)
        anchor = rollouts[plan.anchor_index]
        accepted, reason, _ = validate_refinement(
            response("T cell | monocyte | B cell", "Revised"),
            CANDIDATES,
            anchor,
            plan,
        )
        rejected, rejected_reason, _ = validate_refinement(
            response("B cell | T cell | monocyte", "Bad revision"),
            CANDIDATES,
            anchor,
            plan,
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "accepted")
        self.assertFalse(rejected)
        self.assertEqual(rejected_reason, "changed_stable_cell")

    def test_refinement_prompt_contains_traces_but_no_ground_truth(self):
        texts = [
            response("T cell | B cell | monocyte", "Anchor"),
            response("T cell | monocyte | B cell", "Alternative"),
        ]
        rollouts = [parse_rollout(i, text, CANDIDATES) for i, text in enumerate(texts)]
        plan = build_dgpr_plan(rollouts, 3, consensus_threshold=0.75)
        instruction = build_refinement_instruction(
            rollouts[plan.anchor_index],
            rollouts[plan.challenger_index],
            plan,
            CANDIDATES,
        )
        self.assertIn("Anchor attempt", instruction)
        self.assertIn("Alternative attempt", instruction)
        self.assertNotIn("ground truth", instruction.casefold())

    def test_comparison_reports_corrected_and_harmed_batches(self):
        gold = "T cell | B cell | monocyte"
        anchor_rows = [
            {"id": "a", "answer": gold, "prediction": response("T cell | monocyte | B cell")},
            {"id": "b", "answer": gold, "prediction": response(gold)},
        ]
        final_rows = [
            {
                "id": "a",
                "answer": gold,
                "prediction": response(gold),
                "dgpr": {"trigger_refinement": True, "refinement_accepted": True, "refinement_status": "accepted"},
            },
            {
                "id": "b",
                "answer": gold,
                "prediction": response("T cell | monocyte | B cell"),
                "dgpr": {"trigger_refinement": True, "refinement_accepted": True, "refinement_status": "accepted"},
            },
        ]
        result = compare(anchor_rows, final_rows)
        self.assertEqual(result["batch_corrected"], 1)
        self.assertEqual(result["batch_harmed"], 1)
        self.assertEqual(result["cell_improved_batches"], 1)
        self.assertEqual(result["cell_harmed_batches"], 1)


if __name__ == "__main__":
    unittest.main()

