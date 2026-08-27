from __future__ import annotations

import json
import unittest

from cell_annotation.dgpr import CANDIDATE_HEADER, parse_rollout
from cell_annotation.m1_refinement import (
    parse_cell_genes,
    validate_refinement,
    verify_critic_output,
)


CANDIDATES = ["T cell", "B cell", "monocyte"]


def response(answer: str) -> str:
    reasoning = " ".join(
        ["Marker genes support each lineage and the global one-to-one assignment."] * 12
    )
    return f"<reasoning>{reasoning}</reasoning>\n<answer>{answer}</answer>"


USER = (
    "Context: blood\n"
    "Cell 1: CD3D, CD3E, TRBC1, LTB\n"
    "Cell 2: MS4A1, CD79A, CD74, CD37\n"
    "Cell 3: LST1, S100A8, CTSS, TYROBP\n\n"
    f"{CANDIDATE_HEADER}\nT cell\nB cell\nmonocyte\n"
)


class M1RefinementTest(unittest.TestCase):
    def setUp(self):
        self.initial = parse_rollout(
            0, response("B cell | T cell | monocyte"), CANDIDATES
        )
        self.genes = parse_cell_genes(USER, CANDIDATE_HEADER)

    def test_cell_gene_parser(self):
        self.assertEqual(len(self.genes), 3)
        self.assertIn("cd3d", self.genes[0])
        self.assertIn("ms4a1", self.genes[1])

    def test_verified_swap_requires_grounded_genes(self):
        critic = json.dumps(
            {
                "verdict": "REVISE",
                "issue": {
                    "cell": 1,
                    "current_label": "B cell",
                    "alternative_label": "T cell",
                    "swap_partner": 2,
                    "supporting_genes": ["CD3D", "CD3E"],
                    "reason": "T-cell receptor genes support T cell.",
                },
            }
        )
        decision = verify_critic_output(
            critic, self.initial, CANDIDATES, self.genes
        )
        self.assertTrue(decision.valid)
        self.assertEqual(decision.status, "verified_revision")
        self.assertEqual(decision.issue.cell_index, 0)

    def test_hallucinated_evidence_is_rejected(self):
        critic = json.dumps(
            {
                "verdict": "REVISE",
                "issue": {
                    "cell": 1,
                    "current_label": "B cell",
                    "alternative_label": "T cell",
                    "swap_partner": 2,
                    "supporting_genes": ["FAKE1", "FAKE2"],
                    "reason": "invented evidence",
                },
            }
        )
        decision = verify_critic_output(
            critic, self.initial, CANDIDATES, self.genes
        )
        self.assertFalse(decision.valid)
        self.assertEqual(decision.status, "ungrounded_supporting_genes")

    def test_critic_accept_does_not_create_issue(self):
        decision = verify_critic_output(
            '{"verdict":"ACCEPT","issue":null}',
            self.initial,
            CANDIDATES,
            self.genes,
        )
        self.assertTrue(decision.valid)
        self.assertIsNone(decision.issue)

    def test_refinement_can_only_apply_verified_swap(self):
        critic = json.dumps(
            {
                "verdict": "REVISE",
                "issue": {
                    "cell": 1,
                    "current_label": "B cell",
                    "alternative_label": "T cell",
                    "swap_partner": 2,
                    "supporting_genes": ["CD3D", "CD3E"],
                    "reason": "T-cell evidence",
                },
            }
        )
        issue = verify_critic_output(
            critic, self.initial, CANDIDATES, self.genes
        ).issue
        accepted, _, _ = validate_refinement(
            response("T cell | B cell | monocyte"), CANDIDATES, self.initial, issue
        )
        rejected, reason, _ = validate_refinement(
            response("monocyte | T cell | B cell"), CANDIDATES, self.initial, issue
        )
        self.assertTrue(accepted)
        self.assertFalse(rejected)
        self.assertEqual(reason, "changed_unrelated_cell")


if __name__ == "__main__":
    unittest.main()

