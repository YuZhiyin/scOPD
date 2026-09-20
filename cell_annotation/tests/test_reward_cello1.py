import unittest

from cell_annotation.reward_cello1 import (
    compute_nonthinking_reasoning_joint_gt_score,
    compute_nonthinking_reasoning_joint_hierarchical_score,
    compute_nonthinking_reasoning_repo_o1_fallback_sibling_score,
    compute_nonthinking_reasoning_repo_sibling_score,
    compute_nonthinking_reasoning_sparse_o1_fallback_sibling_score,
    compute_nonthinking_reasoning_sparse_score,
    compute_nonthinking_reasoning_sparse_sibling_score,
    compute_repo_o1_fallback_sibling_score,
    compute_repo_sibling_score,
    compute_scopd_ropsd_score,
    compute_score,
    compute_sparse_o1_fallback_sibling_score,
    compute_sparse_score,
    compute_sparse_sibling_score,
)

GOLD = "T cell | B cell"


class CellO1RewardTest(unittest.TestCase):
    def test_strict_exact_routes_to_grpo_with_one_or_two_newlines(self):
        for separator in ("\n", "\n\n"):
            with self.subTest(separator=repr(separator)):
                prediction = (
                    "<think>Use marker genes and the one-to-one constraint.</think>"
                    f"{separator}"
                    "<answer>T cell | B cell</answer>"
                )
                result = compute_score("cell_annotation", prediction, GOLD)
                self.assertEqual(result["score"], 1.0)
                self.assertEqual(result["strict_format"], 1.0)
                self.assertEqual(result["format_reward"], 1.0)
                self.assertEqual(result["route_to_grpo"], 1.0)
                self.assertEqual(result["route_to_sdpo"], 0.0)
                self.assertIsNone(result["feedback"])

    def test_strict_format_accepts_crlf_and_whitespace_only_blank_lines(self):
        prediction = (
            "<think>Use marker genes.</think> \r\n"
            " \t\r\n"
            "\t<answer>T cell | B cell</answer>"
        )
        result = compute_score("cell_annotation", prediction, GOLD)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["strict_format"], 1.0)
        self.assertEqual(result["route_to_grpo"], 1.0)

    def test_strict_format_rejects_same_line_or_text_between_blocks(self):
        predictions = (
            "<think>x</think><answer>T cell | B cell</answer>",
            "<think>x</think>\nextra text\n<answer>T cell | B cell</answer>",
        )
        for prediction in predictions:
            with self.subTest(prediction=prediction):
                result = compute_score("cell_annotation", prediction, GOLD)
                self.assertEqual(result["score"], -1.0)
                self.assertEqual(result["strict_format"], 0.0)
                self.assertEqual(result["route_to_sdpo"], 1.0)

    def test_parseable_partial_matches_cello1_reward(self):
        prediction = (
            "<think>One assignment is still ambiguous.</think>\n"
            "<answer>T cell | monocyte</answer>"
        )
        result = compute_score("cell_annotation", prediction, GOLD)
        self.assertEqual(result["score"], 0.25)
        self.assertEqual(result["partial_accuracy"], 0.5)
        self.assertEqual(result["route_to_sdpo"], 1.0)
        self.assertIn("<answer>T cell | B cell</answer>", result["feedback"])
        self.assertNotIn("monocyte", result["feedback"])

    def test_invalid_format_gets_minus_one_and_sdpo(self):
        result = compute_score(
            "cell_annotation", "<answer>T cell | B cell</answer>", GOLD
        )
        self.assertEqual(result["score"], -1.0)
        self.assertEqual(result["route_to_sdpo"], 1.0)

    def test_duplicate_gets_minus_one(self):
        prediction = (
            "<think>bad permutation</think>\n"
            "<answer>T cell | T cell</answer>"
        )
        result = compute_score("cell_annotation", prediction, GOLD)
        self.assertEqual(result["score"], -1.0)
        self.assertEqual(result["route_to_sdpo"], 1.0)


class CellO1SparseRewardTest(unittest.TestCase):
    def test_exact_batch_gets_one_and_routes_to_grpo(self):
        prediction = (
            "<think>Both assignments are supported.</think>\n\n"
            "<answer>T cell | B cell</answer>"
        )
        result = compute_sparse_score("cell_annotation", prediction, GOLD)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["route_to_grpo"], 1.0)
        self.assertEqual(result["route_to_sdpo"], 0.0)

    def test_valid_partial_batch_gets_zero_and_routes_to_sdpo(self):
        prediction = (
            "<think>One assignment is still ambiguous.</think>\n"
            "<answer>T cell | monocyte</answer>"
        )
        result = compute_sparse_score("cell_annotation", prediction, GOLD)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["partial_accuracy"], 0.5)
        self.assertEqual(result["route_to_grpo"], 0.0)
        self.assertEqual(result["route_to_sdpo"], 1.0)
        self.assertIn("<answer>T cell | B cell</answer>", result["feedback"])

    def test_malformed_output_gets_minus_one(self):
        result = compute_sparse_score(
            "cell_annotation", "<answer>T cell | B cell</answer>", GOLD
        )
        self.assertEqual(result["score"], -1.0)
        self.assertEqual(result["route_to_sdpo"], 1.0)

    def test_count_mismatch_and_duplicate_get_minus_one(self):
        predictions = (
            "<think>missing assignment</think>\n<answer>T cell</answer>",
            "<think>duplicate assignment</think>\n"
            "<answer>T cell | T cell</answer>",
        )
        for prediction in predictions:
            with self.subTest(prediction=prediction):
                result = compute_sparse_score(
                    "cell_annotation", prediction, GOLD
                )
                self.assertEqual(result["score"], -1.0)
                self.assertEqual(result["route_to_sdpo"], 1.0)


class NonThinkingReasoningRewardTest(unittest.TestCase):
    @staticmethod
    def _substantive_reasoning() -> str:
        return (
            "The marker genes support the candidate identity while the cells "
            "must also satisfy the global one-to-one matching constraint. " * 10
        )

    def test_sparse_reward_accepts_only_reasoning_protocol(self):
        valid = (
            "<reasoning>Marker genes resolve both cells.</reasoning>\n"
            "<answer>T cell | B cell</answer>"
        )
        result = compute_nonthinking_reasoning_sparse_score(
            "cell_annotation", valid, GOLD
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["strict_format"], 1.0)

        native_thinking = (
            "<think>Marker genes resolve both cells.</think>\n"
            "<answer>T cell | B cell</answer>"
        )
        rejected = compute_nonthinking_reasoning_sparse_score(
            "cell_annotation", native_thinking, GOLD
        )
        self.assertEqual(rejected["score"], -1.0)
        self.assertEqual(rejected["strict_format"], 0.0)

    def test_dense_sibling_o1_reward_uses_reasoning_quality_and_converts_o1(self):
        wrong = (
            f"<reasoning>{self._substantive_reasoning()}</reasoning>\n"
            "<answer>T cell | monocyte</answer>"
        )
        o1_trace = (
            "<think>Expert marker-gene and matching analysis.</think>\n"
            "<answer>T cell | B cell</answer>"
        )
        result = compute_nonthinking_reasoning_repo_o1_fallback_sibling_score(
            "cell_annotation", wrong, GOLD, {"o1_reasoning": o1_trace}
        )
        self.assertEqual(result["score"], 0.25)
        self.assertEqual(result["reasoning_normal"], 1.0)
        self.assertEqual(result["teacher_eligible"], 0.0)
        self.assertEqual(result["o1_fallback_requested"], 1.0)
        self.assertIn("<reasoning>", result["feedback"])
        self.assertNotIn("<think>", result["feedback"])

    def test_canonical_ropsd_entry_point_matches_main_route(self):
        wrong = (
            f"<reasoning>{self._substantive_reasoning()}</reasoning>\n"
            "<answer>T cell | monocyte</answer>"
        )
        o1_trace = (
            "<think>Expert marker-gene and matching analysis.</think>\n"
            "<answer>T cell | B cell</answer>"
        )
        expected = compute_nonthinking_reasoning_repo_o1_fallback_sibling_score(
            "cell_annotation", wrong, GOLD, {"o1_reasoning": o1_trace}
        )
        actual = compute_scopd_ropsd_score(
            "cell_annotation", wrong, GOLD, {"o1_reasoning": o1_trace}
        )
        self.assertEqual(actual, expected)

    def test_dense_exact_reasoning_rollout_is_sibling_eligible(self):
        exact = (
            f"<reasoning>{self._substantive_reasoning()}</reasoning>\n\n"
            "<answer>T cell | B cell</answer>"
        )
        result = compute_nonthinking_reasoning_repo_o1_fallback_sibling_score(
            "cell_annotation", exact, GOLD, {"o1_reasoning": None}
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["teacher_eligible"], 1.0)
        self.assertEqual(result["route_to_grpo"], 1.0)
        self.assertIsNone(result["feedback"])

    def test_dense_sibling_only_has_no_o1_feedback(self):
        wrong = (
            f"<reasoning>{self._substantive_reasoning()}</reasoning>\n"
            "<answer>T cell | monocyte</answer>"
        )
        result = compute_nonthinking_reasoning_repo_sibling_score(
            "cell_annotation",
            wrong,
            GOLD,
            {"o1_reasoning": "<reasoning>privileged trace</reasoning>"},
        )
        self.assertEqual(result["score"], 0.25)
        self.assertEqual(result["route_to_sdpo"], 1.0)
        self.assertIsNone(result["feedback"])
        self.assertNotIn("o1_fallback_requested", result)

    def test_sparse_sibling_uses_batch_exact_route_and_no_gt_feedback(self):
        exact = (
            f"<reasoning>{self._substantive_reasoning()}</reasoning>\n"
            "<answer>T cell | B cell</answer>"
        )
        exact_result = compute_nonthinking_reasoning_sparse_sibling_score(
            "cell_annotation", exact, GOLD, {"o1_reasoning": "ignored"}
        )
        self.assertEqual(exact_result["score"], 1.0)
        self.assertEqual(exact_result["teacher_eligible"], 1.0)
        self.assertEqual(exact_result["route_to_grpo"], 1.0)
        self.assertIsNone(exact_result["feedback"])

        partial = (
            f"<reasoning>{self._substantive_reasoning()}</reasoning>\n"
            "<answer>T cell | monocyte</answer>"
        )
        partial_result = compute_nonthinking_reasoning_sparse_sibling_score(
            "cell_annotation", partial, GOLD, {"o1_reasoning": "ignored"}
        )
        self.assertEqual(partial_result["score"], 0.0)
        self.assertEqual(partial_result["teacher_eligible"], 0.0)
        self.assertEqual(partial_result["route_to_sdpo"], 1.0)
        self.assertIsNone(partial_result["feedback"])

    def test_sparse_o1_fallback_converts_trace_and_preserves_sparse_reward(self):
        partial = (
            f"<reasoning>{self._substantive_reasoning()}</reasoning>\n"
            "<answer>T cell | monocyte</answer>"
        )
        o1_trace = (
            "<think>Expert marker-gene and matching analysis.</think>\n"
            "<answer>T cell | B cell</answer>"
        )
        result = compute_nonthinking_reasoning_sparse_o1_fallback_sibling_score(
            "cell_annotation", partial, GOLD, {"o1_reasoning": o1_trace}
        )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["teacher_eligible"], 0.0)
        self.assertEqual(result["o1_fallback_requested"], 1.0)
        self.assertIn("<reasoning>", result["feedback"])
        self.assertNotIn("<think>", result["feedback"])

    def test_joint_gt_exposes_gt_for_correct_and_wrong_rollouts(self):
        responses = (
            (
                f"<reasoning>{self._substantive_reasoning()}</reasoning>\n"
                "<answer>T cell | B cell</answer>"
            ),
            (
                f"<reasoning>{self._substantive_reasoning()}</reasoning>\n"
                "<answer>T cell | monocyte</answer>"
            ),
        )
        for response in responses:
            with self.subTest(response=response[-80:]):
                result = compute_nonthinking_reasoning_joint_gt_score(
                    "cell_annotation", response, GOLD, {"o1_reasoning": "ignored"}
                )
                self.assertIn(
                    "<answer>T cell | B cell</answer>",
                    result["teacher_base_feedback"],
                )
                self.assertIsNone(result["feedback"])

    def test_joint_hierarchy_always_has_gt_and_exposes_o1_as_fallback(self):
        exact = (
            f"<reasoning>{self._substantive_reasoning()}</reasoning>\n"
            "<answer>T cell | B cell</answer>"
        )
        o1_trace = (
            "<think>Expert process trace.</think>\n"
            "<answer>T cell | B cell</answer>"
        )
        result = compute_nonthinking_reasoning_joint_hierarchical_score(
            "cell_annotation", exact, GOLD, {"o1_reasoning": o1_trace}
        )
        self.assertEqual(result["teacher_eligible"], 1.0)
        self.assertIn("<answer>T cell | B cell</answer>", result["teacher_base_feedback"])
        self.assertIn("<reasoning>", result["feedback"])
        self.assertNotIn("<think>", result["feedback"])
        self.assertEqual(result["o1_reasoning_available"], 1.0)


class SiblingSRPORewardTest(unittest.TestCase):
    @staticmethod
    def _substantive_reasoning() -> str:
        sentence = (
            "The ranked genes support a lymphoid interpretation and the two "
            "profiles must satisfy the one-to-one assignment constraint. "
        )
        return sentence * 10

    def test_exact_with_substantive_reasoning_is_teacher_eligible(self):
        prediction = (
            f"<think>{self._substantive_reasoning()}</think>\n\n"
            "<answer>T cell | B cell</answer>"
        )
        result = compute_sparse_sibling_score(
            "cell_annotation", prediction, GOLD
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["reasoning_normal"], 1.0)
        self.assertEqual(result["teacher_eligible"], 1.0)
        self.assertEqual(result["route_to_grpo"], 1.0)
        self.assertIsNone(result["feedback"])

    def test_exact_placeholder_is_not_teacher_eligible(self):
        prediction = (
            "<think>See detailed reasoning attached. "
            + "Additional details are available elsewhere. " * 30
            + "</think>\n<answer>T cell | B cell</answer>"
        )
        result = compute_sparse_sibling_score(
            "cell_annotation", prediction, GOLD
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["reasoning_placeholder"], 1.0)
        self.assertEqual(result["teacher_eligible"], 0.0)
        self.assertEqual(result["route_to_sdpo"], 1.0)

    def test_wrong_answer_requests_sdpo_without_gt_feedback(self):
        prediction = (
            f"<think>{self._substantive_reasoning()}</think>\n"
            "<answer>T cell | monocyte</answer>"
        )
        result = compute_sparse_sibling_score(
            "cell_annotation", prediction, GOLD
        )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["teacher_eligible"], 0.0)
        self.assertEqual(result["route_to_sdpo"], 1.0)
        self.assertIsNone(result["feedback"])

    def test_repo_variant_keeps_dense_partial_reward_and_same_route(self):
        prediction = (
            f"<think>{self._substantive_reasoning()}</think>\n"
            "<answer>T cell | monocyte</answer>"
        )
        result = compute_repo_sibling_score(
            "cell_annotation", prediction, GOLD
        )
        self.assertEqual(result["score"], 0.25)
        self.assertEqual(result["partial_accuracy"], 0.5)
        self.assertEqual(result["teacher_eligible"], 0.0)
        self.assertEqual(result["route_to_sdpo"], 1.0)
        self.assertIsNone(result["feedback"])

    def test_repo_variant_exact_rollout_is_teacher_eligible(self):
        prediction = (
            f"<think>{self._substantive_reasoning()}</think>\n\n"
            "<answer>T cell | B cell</answer>"
        )
        result = compute_repo_sibling_score(
            "cell_annotation", prediction, GOLD
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["teacher_eligible"], 1.0)
        self.assertEqual(result["route_to_grpo"], 1.0)

    def test_o1_feedback_is_exposed_only_for_wrong_answer(self):
        reasoning = "<think>Verified expert reasoning.</think>\n<answer>T cell | B cell</answer>"
        wrong = (
            f"<think>{self._substantive_reasoning()}</think>\n"
            "<answer>T cell | monocyte</answer>"
        )
        wrong_result = compute_repo_o1_fallback_sibling_score(
            "cell_annotation", wrong, GOLD, {"o1_reasoning": reasoning}
        )
        self.assertEqual(wrong_result["score"], 0.25)
        self.assertEqual(wrong_result["o1_fallback_requested"], 1.0)
        self.assertEqual(wrong_result["feedback"], reasoning)

        exact = (
            f"<think>{self._substantive_reasoning()}</think>\n"
            "<answer>T cell | B cell</answer>"
        )
        exact_result = compute_repo_o1_fallback_sibling_score(
            "cell_annotation", exact, GOLD, {"o1_reasoning": reasoning}
        )
        self.assertEqual(exact_result["teacher_eligible"], 1.0)
        self.assertEqual(exact_result["o1_fallback_requested"], 0.0)
        self.assertIsNone(exact_result["feedback"])

    def test_wrong_answer_without_o1_has_no_feedback(self):
        wrong = (
            f"<think>{self._substantive_reasoning()}</think>\n"
            "<answer>T cell | monocyte</answer>"
        )
        result = compute_repo_o1_fallback_sibling_score(
            "cell_annotation", wrong, GOLD, {"o1_reasoning": None}
        )
        self.assertEqual(result["o1_reasoning_available"], 0.0)
        self.assertEqual(result["o1_fallback_requested"], 0.0)
        self.assertIsNone(result["feedback"])

    def test_sparse_o1_variant_preserves_sparse_reward_and_feedback(self):
        reasoning = (
            "<think>Verified expert reasoning.</think>\n"
            "<answer>T cell | B cell</answer>"
        )
        wrong = (
            f"<think>{self._substantive_reasoning()}</think>\n"
            "<answer>T cell | monocyte</answer>"
        )
        wrong_result = compute_sparse_o1_fallback_sibling_score(
            "cell_annotation", wrong, GOLD, {"o1_reasoning": reasoning}
        )
        self.assertEqual(wrong_result["score"], 0.0)
        self.assertEqual(wrong_result["teacher_eligible"], 0.0)
        self.assertEqual(wrong_result["o1_fallback_requested"], 1.0)
        self.assertEqual(wrong_result["feedback"], reasoning)

        exact = (
            f"<think>{self._substantive_reasoning()}</think>\n"
            "<answer>T cell | B cell</answer>"
        )
        exact_result = compute_sparse_o1_fallback_sibling_score(
            "cell_annotation", exact, GOLD, {"o1_reasoning": reasoning}
        )
        self.assertEqual(exact_result["score"], 1.0)
        self.assertEqual(exact_result["teacher_eligible"], 1.0)
        self.assertEqual(exact_result["o1_fallback_requested"], 0.0)
        self.assertIsNone(exact_result["feedback"])


if __name__ == "__main__":
    unittest.main()
