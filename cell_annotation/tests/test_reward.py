import unittest

from cell_annotation.common import score_prediction
from cell_annotation.reward import compute_score


GOLD = "T cell | B cell"


class RewardTest(unittest.TestCase):
    def test_strict_exact(self):
        prediction = "<think>joint assignment</think>\n<answer>T cell | B cell</answer>"
        metrics = score_prediction(prediction, GOLD)
        self.assertTrue(metrics["strict_exact_match"])
        self.assertEqual(compute_score("cell_annotation", prediction, GOLD)["score"], 1.0)

    def test_format_and_correctness_are_separate(self):
        result = compute_score("cell_annotation", "T cell | B cell", GOLD)
        self.assertEqual(result["format_reward"], 0.0)
        self.assertEqual(result["exact_match"], 1.0)
        self.assertAlmostEqual(result["score"], 0.9)

    def test_nonthinking_answer_only_is_strict(self):
        result = compute_score(
            "cell_annotation", "<answer>T cell | B cell</answer>", GOLD
        )
        self.assertEqual(result["format_reward"], 1.0)
        self.assertEqual(result["score"], 1.0)

    def test_nonthinking_explicit_reasoning_is_strict(self):
        prediction = (
            "<reasoning>NKG7 supports the T/NK assignment, while MS4A1 "
            "supports the B-cell assignment under the one-to-one constraint."
            "</reasoning>\n<answer>T cell | B cell</answer>"
        )
        metrics = score_prediction(prediction, GOLD)
        self.assertTrue(metrics["strict_format"])
        self.assertTrue(metrics["strict_exact_match"])

    def test_empty_nonthinking_reasoning_is_not_strict(self):
        prediction = (
            "<reasoning>  </reasoning>\n<answer>T cell | B cell</answer>"
        )
        metrics = score_prediction(prediction, GOLD)
        self.assertFalse(metrics["strict_format"])
        self.assertTrue(metrics["exact_match"])

    def test_partial_reward(self):
        prediction = "<think>x</think><answer>T cell | monocyte</answer>"
        result = compute_score("cell_annotation", prediction, GOLD)
        self.assertEqual(result["partial_accuracy"], 0.5)
        self.assertEqual(result["exact_match"], 0.0)
        self.assertAlmostEqual(result["score"], 0.225)
        self.assertIn("<answer>T cell | B cell</answer>", result["feedback"])


if __name__ == "__main__":
    unittest.main()
