import unittest

from cell_annotation.prepare_reasoning_sft import (
    EXPLICIT_REASONING_INSTRUCTION,
    NONTHINKING_REASONING_INSTRUCTION,
    ORIGINAL_REASONING_INSTRUCTION,
    validate_and_normalize,
)


class NonThinkingReasoningDataTest(unittest.TestCase):
    def setUp(self) -> None:
        system = (
            "Annotate the cells. "
            f"{ORIGINAL_REASONING_INSTRUCTION} "
            "The final answer must preserve Cell order."
        )
        self.reasoning = [
            {
                "system_msg": system,
                "user_msg": "Cell 1: NKG7, GNLY\nCell 2: MS4A1, CD79A",
                "assistant_msg": (
                    "<think>Cell 1 has NK markers and Cell 2 has B-cell "
                    "markers; the one-to-one constraint fixes the mapping."
                    "</think>\n<answer>natural killer cell | B cell</answer>"
                ),
            }
        ]
        self.train = [
            {
                "system_msg": system,
                "user_msg": self.reasoning[0]["user_msg"],
                "assistant_msg": "natural killer cell | B cell",
            }
        ]
        self.test = [
            {
                "system_msg": system,
                "user_msg": "A disjoint official test prompt",
                "assistant_msg": "T cell | monocyte",
            }
        ]

    def test_converts_trace_and_prompt_for_hard_nonthinking_template(self) -> None:
        row = validate_and_normalize(
            self.reasoning,
            self.train,
            self.test,
            response_mode="nonthinking_reasoning",
        )[0]

        self.assertIn(NONTHINKING_REASONING_INSTRUCTION, row["system_msg"])
        self.assertNotIn(ORIGINAL_REASONING_INSTRUCTION, row["system_msg"])
        self.assertEqual(
            row["assistant_msg"],
            "<reasoning>Cell 1 has NK markers and Cell 2 has B-cell markers; "
            "the one-to-one constraint fixes the mapping.</reasoning>\n"
            "<answer>natural killer cell | B cell</answer>",
        )
        self.assertNotIn("<think>", row["assistant_msg"])

    def test_thinking_mode_remains_backward_compatible(self) -> None:
        row = validate_and_normalize(
            self.reasoning,
            self.train,
            self.test,
            response_mode="thinking",
        )[0]
        self.assertEqual(row["assistant_msg"], self.reasoning[0]["assistant_msg"])
        self.assertIn(ORIGINAL_REASONING_INSTRUCTION, row["system_msg"])

    def test_explicit_reasoning_uses_model_agnostic_instruction(self) -> None:
        row = validate_and_normalize(
            self.reasoning,
            self.train,
            self.test,
            response_mode="explicit_reasoning",
        )[0]
        self.assertIn(EXPLICIT_REASONING_INSTRUCTION, row["system_msg"])
        self.assertNotIn("non-thinking mode", row["system_msg"])
        self.assertTrue(row["assistant_msg"].startswith("<reasoning>"))
        self.assertTrue(row["assistant_msg"].endswith("</answer>"))

    def test_rejects_missing_source_instruction(self) -> None:
        self.reasoning[0]["system_msg"] = "Annotate the cells."
        with self.assertRaisesRegex(ValueError, "missing expected output instruction"):
            validate_and_normalize(
                self.reasoning,
                self.train,
                self.test,
                response_mode="nonthinking_reasoning",
            )


if __name__ == "__main__":
    unittest.main()
