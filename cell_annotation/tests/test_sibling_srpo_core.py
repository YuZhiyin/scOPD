import unittest

import torch

from verl.trainer.ppo.core_algos import (
    build_token_mask_before_subsequence,
    combine_joint_grpo_sdpo_losses,
    combine_routed_branch_losses,
    combine_selective_joint_grpo_sdpo_losses,
    combine_source_weighted_joint_losses,
    entropy_confidence_weights,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class SiblingRoutingTest(unittest.TestCase):
    def test_failed_rollout_with_sibling_routes_to_sdpo(self):
        routes = RayPPOTrainer._resolve_routed_sdpo_routes(
            route_requests=[False, True, True],
            solution_strs=["correct sibling", "correct sibling", None],
            require_successful_solution=True,
        )
        self.assertEqual(routes, [0.0, 1.0, 0.0])

    def test_legacy_feedback_routing_does_not_require_sibling(self):
        routes = RayPPOTrainer._resolve_routed_sdpo_routes(
            route_requests=[False, True],
            solution_strs=[None, None],
            require_successful_solution=False,
        )
        self.assertEqual(routes, [0.0, 1.0])

    def test_o1_feedback_routes_only_when_sibling_is_missing(self):
        routes = RayPPOTrainer._resolve_routed_sdpo_routes(
            route_requests=[True, True, True],
            solution_strs=["correct sibling", None, None],
            require_successful_solution=True,
            feedback_used=[False, True, False],
            allow_feedback_fallback=True,
        )
        self.assertEqual(routes, [1.0, 1.0, 0.0])

    def test_selective_joint_sdpo_uses_same_qualified_teacher_routes(self):
        """Selective Joint changes the GRPO branch, not teacher eligibility."""
        sdpo_mask = RayPPOTrainer._resolve_routed_sdpo_routes(
            route_requests=[False, True, True, True],
            solution_strs=["qualified sibling", "qualified sibling", None, None],
            require_successful_solution=True,
            feedback_used=[False, False, True, False],
            allow_feedback_fallback=True,
        )
        self.assertEqual(sdpo_mask, [0.0, 1.0, 1.0, 0.0])
        # Dense GRPO is intentionally active for every rollout, including the
        # two samples that additionally receive SDPO.
        grpo_mask = [1.0] * len(sdpo_mask)
        self.assertEqual(grpo_mask, [1.0, 1.0, 1.0, 1.0])


class RoutedLossNormalizationTest(unittest.TestCase):
    def test_local_token_normalization(self):
        loss = combine_routed_branch_losses(
            sdpo_loss=torch.tensor(2.0),
            grpo_loss=torch.tensor(1.0),
            sdpo_token_count=torch.tensor(2.0),
            grpo_token_count=torch.tensor(6.0),
        )
        self.assertAlmostEqual(loss.item(), 1.25)

    def test_global_token_normalization_accounts_for_dp_average(self):
        # This rank contributes a routed-token numerator of 10. With two DP
        # ranks and 20 tokens globally, its pre-averaging loss is 10*2/20=1.
        loss = combine_routed_branch_losses(
            sdpo_loss=torch.tensor(2.0),
            grpo_loss=torch.tensor(1.0),
            sdpo_token_count=torch.tensor(2.0),
            grpo_token_count=torch.tensor(6.0),
            global_token_count=torch.tensor(20.0),
            dp_size=2,
        )
        self.assertAlmostEqual(loss.item(), 1.0)

    def test_entropy_confidence_weights_have_unit_mean(self):
        entropy = torch.tensor([0.0, 1.0, 2.0])
        raw = torch.exp(-entropy)
        weights = entropy_confidence_weights(
            teacher_entropy=entropy,
            beta=1.0,
            mean_unnormalized_weight=raw.mean(),
        )
        self.assertAlmostEqual(weights.mean().item(), 1.0, places=6)
        self.assertGreater(weights[0].item(), weights[1].item())
        self.assertGreater(weights[1].item(), weights[2].item())


class JointLossTest(unittest.TestCase):
    def test_lambda_point_three_normalized_combination(self):
        loss = combine_joint_grpo_sdpo_losses(
            grpo_loss=torch.tensor(2.0),
            sdpo_loss=torch.tensor(1.0),
            sdpo_weight=0.3,
        )
        self.assertAlmostEqual(loss.item(), 2.3 / 1.3, places=6)

    def test_negative_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            combine_joint_grpo_sdpo_losses(
                grpo_loss=torch.tensor(1.0),
                sdpo_loss=torch.tensor(1.0),
                sdpo_weight=-0.1,
            )

    def test_selective_joint_preserves_full_grpo_coefficient(self):
        loss = combine_selective_joint_grpo_sdpo_losses(
            grpo_loss=torch.tensor(2.0),
            sdpo_loss=torch.tensor(1.0),
            sdpo_weight=0.3,
        )
        self.assertAlmostEqual(loss.item(), 2.3, places=6)

    def test_selective_joint_empty_sdpo_is_exact_grpo(self):
        loss = combine_selective_joint_grpo_sdpo_losses(
            grpo_loss=torch.tensor(2.0),
            sdpo_loss=torch.tensor(0.0),
            sdpo_weight=0.3,
        )
        self.assertAlmostEqual(loss.item(), 2.0, places=6)

    def test_source_weighted_joint_normalization(self):
        loss = combine_source_weighted_joint_losses(
            grpo_loss=torch.tensor(2.0),
            source_weighted_sdpo_loss=torch.tensor(0.4),
            mean_sdpo_weight=torch.tensor(0.2),
        )
        self.assertAlmostEqual(loss.item(), 2.0, places=6)


class ReasoningTokenMaskTest(unittest.TestCase):
    def test_masks_answer_and_keeps_unfinished_reasoning(self):
        token_ids = torch.tensor(
            [
                [1, 2, 7, 8, 9, 0],
                [1, 2, 3, 4, 0, 0],
                [7, 8, 1, 2, 0, 0],
            ]
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0],
            ]
        )
        result = build_token_mask_before_subsequence(
            token_ids=token_ids,
            attention_mask=attention_mask,
            delimiter_token_ids=[7, 8],
        )
        self.assertTrue(
            torch.equal(
                result,
                torch.tensor(
                    [
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            )
        )


class JointTeacherContextTest(unittest.TestCase):
    def test_leave_one_out_never_selects_self(self):
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        success_by_uid = {"group": [0]}
        solution = trainer._get_solution(
            idx=0,
            success_by_uid=success_by_uid,
            uids=["group"],
            response_texts=["self response"],
            dont_reprompt_on_self_success=True,
        )
        self.assertIsNone(solution)

    def test_leave_one_out_selects_distinct_sibling(self):
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        success_by_uid = {"group": [0, 1]}
        solution = trainer._get_solution(
            idx=0,
            success_by_uid=success_by_uid,
            uids=["group", "group"],
            response_texts=["self response", "sibling response"],
            dont_reprompt_on_self_success=True,
        )
        self.assertEqual(solution, "sibling response")


if __name__ == "__main__":
    unittest.main()
