#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-8b-nt-rsft10ep-joint-dense-gt-dw-lam03-k8-5ep-restart-v2}
JOB_NAME=${JOB_NAME:-qwen3-8b-nt-joint-gt-lam03-v2}

export EXPERIMENT_NAME JOB_NAME
export STUDENT_MODEL=${STUDENT_MODEL:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation/qwen3-8b-nonthinking-reasoning-sft-10ep-seed0/merged_model}
export DATA_DIR=${DATA_DIR:-${REPO_DIR}/cell_annotation/data/rl_nonthinking_reasoning_o1_privileged/processed}
export CONFIG_NAME=${CONFIG_NAME:-nonthinking_reasoning_joint_gt_sdpo}
export REWARD_FN_NAME=${REWARD_FN_NAME:-compute_nonthinking_reasoning_joint_gt_score}
export ENABLE_THINKING=false
export LOSS_MODE=joint_grpo_sdpo
export JOINT_LOSS_WEIGHT=${JOINT_LOSS_WEIGHT:-0.3}
export INCLUDE_ENVIRONMENT_FEEDBACK=False
export ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION=True
export ALLOW_ENVIRONMENT_FEEDBACK_FALLBACK=False
export DYNAMIC_ENTROPY_WEIGHTING=True
export DYNAMIC_ENTROPY_BETA=${DYNAMIC_ENTROPY_BETA:-1.0}
export ROLLOUT_N=${ROLLOUT_N:-8}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
export SAVE_FREQ=${SAVE_FREQ:-108}
export TEST_FREQ=${TEST_FREQ:-108}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-6}

exec bash "${SCRIPT_DIR}/submit_qwen3_8b_sibling_srpo_rjob.sh" "$@"
