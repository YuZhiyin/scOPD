#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-8b-paper10ep-repo-dw-sibling-k8-1ep-seed0}
JOB_NAME=${JOB_NAME:-qwen3-8b-repo-dw-sibling-k8-1ep-v1}

export EXPERIMENT_NAME JOB_NAME
export STUDENT_MODEL=${STUDENT_MODEL:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation/qwen3-8b-cello1-paper-reasoning-sft-seed0/merged_model}
export CONFIG_NAME=${CONFIG_NAME:-dynamic_entropy_sibling_srpo}
export REWARD_FN_NAME=${REWARD_FN_NAME:-compute_repo_sibling_score}
export DYNAMIC_ENTROPY_WEIGHTING=${DYNAMIC_ENTROPY_WEIGHTING:-True}
export DYNAMIC_ENTROPY_BETA=${DYNAMIC_ENTROPY_BETA:-1.0}
export INCLUDE_ENVIRONMENT_FEEDBACK=${INCLUDE_ENVIRONMENT_FEEDBACK:-False}
export ALLOW_ENVIRONMENT_FEEDBACK_FALLBACK=${ALLOW_ENVIRONMENT_FEEDBACK_FALLBACK:-False}
export ROLLOUT_N=${ROLLOUT_N:-8}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export SAVE_FREQ=${SAVE_FREQ:-108}
export TEST_FREQ=${TEST_FREQ:-108}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}

exec bash "${SCRIPT_DIR}/submit_qwen3_8b_sibling_srpo_rjob.sh" "$@"
