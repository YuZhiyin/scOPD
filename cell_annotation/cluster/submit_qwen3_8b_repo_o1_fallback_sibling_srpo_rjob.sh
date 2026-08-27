#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-8b-paper10ep-repo-reward-sibling-o1-fallback-k8-5ep-seed0}
JOB_NAME=${JOB_NAME:-${EXPERIMENT_NAME}}

export EXPERIMENT_NAME JOB_NAME
export CONFIG_NAME=${CONFIG_NAME:-o1_fallback_sibling_srpo}
export DATA_DIR=${DATA_DIR:-${REPO_DIR}/cell_annotation/data/rl_thinking_o1_privileged/processed}
export REWARD_FN_NAME=${REWARD_FN_NAME:-compute_repo_o1_fallback_sibling_score}
export INCLUDE_ENVIRONMENT_FEEDBACK=${INCLUDE_ENVIRONMENT_FEEDBACK:-True}
export ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION=${ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION:-True}
export ALLOW_ENVIRONMENT_FEEDBACK_FALLBACK=${ALLOW_ENVIRONMENT_FEEDBACK_FALLBACK:-True}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
export TEST_FREQ=${TEST_FREQ:-108}
export SAVE_FREQ=${SAVE_FREQ:-540}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}

exec bash "${SCRIPT_DIR}/submit_qwen3_8b_sibling_srpo_rjob.sh" "$@"
