#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}

export REPO_DIR
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-llama31-8b-explicit-rsft4ep-dense-sibling-o1-dw-k8-5ep-fixed-v2}
export JOB_NAME=${JOB_NAME:-llama31-8b-dense-sibling-o1-dw-p5-fixed-v2}
export PRIORITY=${PRIORITY:-5}
export STUDENT_MODEL=${STUDENT_MODEL:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation/llama31-8b-explicit-reasoning-sft-4ep-seed0/merged_model}
export DATA_DIR=${DATA_DIR:-${REPO_DIR}/cell_annotation/data/rl_explicit_reasoning_o1_privileged/processed}
export CONFIG_NAME=${CONFIG_NAME:-explicit_reasoning_o1_fallback_dynamic_entropy_sibling_srpo}
export REWARD_FN_NAME=${REWARD_FN_NAME:-compute_nonthinking_reasoning_repo_o1_fallback_sibling_score}

# Llama 3.1 has no native thinking-mode switch. Passing false is harmless for
# its standard chat template and keeps VERL's explicit-reasoning path aligned.
export ENABLE_THINKING=false
export INCLUDE_ENVIRONMENT_FEEDBACK=True
export ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION=True
export ALLOW_ENVIRONMENT_FEEDBACK_FALLBACK=True
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
