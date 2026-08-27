#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-8b-paper10ep-sparse-grpo-k8-1ep-seed0}
JOB_NAME=${JOB_NAME:-qwen3-8b-sparse-grpo-k8-1ep-v1}

export EXPERIMENT_NAME JOB_NAME
export STUDENT_MODEL=${STUDENT_MODEL:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation/qwen3-8b-cello1-paper-reasoning-sft-seed0/merged_model}
export REWARD_FN_NAME=${REWARD_FN_NAME:-compute_sparse_score}
export ROLLOUT_N=${ROLLOUT_N:-8}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export SAVE_FREQ=${SAVE_FREQ:-108}
export TEST_FREQ=${TEST_FREQ:-108}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}

exec bash "${SCRIPT_DIR}/submit_qwen3_8b_cello1_grpo_rjob.sh" "$@"
