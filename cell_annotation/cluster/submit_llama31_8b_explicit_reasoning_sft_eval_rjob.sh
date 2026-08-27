#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}

export REPO_DIR RUN_TS
export MODEL_PATH=${MODEL_PATH:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation/llama31-8b-explicit-reasoning-sft-4ep-seed0/merged_model}
export JOB_NAME=${JOB_NAME:-llama31-8b-explicit-rsft4ep-eval-p5-v1}
export OUTPUT_DIR=${OUTPUT_DIR:-${REPO_DIR}/outputs/cell_annotation/eval_llama31_8b_explicit_reasoning_sft_4ep_${RUN_TS}}
export PRIORITY=${PRIORITY:-5}
export GPU=${GPU:-2}
export ENABLE_THINKING=0
export NONTHINKING_REASONING=1
export NATIVE_CELLO1=0
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}

exec bash "${REPO_DIR}/cell_annotation/cluster/submit_qwen3_8b_eval_rjob.sh" "$@"
