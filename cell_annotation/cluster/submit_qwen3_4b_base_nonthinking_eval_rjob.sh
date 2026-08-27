#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}

export REPO_DIR RUN_TS
export JOB_NAME=${JOB_NAME:-qwen-4b-base-nt-eval}
export GPU=2
export MODEL_PATH=${MODEL_PATH:-/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-4B/Qwen3-4B}
export OUTPUT_DIR=${OUTPUT_DIR:-${REPO_DIR}/outputs/cell_annotation/eval_qwen_4b_base_nonthinking_${RUN_TS}}
export ENABLE_THINKING=0
export NONTHINKING_REASONING=0
export NATIVE_CELLO1=0

exec bash "${REPO_DIR}/cell_annotation/cluster/submit_qwen3_8b_eval_rjob.sh" "$@"
