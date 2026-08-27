#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
export REPO_DIR
export RUN_SCRIPT=${RUN_SCRIPT:-${REPO_DIR}/cell_annotation/cluster/run_llama31_8b_explicit_reasoning_sft_rjob.sh}
export BASE_MODEL=${BASE_MODEL:-/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Meta-Llama-3.1-8B-Instruct}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-llama31-8b-explicit-reasoning-sft-10ep-seed0}
export JOB_NAME=${JOB_NAME:-llama31-8b-explicit-rsft-10ep-seed0}
export DATA_ROOT=${DATA_ROOT:-${REPO_DIR}/cell_annotation/data/cello1_llama_explicit_reasoning_sft_reproduction}
export OUTPUT_DIR=${OUTPUT_DIR:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation/${EXPERIMENT_NAME}}

exec bash "${REPO_DIR}/cell_annotation/cluster/submit_qwen3_8b_nonthinking_reasoning_sft_rjob.sh" "$@"
