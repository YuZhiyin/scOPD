#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
TRAIN_ROOT=${TRAIN_ROOT:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation/qwen3-8b-nt-rsft10ep-dense-sibling-o1-dw-beta0p75-k8-5ep-restart-v2-finalckpt}

export REPO_DIR RUN_TS
export PRIORITY=${PRIORITY:-5}
export JOB_NAME=${JOB_NAME:-qwen3-8b-nt-o1-dw-beta0p75-dgpr-m8}
export FSDP_ACTOR_DIR=${FSDP_ACTOR_DIR:-${TRAIN_ROOT}/checkpoints/global_step_540/actor}
export MERGED_MODEL_DIR=${MERGED_MODEL_DIR:-${TRAIN_ROOT}/merged_model_step540}
export OUTPUT_DIR=${OUTPUT_DIR:-${REPO_DIR}/outputs/cell_annotation/eval_qwen3_8b_nt_dense_sibling_o1_dw_beta0p75_step540_dgpr_m8_${RUN_TS}}

exec bash "${SCRIPT_DIR}/submit_qwen3_8b_dgpr_eval_rjob.sh" "$@"
