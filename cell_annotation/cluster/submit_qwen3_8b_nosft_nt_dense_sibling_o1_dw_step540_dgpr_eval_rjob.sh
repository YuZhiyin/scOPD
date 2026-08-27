#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
TRAIN_ROOT=${TRAIN_ROOT:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation/qwen3-8b-nosft-nt-dense-sibling-o1-dw-k8-5ep-seed0}

export RUN_TS
export JOB_NAME=${JOB_NAME:-qwen3-8b-nosft-nt-o1-dw-step540-dgpr-m8}
export FSDP_ACTOR_DIR=${FSDP_ACTOR_DIR:-${TRAIN_ROOT}/checkpoints/global_step_540/actor}
export MERGED_MODEL_DIR=${MERGED_MODEL_DIR:-${TRAIN_ROOT}/merged_model_step540}
export OUTPUT_DIR=${OUTPUT_DIR:-${REPO_DIR}/outputs/cell_annotation/eval_qwen3_8b_nosft_nt_dense_sibling_o1_dw_step540_dgpr_m8_${RUN_TS}}
export PRIORITY=${PRIORITY:-5}
export GPU=${GPU:-2}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
export NUM_ROLLOUTS=${NUM_ROLLOUTS:-8}

exec bash "${SCRIPT_DIR}/submit_qwen3_8b_dgpr_eval_rjob.sh" "$@"
