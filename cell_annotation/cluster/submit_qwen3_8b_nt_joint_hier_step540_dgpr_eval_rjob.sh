#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
export RUN_TS
export JOB_NAME=${JOB_NAME:-qwen3-8b-nt-joint-hier-dgpr-m8}
export MODEL_PATH=${MODEL_PATH:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation/qwen3-8b-nt-rsft10ep-joint-dense-hier-dw-lam03-k8-5ep-seed0/merged_model_step540}
export OUTPUT_DIR=${OUTPUT_DIR:-${REPO_DIR}/outputs/cell_annotation/eval_qwen3_8b_nt_joint_hier_step540_dgpr_m8_${RUN_TS}}

exec bash "${SCRIPT_DIR}/submit_qwen3_8b_dgpr_eval_rjob.sh" "$@"

