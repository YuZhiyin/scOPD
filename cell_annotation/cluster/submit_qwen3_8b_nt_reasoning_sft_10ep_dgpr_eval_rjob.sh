#!/usr/bin/env bash
set -euo pipefail

# Ablation: remove the complete RL stage and run the same training-free DGPR
# evaluation directly from the 10-epoch non-thinking reasoning-SFT model.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
export RUN_TS
export JOB_NAME=${JOB_NAME:-qwen3-8b-nt-rsft10ep-dgpr-m8}
export MODEL_PATH=${MODEL_PATH:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation/qwen3-8b-nonthinking-reasoning-sft-10ep-seed0/merged_model}
export OUTPUT_DIR=${OUTPUT_DIR:-${REPO_DIR}/outputs/cell_annotation/eval_qwen3_8b_nt_reasoning_sft_10ep_dgpr_m8_${RUN_TS}}

exec bash "${SCRIPT_DIR}/submit_qwen3_8b_dgpr_eval_rjob.sh" "$@"
