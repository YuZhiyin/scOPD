#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
MODEL_PATH=${MODEL_PATH:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation/qwen3-8b-nt-rsft10ep-dense-sibling-o1-dw-k8-5ep-restart-v2/merged_model_step540}

# M=8 has already been evaluated with the same checkpoint and DGPR settings.
M_VALUES=(2 4 6 10)

for m in "${M_VALUES[@]}"; do
    RUN_TS="${RUN_TS}" \
    JOB_NAME="qwen3-8b-nt-o1-dw-dgpr-m${m}" \
    MODEL_PATH="${MODEL_PATH}" \
    NUM_ROLLOUTS="${m}" \
    OUTPUT_DIR="${REPO_DIR}/outputs/cell_annotation/eval_qwen3_8b_nt_dense_sibling_o1_dw_step540_dgpr_m${m}_${RUN_TS}" \
        bash "${SCRIPT_DIR}/submit_qwen3_8b_dgpr_eval_rjob.sh"
done
