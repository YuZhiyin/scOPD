#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}

export REPO_DIR RUN_TS
export MODEL_PATH=${MODEL_PATH:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation/qwen3-8b-nonthinking-answer-only-sft-fulltrain-10ep-seed42/merged_model}
export JOB_NAME=${JOB_NAME:-qwen3-8b-nt-answer-sft-eval-${RUN_TS}}
export OUTPUT_DIR=${OUTPUT_DIR:-${REPO_DIR}/outputs/cell_annotation/eval_qwen3_8b_nonthinking_answer_only_sft_10ep_${RUN_TS}}

# Match the training protocol: Qwen3 native thinking is disabled and the model
# is expected to emit only <answer>...</answer>.  Keep the same evaluation
# context/response budget used for the other final non-thinking checkpoints.
export ENABLE_THINKING=0
export NONTHINKING_REASONING=0
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}

exec bash "${REPO_DIR}/cell_annotation/cluster/submit_qwen3_8b_eval_rjob.sh" "$@"
