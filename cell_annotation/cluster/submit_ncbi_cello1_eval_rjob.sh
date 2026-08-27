#!/usr/bin/env bash
set -euo pipefail

# Native Cell-o1 evaluation. Cell-o1 is a Qwen2-family model trained to emit
# <think>...</think><answer>...</answer>; it has no Qwen3 thinking-mode switch.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}

export MODEL_PATH=${MODEL_PATH:-/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/cell-o1}
export JOB_NAME=${JOB_NAME:-ncbi-cello1-native-eval-${RUN_TS}}
export OUTPUT_DIR=${OUTPUT_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO/outputs/cell_annotation/eval_ncbi_cello1_native_${RUN_TS}}
export GPU=${GPU:-2}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
export ENABLE_THINKING=0
export NONTHINKING_REASONING=0
export NATIVE_CELLO1=1

exec bash "${SCRIPT_DIR}/submit_qwen3_8b_eval_rjob.sh" "$@"
