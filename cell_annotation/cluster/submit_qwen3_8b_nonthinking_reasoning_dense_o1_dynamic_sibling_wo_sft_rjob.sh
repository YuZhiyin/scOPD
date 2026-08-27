#!/usr/bin/env bash
set -euo pipefail

# Ablation: remove reasoning SFT initialization while keeping the complete
# dense-reward, sibling-first/O1-fallback, dynamically weighted GRPO/SDPO run.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-8b-nosft-nt-dense-sibling-o1-dw-k8-5ep-seed0}
export JOB_NAME=${JOB_NAME:-qwen3-8b-nosft-nt-dense-sibling-o1-dw}
export STUDENT_MODEL=${STUDENT_MODEL:-/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-8B}

exec bash "${SCRIPT_DIR}/submit_qwen3_8b_nonthinking_reasoning_dense_o1_dynamic_sibling_rjob.sh" "$@"
