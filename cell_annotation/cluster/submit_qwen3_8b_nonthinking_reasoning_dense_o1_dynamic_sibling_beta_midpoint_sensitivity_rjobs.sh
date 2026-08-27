#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}

# Additional midpoint settings for the dynamic-entropy beta sensitivity study.
# Keep every other setting aligned with the completed beta={0.25,0.5,1.5,2.0}
# runs: K=8, five epochs, validation once per epoch, and final-only checkpoint.
BETA_SPECS=(
    "1.25:1p25"
    "1.75:1p75"
)

for spec in "${BETA_SPECS[@]}"; do
    beta=${spec%%:*}
    beta_tag=${spec##*:}

    RUN_TS="${RUN_TS}" \
    PRIORITY=5 \
    EXPERIMENT_NAME="qwen3-8b-nt-rsft10ep-dense-sibling-o1-dw-beta${beta_tag}-k8-5ep-restart-v2-finalckpt" \
    JOB_NAME="qwen3-8b-nt-o1-dw-beta${beta_tag}-v2-final" \
    DYNAMIC_ENTROPY_BETA="${beta}" \
    SAVE_FREQ=540 \
    MAX_ACTOR_CKPT_TO_KEEP=1 \
        bash "${SCRIPT_DIR}/submit_qwen3_8b_nonthinking_reasoning_dense_o1_dynamic_sibling_rjob.sh"
done
