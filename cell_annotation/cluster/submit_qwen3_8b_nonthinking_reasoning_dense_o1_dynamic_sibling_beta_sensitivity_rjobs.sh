#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}

# beta=1.0 is the existing full-method run. Submit only the four missing
# sensitivity points. Save only the final step-540 checkpoint while retaining
# the original once-per-epoch validation schedule.
BETA_SPECS=(
    "0.25:0p25"
    "0.5:0p5"
    "1.5:1p5"
    "2.0:2p0"
)

for spec in "${BETA_SPECS[@]}"; do
    beta=${spec%%:*}
    beta_tag=${spec##*:}

    RUN_TS="${RUN_TS}" \
    EXPERIMENT_NAME="qwen3-8b-nt-rsft10ep-dense-sibling-o1-dw-beta${beta_tag}-k8-5ep-restart-v2-finalckpt" \
    JOB_NAME="qwen3-8b-nt-o1-dw-beta${beta_tag}-v2-final" \
    DYNAMIC_ENTROPY_BETA="${beta}" \
    SAVE_FREQ=540 \
    MAX_ACTOR_CKPT_TO_KEEP=1 \
        bash "${SCRIPT_DIR}/submit_qwen3_8b_nonthinking_reasoning_dense_o1_dynamic_sibling_rjob.sh"
done
