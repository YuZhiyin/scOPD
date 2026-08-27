#!/usr/bin/env bash
set -euo pipefail

# Qwen3 hard non-thinking template with an explicit, supervised rationale:
#   <reasoning>marker/context/global-matching rationale</reasoning>
#   <answer>ordered cell types</answer>
# The optimizer, data split, and 10-epoch schedule remain aligned with the
# Cell-o1 paper SFT recipe.

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-8b-nonthinking-reasoning-sft-10ep-seed0}

export REPO_DIR RUN_TS EXPERIMENT_NAME
export RESPONSE_MODE=nonthinking_reasoning
export SFT_TARGET_MODE=nonthinking_reasoning
export DATA_ROOT=${DATA_ROOT:-${REPO_DIR}/cell_annotation/data/cello1_nonthinking_reasoning_sft_reproduction}
export DATA_DIR=${DATA_DIR:-${DATA_ROOT}/processed}
export OUTPUT_DIR=${OUTPUT_DIR:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation/${EXPERIMENT_NAME}}
export EPOCHS=${EPOCHS:-10}

exec bash "${REPO_DIR}/cell_annotation/cluster/run_qwen3_8b_cello1_paper_reasoning_sft_rjob.sh" "$@"
