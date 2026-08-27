#!/usr/bin/env bash
set -euo pipefail

# Llama 3.1 has no native thinking toggle.  We nevertheless supervise the
# visible o1-distilled trajectory as:
#   <reasoning>...</reasoning>
#   <answer>...</answer>
# using the model's standard assistant generation prompt.

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-llama31-8b-explicit-reasoning-sft-10ep-seed0}

export REPO_DIR RUN_TS EXPERIMENT_NAME
export BASE_MODEL=${BASE_MODEL:-/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Meta-Llama-3.1-8B-Instruct}
export RESPONSE_MODE=explicit_reasoning
export SFT_TARGET_MODE=explicit_reasoning
export DATA_ROOT=${DATA_ROOT:-${REPO_DIR}/cell_annotation/data/cello1_llama_explicit_reasoning_sft_reproduction}
export DATA_DIR=${DATA_DIR:-${DATA_ROOT}/processed}
export OUTPUT_DIR=${OUTPUT_DIR:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation/${EXPERIMENT_NAME}}
export EPOCHS=${EPOCHS:-10}

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN:-python3}" -m pytest -q \
    cell_annotation/tests/test_nonthinking_reasoning_sft.py \
    cell_annotation/tests/test_sft_lora_tokenization.py

exec bash "${REPO_DIR}/cell_annotation/cluster/run_qwen3_8b_cello1_paper_reasoning_sft_rjob.sh" "$@"
