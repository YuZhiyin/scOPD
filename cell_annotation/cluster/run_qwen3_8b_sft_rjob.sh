#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
BASE_MODEL=${BASE_MODEL:-/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-8B}
DATA_DIR=${DATA_DIR:-${REPO_DIR}/cell_annotation/data/processed}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-cell-qwen3-8b-format-sft-${RUN_TS}}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_DIR}/outputs/cell_annotation/${EXPERIMENT_NAME}}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
PYTHON_BIN=${PYTHON_BIN:-python3}

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/wandb"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR="${OUTPUT_DIR}/wandb"

exec > >(tee -a "${OUTPUT_DIR}/logs/sft.log") 2>&1

"${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m cell_annotation.sft_lora \
    --model "${BASE_MODEL}" \
    --train-file "${DATA_DIR}/sft_train.jsonl" \
    --dev-file "${DATA_DIR}/sft_dev.jsonl" \
    --output-dir "${OUTPUT_DIR}/lora" \
    --max-length "${MAX_LENGTH:-4096}" \
    --epochs "${EPOCHS:-1}" \
    --learning-rate "${LEARNING_RATE:-2e-4}" \
    --per-device-batch-size "${PER_DEVICE_BATCH_SIZE:-1}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
    "$@"

if [ "${MERGE_AFTER_SFT:-1}" = "1" ]; then
    "${PYTHON_BIN}" -m cell_annotation.merge_lora \
        --base-model "${BASE_MODEL}" \
        --adapter "${OUTPUT_DIR}/lora/final_adapter" \
        --output-dir "${OUTPUT_DIR}/merged_model"
fi

echo "[done] SFT output: ${OUTPUT_DIR}"

