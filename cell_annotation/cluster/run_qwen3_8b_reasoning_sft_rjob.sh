#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
BASE_MODEL=${BASE_MODEL:-/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-8B}
DATA_ROOT=${DATA_ROOT:-${REPO_DIR}/cell_annotation/data}
DATA_DIR=${DATA_DIR:-${DATA_ROOT}/processed}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-cell-qwen3-8b-o1-reasoning-sft-${RUN_TS}}
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

exec > >(tee -a "${OUTPUT_DIR}/logs/reasoning_sft.log") 2>&1

if [ "${PREPARE_DATA:-1}" = "1" ]; then
    "${PYTHON_BIN}" -m cell_annotation.prepare_reasoning_sft \
        --data-root "${DATA_ROOT}" \
        --dev-ratio "${DEV_RATIO:-0.05}" \
        --seed "${SEED:-42}"
fi

for path in \
    "${DATA_DIR}/sft_reasoning_train.jsonl" \
    "${DATA_DIR}/sft_reasoning_dev.jsonl"; do
    [ -s "${path}" ] || {
        echo "ERROR: missing reasoning SFT data: ${path}" >&2
        exit 1
    }
done

"${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m cell_annotation.sft_lora \
    --model "${BASE_MODEL}" \
    --train-file "${DATA_DIR}/sft_reasoning_train.jsonl" \
    --dev-file "${DATA_DIR}/sft_reasoning_dev.jsonl" \
    --output-dir "${OUTPUT_DIR}/lora" \
    --target-mode reasoning \
    --max-length "${MAX_LENGTH:-6144}" \
    --epochs "${EPOCHS:-1}" \
    --learning-rate "${LEARNING_RATE:-5e-5}" \
    --per-device-batch-size "${PER_DEVICE_BATCH_SIZE:-1}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
    --eval-steps "${EVAL_STEPS:-25}" \
    --save-steps "${SAVE_STEPS:-25}" \
    --lora-r "${LORA_R:-64}" \
    --lora-alpha "${LORA_ALPHA:-128}" \
    --lora-dropout "${LORA_DROPOUT:-0.05}" \
    "$@"

if [ "${MERGE_AFTER_SFT:-1}" = "1" ]; then
    "${PYTHON_BIN}" -m cell_annotation.merge_lora \
        --base-model "${BASE_MODEL}" \
        --adapter "${OUTPUT_DIR}/lora/final_adapter" \
        --output-dir "${OUTPUT_DIR}/merged_model"
fi

echo "[done] o1 reasoning SFT output: ${OUTPUT_DIR}"
