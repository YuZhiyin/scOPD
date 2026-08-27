#!/usr/bin/env bash
set -euo pipefail

# Cell-o1 reasoning-SFT reproduction with Qwen3-8B as the only model change.
#
# Original Cell-o1 SFT settings retained:
#   reasoning split, 90/10 train/dev, seed 0, completion-only loss,
#   LoRA/DoRA r=256 alpha=512 dropout=0.05, lr=5e-5, one epoch,
#   max length 4096, linear schedule without warmup, global batch 96.
#
# Qwen3-specific adaptation:
#   enable_thinking=True and 8-way DDP with micro batch 1 x grad accum 12,
#   which preserves the original effective global batch size of 96.

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
BASE_MODEL=${BASE_MODEL:-/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-8B}
DATA_ROOT=${DATA_ROOT:-${REPO_DIR}/cell_annotation/data/cello1_reasoning_sft_reproduction}
DATA_DIR=${DATA_DIR:-${DATA_ROOT}/processed}
SOURCE_RAW_DIR=${SOURCE_RAW_DIR:-${REPO_DIR}/cell_annotation/data/raw}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-8b-cello1-reasoning-sft-${RUN_TS}}
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
        --source-raw-dir "${SOURCE_RAW_DIR}" \
        --dev-ratio "${DEV_RATIO:-0.10}" \
        --seed "${SEED:-0}"
fi

for path in \
    "${DATA_DIR}/sft_reasoning_train.jsonl" \
    "${DATA_DIR}/sft_reasoning_dev.jsonl"; do
    [ -s "${path}" ] || {
        echo "ERROR: missing Cell-o1 reasoning SFT data: ${path}" >&2
        exit 1
    }
done

SFT_EXTRA_ARGS=()
if [ "${SAVE_ONLY_MODEL:-0}" = "1" ]; then
    SFT_EXTRA_ARGS+=(--save-only-model)
fi
if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
    SFT_EXTRA_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

"${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m cell_annotation.sft_lora \
    --model "${BASE_MODEL}" \
    --train-file "${DATA_DIR}/sft_reasoning_train.jsonl" \
    --dev-file "${DATA_DIR}/sft_reasoning_dev.jsonl" \
    --output-dir "${OUTPUT_DIR}/lora" \
    --target-mode reasoning \
    --max-length "${MAX_LENGTH:-4096}" \
    --epochs "${EPOCHS:-1}" \
    --learning-rate "${LEARNING_RATE:-5e-5}" \
    --per-device-batch-size "${PER_DEVICE_BATCH_SIZE:-1}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-12}" \
    --eval-strategy epoch \
    --save-strategy epoch \
    --save-total-limit "${SAVE_TOTAL_LIMIT:-2}" \
    --lr-scheduler-type linear \
    --warmup-ratio 0 \
    --lora-r "${LORA_R:-256}" \
    --lora-alpha "${LORA_ALPHA:-512}" \
    --lora-dropout "${LORA_DROPOUT:-0.05}" \
    --use-dora \
    "${SFT_EXTRA_ARGS[@]}" \
    "$@"

if [ "${MERGE_AFTER_SFT:-1}" = "1" ]; then
    "${PYTHON_BIN}" -m cell_annotation.merge_lora \
        --base-model "${BASE_MODEL}" \
        --adapter "${OUTPUT_DIR}/lora/final_adapter" \
        --output-dir "${OUTPUT_DIR}/merged_model"
fi

echo "[done] Cell-o1-aligned Qwen3 reasoning SFT output: ${OUTPUT_DIR}"
