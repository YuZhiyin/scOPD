#!/usr/bin/env bash
set -euo pipefail

# Answer-only SFT for Qwen3-8B with native thinking disabled.
#
# The supervision target is exactly:
#   <answer>cell type 1 | cell type 2 | ...</answer>
# Prompt tokens (including Qwen3's empty non-thinking prefill) are masked, so
# the loss is computed only on the answer continuation.
#
# Optimizer and LoRA defaults follow the Cell-o1 paper-aligned SFT recipe used
# by this repository. Unlike the reasoning-SFT reproduction, this run uses all
# 6,912 official CellPuzzles train examples (6,566 train / 346 dev).

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
BASE_MODEL=${BASE_MODEL:-/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-8B}
DATA_DIR=${DATA_DIR:-${REPO_DIR}/cell_annotation/data/processed}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-8b-nonthinking-answer-only-sft-fulltrain-10ep-seed42}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation/${EXPERIMENT_NAME}}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
PYTHON_BIN=${PYTHON_BIN:-python3}

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/wandb"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR="${OUTPUT_DIR}/wandb"

exec > >(tee -a "${OUTPUT_DIR}/logs/answer_only_sft.log") 2>&1

echo "[preflight] Python and training package versions"
"${PYTHON_BIN}" -c \
    'import sys, transformers, peft, torch; print(sys.version); print("torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__)'

[ -s "${BASE_MODEL}/config.json" ] || {
    echo "ERROR: invalid base model path: ${BASE_MODEL}" >&2
    exit 1
}

TRAIN_FILE="${DATA_DIR}/sft_train.jsonl"
DEV_FILE="${DATA_DIR}/sft_dev.jsonl"
for path in "${TRAIN_FILE}" "${DEV_FILE}"; do
    [ -s "${path}" ] || {
        echo "ERROR: missing answer-only SFT data: ${path}" >&2
        exit 1
    }
done

TRAIN_COUNT=$(wc -l < "${TRAIN_FILE}")
DEV_COUNT=$(wc -l < "${DEV_FILE}")
if [ "${TRAIN_COUNT}" -ne 6566 ] || [ "${DEV_COUNT}" -ne 346 ]; then
    echo "ERROR: expected 6,566 train and 346 dev examples; got ${TRAIN_COUNT} and ${DEV_COUNT}" >&2
    exit 1
fi

EPOCHS=${EPOCHS:-10}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-16}
MAX_LENGTH=${MAX_LENGTH:-8192}
LORA_R=${LORA_R:-256}
LORA_ALPHA=${LORA_ALPHA:-512}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
TRAIN_SEED=${TRAIN_SEED:-42}
USE_DORA=${USE_DORA:-0}
GLOBAL_MICRO_BATCH=$((NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE))
EFFECTIVE_BATCH=$((GLOBAL_MICRO_BATCH * GRADIENT_ACCUMULATION_STEPS))

if [ "${GLOBAL_MICRO_BATCH}" -ne 8 ] || [ "${EFFECTIVE_BATCH}" -ne 128 ]; then
    echo "ERROR: expected global micro batch 8 and effective batch 128;" >&2
    echo "       resolved values are ${GLOBAL_MICRO_BATCH} and ${EFFECTIVE_BATCH}" >&2
    exit 1
fi

{
    echo "base_model=${BASE_MODEL}"
    echo "train_file=${TRAIN_FILE}"
    echo "dev_file=${DEV_FILE}"
    echo "train_examples=${TRAIN_COUNT}"
    echo "dev_examples=${DEV_COUNT}"
    echo "prompt_mode=nonthinking"
    echo "enable_thinking=false"
    echo "target_mode=answer"
    echo "target_format=<answer>...</answer>"
    echo "completion_only_loss=true"
    echo "epochs=${EPOCHS}"
    echo "learning_rate=${LEARNING_RATE}"
    echo "max_length=${MAX_LENGTH}"
    echo "world_size=${NPROC_PER_NODE}"
    echo "per_device_batch_size=${PER_DEVICE_BATCH_SIZE}"
    echo "global_micro_batch=${GLOBAL_MICRO_BATCH}"
    echo "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
    echo "effective_batch=${EFFECTIVE_BATCH}"
    echo "lora_r=${LORA_R}"
    echo "lora_alpha=${LORA_ALPHA}"
    echo "lora_dropout=${LORA_DROPOUT}"
    echo "use_dora=${USE_DORA}"
    echo "lr_scheduler_type=linear"
    echo "warmup_ratio=0"
    echo "eval_strategy=epoch"
    echo "save_strategy=epoch"
    echo "save_only_model=${SAVE_ONLY_MODEL:-0}"
    echo "save_total_limit=${SAVE_TOTAL_LIMIT:-2}"
    echo "resume_from_checkpoint=${RESUME_FROM_CHECKPOINT:-}"
    echo "train_seed=${TRAIN_SEED}"
} | tee "${OUTPUT_DIR}/resolved_config.txt"

SFT_EXTRA_ARGS=()
if [ "${USE_DORA}" = "1" ]; then
    SFT_EXTRA_ARGS+=(--use-dora)
fi
if [ "${REPORT_TO_WANDB:-0}" = "1" ]; then
    SFT_EXTRA_ARGS+=(--report-to-wandb)
fi
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
    --train-file "${TRAIN_FILE}" \
    --dev-file "${DEV_FILE}" \
    --output-dir "${OUTPUT_DIR}/lora" \
    --target-mode answer \
    --max-length "${MAX_LENGTH}" \
    --epochs "${EPOCHS}" \
    --learning-rate "${LEARNING_RATE}" \
    --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --eval-strategy epoch \
    --save-strategy epoch \
    --save-total-limit "${SAVE_TOTAL_LIMIT:-2}" \
    --lr-scheduler-type linear \
    --warmup-ratio 0 \
    --lora-r "${LORA_R}" \
    --lora-alpha "${LORA_ALPHA}" \
    --lora-dropout "${LORA_DROPOUT}" \
    --seed "${TRAIN_SEED}" \
    --fail-on-target-truncation \
    "${SFT_EXTRA_ARGS[@]}" \
    "$@"

if [ "${MERGE_AFTER_SFT:-1}" = "1" ]; then
    "${PYTHON_BIN}" -m cell_annotation.merge_lora \
        --base-model "${BASE_MODEL}" \
        --adapter "${OUTPUT_DIR}/lora/final_adapter" \
        --output-dir "${OUTPUT_DIR}/merged_model"
fi

echo "[done] Non-thinking answer-only SFT output: ${OUTPUT_DIR}"
