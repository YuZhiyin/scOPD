#!/usr/bin/env bash
set -euo pipefail

# Paper-aligned Cell-o1 reasoning SFT with Qwen3-8B as the backbone.
#
# Cell-o1 Appendix B.3:
#   3,912 o1-distilled examples; 90/10 train/dev; 10 epochs;
#   learning rate 5e-5; micro batch 8; gradient accumulation 16;
#   LoRA r=256 on all linear layers; max sequence length 8192;
#   evaluation at the end of every epoch.
#
# Distributed equivalence:
#   8 DDP ranks x per-device batch 1 = global micro batch 8.
#   Global effective batch = 8 x 16 = 128.
#
# Qwen3 adaptation defaults to native thinking for backward compatibility.
# RESPONSE_MODE/SFT_TARGET_MODE can select explicit <reasoning> supervision
# under enable_thinking=False without changing the paper-aligned optimizer.

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
BASE_MODEL=${BASE_MODEL:-/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-8B}
DATA_ROOT=${DATA_ROOT:-${REPO_DIR}/cell_annotation/data/cello1_reasoning_sft_reproduction}
DATA_DIR=${DATA_DIR:-${DATA_ROOT}/processed}
SOURCE_RAW_DIR=${SOURCE_RAW_DIR:-${REPO_DIR}/cell_annotation/data/raw}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-8b-cello1-paper-reasoning-sft-${RUN_TS}}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_DIR}/outputs/cell_annotation/${EXPERIMENT_NAME}}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
PYTHON_BIN=${PYTHON_BIN:-python3}
RESPONSE_MODE=${RESPONSE_MODE:-thinking}
SFT_TARGET_MODE=${SFT_TARGET_MODE:-reasoning}

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/wandb"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR="${OUTPUT_DIR}/wandb"

exec > >(tee -a "${OUTPUT_DIR}/logs/reasoning_sft.log") 2>&1

echo "[preflight] Python and training package versions"
"${PYTHON_BIN}" -c \
    'import sys, transformers, peft, torch; print(sys.version); print("torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__)'

[ -s "${BASE_MODEL}/config.json" ] || {
    echo "ERROR: invalid base model path: ${BASE_MODEL}" >&2
    exit 1
}

if [ "${PREPARE_DATA:-1}" = "1" ]; then
    "${PYTHON_BIN}" -m cell_annotation.prepare_reasoning_sft \
        --data-root "${DATA_ROOT}" \
        --source-raw-dir "${SOURCE_RAW_DIR}" \
        --dev-ratio "${DEV_RATIO:-0.10}" \
        --seed "${DATA_SEED:-0}" \
        --response-mode "${RESPONSE_MODE}"
fi

TRAIN_FILE="${DATA_DIR}/sft_reasoning_train.jsonl"
DEV_FILE="${DATA_DIR}/sft_reasoning_dev.jsonl"
for path in "${TRAIN_FILE}" "${DEV_FILE}"; do
    [ -s "${path}" ] || {
        echo "ERROR: missing Cell-o1 reasoning SFT data: ${path}" >&2
        exit 1
    }
done

TRAIN_COUNT=$(wc -l < "${TRAIN_FILE}")
DEV_COUNT=$(wc -l < "${DEV_FILE}")
if [ $((TRAIN_COUNT + DEV_COUNT)) -ne 3912 ]; then
    echo "ERROR: expected 3,912 Cell-o1 reasoning examples, got $((TRAIN_COUNT + DEV_COUNT))" >&2
    exit 1
fi

EPOCHS=${EPOCHS:-10}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-16}
MAX_LENGTH=${MAX_LENGTH:-8192}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
LORA_R=${LORA_R:-256}
LORA_ALPHA=${LORA_ALPHA:-512}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
TRAIN_SEED=${TRAIN_SEED:-42}
USE_DORA=${USE_DORA:-0}
GLOBAL_MICRO_BATCH=$((NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE))
EFFECTIVE_BATCH=$((GLOBAL_MICRO_BATCH * GRADIENT_ACCUMULATION_STEPS))

if [ "${GLOBAL_MICRO_BATCH}" -ne 8 ] || [ "${EFFECTIVE_BATCH}" -ne 128 ]; then
    echo "ERROR: paper alignment requires global micro batch 8 and effective batch 128;" >&2
    echo "       resolved values are ${GLOBAL_MICRO_BATCH} and ${EFFECTIVE_BATCH}" >&2
    exit 1
fi

{
    echo "base_model=${BASE_MODEL}"
    echo "train_examples=${TRAIN_COUNT}"
    echo "dev_examples=${DEV_COUNT}"
    echo "epochs=${EPOCHS}"
    echo "learning_rate=${LEARNING_RATE}"
    echo "max_length=${MAX_LENGTH}"
    echo "attn_implementation=${ATTN_IMPLEMENTATION}"
    echo "world_size=${NPROC_PER_NODE}"
    echo "per_device_batch_size=${PER_DEVICE_BATCH_SIZE}"
    echo "global_micro_batch=${GLOBAL_MICRO_BATCH}"
    echo "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
    echo "effective_batch=${EFFECTIVE_BATCH}"
    echo "lora_r=${LORA_R}"
    echo "lora_alpha=${LORA_ALPHA}"
    echo "lora_dropout=${LORA_DROPOUT}"
    echo "use_dora=${USE_DORA}"
    echo "response_mode=${RESPONSE_MODE}"
    echo "target_mode=${SFT_TARGET_MODE}"
    case "${SFT_TARGET_MODE}" in
        nonthinking_reasoning)
            echo "enable_thinking=false"
            echo "target_format=<reasoning>...</reasoning><answer>...</answer>"
            ;;
        explicit_reasoning)
            echo "enable_thinking=not_applicable"
            echo "chat_template_mode=standard"
            echo "target_format=<reasoning>...</reasoning><answer>...</answer>"
            ;;
        *)
            echo "enable_thinking=true"
            echo "target_format=<think>...</think><answer>...</answer>"
            ;;
    esac
    echo "completion_only_loss=true"
    echo "lr_scheduler_type=linear"
    echo "warmup_ratio=0"
    echo "eval_strategy=epoch"
    echo "save_strategy=epoch"
    echo "save_only_model=${SAVE_ONLY_MODEL:-0}"
    echo "save_total_limit=${SAVE_TOTAL_LIMIT:-2}"
    echo "resume_from_checkpoint=${RESUME_FROM_CHECKPOINT:-}"
    echo "data_seed=${DATA_SEED:-0}"
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
    --target-mode "${SFT_TARGET_MODE}" \
    --max-length "${MAX_LENGTH}" \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
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

echo "[done] Paper-aligned Qwen3 reasoning SFT output: ${OUTPUT_DIR}"
