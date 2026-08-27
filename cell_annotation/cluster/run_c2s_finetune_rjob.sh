#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
BASE_MODEL=${BASE_MODEL:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/C2S/C2S-Pythia-410m-cell-type-prediction}
SOURCE_DATA_DIR=${SOURCE_DATA_DIR:-${REPO_DIR}/cell_annotation/data/processed}
C2S_DATA_DIR=${C2S_DATA_DIR:-${SOURCE_DATA_DIR}/c2s}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-c2s-ft-${RUN_TS}}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-c2s}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
PYTHON_BIN=${PYTHON_BIN:-python3}

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}/logs"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

exec > >(tee -a "${OUTPUT_DIR}/logs/train.log") 2>&1

"${PYTHON_BIN}" -c "import torch, transformers" || {
    echo "ERROR: C2S fine-tuning requires torch and transformers." >&2
    exit 1
}
[ -f "${BASE_MODEL}/config.json" ] || {
    echo "ERROR: offline C2S checkpoint is missing: ${BASE_MODEL}" >&2
    echo "Run cell_annotation/scripts/download_c2s_model.sh on the development machine first." >&2
    exit 1
}
[ -f "${SOURCE_DATA_DIR}/sft_train.jsonl" ] || {
    echo "ERROR: missing CellPuzzles fit batches: ${SOURCE_DATA_DIR}/sft_train.jsonl" >&2
    exit 1
}
[ -f "${SOURCE_DATA_DIR}/sft_dev.jsonl" ] || {
    echo "ERROR: missing CellPuzzles dev batches: ${SOURCE_DATA_DIR}/sft_dev.jsonl" >&2
    exit 1
}

if [ "${PREPARE_DATA:-1}" = "1" ] || [ ! -f "${C2S_DATA_DIR}/fit.jsonl" ]; then
    "${PYTHON_BIN}" -m cell_annotation.prepare_c2s_data \
        --fit-batches "${SOURCE_DATA_DIR}/sft_train.jsonl" \
        --dev-batches "${SOURCE_DATA_DIR}/sft_dev.jsonl" \
        --output-dir "${C2S_DATA_DIR}"
fi

OPTIONAL_ARGS=()
if [ "${RESPONSE_ONLY_LOSS:-0}" = "1" ]; then
    OPTIONAL_ARGS+=(--response-only-loss)
fi
if [ "${GRADIENT_CHECKPOINTING:-0}" = "1" ]; then
    OPTIONAL_ARGS+=(--gradient-checkpointing)
fi

"${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m cell_annotation.train_c2s \
    --model "${BASE_MODEL}" \
    --train-file "${C2S_DATA_DIR}/fit.jsonl" \
    --dev-file "${C2S_DATA_DIR}/dev.jsonl" \
    --output-dir "${OUTPUT_DIR}" \
    --max-length "${MAX_LENGTH:-1024}" \
    --epochs "${EPOCHS:-5}" \
    --learning-rate "${LEARNING_RATE:-1e-5}" \
    --per-device-batch-size "${PER_DEVICE_BATCH_SIZE:-8}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
    --warmup-ratio "${WARMUP_RATIO:-0.05}" \
    --weight-decay "${WEIGHT_DECAY:-0.0}" \
    --logging-steps "${LOGGING_STEPS:-50}" \
    --seed "${SEED:-42}" \
    --num-workers "${NUM_WORKERS:-4}" \
    "${OPTIONAL_ARGS[@]}" \
    "$@"

echo "[done] C2S fine-tuning output: ${OUTPUT_DIR}"
