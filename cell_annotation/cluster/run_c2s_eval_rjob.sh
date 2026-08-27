#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
MODEL_PATH=${MODEL_PATH:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/C2S/C2S-Pythia-410m-cell-type-prediction}
DATA_DIR=${DATA_DIR:-${REPO_DIR}/cell_annotation/data/processed}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
RUN_LABEL=${RUN_LABEL:-zero}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_DIR}/outputs/c2s/eval_c2s_${RUN_LABEL}_${RUN_TS}}
PYTHON_BIN=${PYTHON_BIN:-python3}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_LENGTH=${MAX_LENGTH:-1024}
DTYPE=${DTYPE:-auto}

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}/predictions" "${OUTPUT_DIR}/metrics" "${OUTPUT_DIR}/logs"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

exec > >(tee -a "${OUTPUT_DIR}/logs/eval.log") 2>&1

"${PYTHON_BIN}" -c "import numpy, torch, transformers" || {
    echo "ERROR: C2S evaluation requires numpy, torch and transformers." >&2
    exit 1
}
[ -f "${MODEL_PATH}/config.json" ] || {
    echo "ERROR: offline C2S checkpoint is missing: ${MODEL_PATH}" >&2
    echo "Download it on the development machine before submitting this rjob." >&2
    exit 1
}
[ -f "${DATA_DIR}/train.json" ] || {
    echo "ERROR: missing train label reference: ${DATA_DIR}/train.json" >&2
    exit 1
}

OPTIONAL_ARGS=(--resume)
if [ -n "${LIMIT:-}" ]; then
    OPTIONAL_ARGS+=(--limit "${LIMIT}")
fi
if [ "${SUM_LOG_PROBS:-0}" = "1" ]; then
    OPTIONAL_ARGS+=(--sum-log-probs)
fi

INPUTS=(
    "${DATA_DIR}/test.json"
    "${DATA_DIR}/test_clean.json"
    "${DATA_DIR}/unseen_all.json"
    "${DATA_DIR}/unseen/breast_cancer.json"
    "${DATA_DIR}/unseen/colorectal_cancer.json"
    "${DATA_DIR}/unseen/melanoma.json"
    "${DATA_DIR}/unseen/systemic_lupus_erythematosus.json"
)

for input_path in "${INPUTS[@]}"; do
    if [ ! -f "${input_path}" ]; then
        echo "ERROR: missing evaluation file: ${input_path}" >&2
        exit 1
    fi
    split_name=$(basename "${input_path}" .json)
    prediction_path="${OUTPUT_DIR}/predictions/${split_name}.json"
    metric_path="${OUTPUT_DIR}/metrics/${split_name}.json"
    "${PYTHON_BIN}" -m cell_annotation.infer_c2s \
        --model "${MODEL_PATH}" \
        --input "${input_path}" \
        --output "${prediction_path}" \
        --batch-size "${BATCH_SIZE}" \
        --max-length "${MAX_LENGTH}" \
        --dtype "${DTYPE}" \
        --save-every "${SAVE_EVERY:-25}" \
        "${OPTIONAL_ARGS[@]}" \
        "$@"
    "${PYTHON_BIN}" -m cell_annotation.evaluate_c2s_cell_level \
        --predictions "${prediction_path}" \
        --train-reference "${DATA_DIR}/train.json" \
        --output "${metric_path}"
done

echo "[done] C2S cell-level evaluation output: ${OUTPUT_DIR}"
