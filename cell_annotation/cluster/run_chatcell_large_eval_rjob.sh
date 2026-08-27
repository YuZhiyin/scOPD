#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
MODEL_PATH=${MODEL_PATH:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/chatcell/chatcell-large}
DATA_DIR=${DATA_DIR:-${REPO_DIR}/cell_annotation/data/processed}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_DIR}/outputs/cell_annotation/eval_chatcell_large_cell_level_${RUN_TS}}
PYTHON_BIN=${PYTHON_BIN:-python3}
BATCH_SIZE=${BATCH_SIZE:-64}
MAX_INPUT_LENGTH=${MAX_INPUT_LENGTH:-512}
MAX_TARGET_LENGTH=${MAX_TARGET_LENGTH:-32}
DTYPE=${DTYPE:-auto}
INCLUDE_CONTEXT=${INCLUDE_CONTEXT:-0}
INCLUDE_CANDIDATES=${INCLUDE_CANDIDATES:-1}
ASSIGNMENT_MODE=${ASSIGNMENT_MODE:-independent}

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}/predictions" "${OUTPUT_DIR}/metrics" "${OUTPUT_DIR}/logs"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

exec > >(tee -a "${OUTPUT_DIR}/logs/eval.log") 2>&1

"${PYTHON_BIN}" -c "import scipy, sentencepiece, torch, transformers" || {
    echo "ERROR: evaluation requires scipy, sentencepiece, torch and transformers." >&2
    exit 1
}

[ -f "${MODEL_PATH}/config.json" ] || {
    echo "ERROR: ChatCell model is missing: ${MODEL_PATH}" >&2
    echo "Run cell_annotation/scripts/download_chatcell_large.sh first." >&2
    exit 1
}

OPTIONAL_ARGS=()
if [ "${INCLUDE_CONTEXT}" = "1" ]; then
    OPTIONAL_ARGS+=(--include-context)
fi
if [ "${INCLUDE_CANDIDATES}" = "1" ]; then
    OPTIONAL_ARGS+=(--include-candidates)
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
    "${PYTHON_BIN}" -m cell_annotation.infer_chatcell \
        --model "${MODEL_PATH}" \
        --input "${input_path}" \
        --output "${prediction_path}" \
        --batch-size "${BATCH_SIZE}" \
        --max-input-length "${MAX_INPUT_LENGTH}" \
        --max-target-length "${MAX_TARGET_LENGTH}" \
        --dtype "${DTYPE}" \
        --assignment-mode "${ASSIGNMENT_MODE}" \
        "${OPTIONAL_ARGS[@]}" \
        "$@"
    if [ "${ASSIGNMENT_MODE}" = "independent" ]; then
        "${PYTHON_BIN}" -m cell_annotation.evaluate_chatcell_cell_level \
            --predictions "${prediction_path}" \
            --output "${metric_path}"
    else
        "${PYTHON_BIN}" -m cell_annotation.evaluate \
            --predictions "${prediction_path}" \
            --output "${metric_path}"
    fi
done

echo "[done] ChatCell evaluation output: ${OUTPUT_DIR}"
