#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH must point to a merged Hugging Face model}
DATA_DIR=${DATA_DIR:-${REPO_DIR}/cell_annotation/data/processed}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_DIR}/outputs/cell_annotation/eval_qwen3_8b_m1_refinement_${RUN_TS}}
PYTHON_BIN=${PYTHON_BIN:-python3}
TP_SIZE=${TP_SIZE:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
CRITIC_MAX_TOKENS=${CRITIC_MAX_TOKENS:-1024}
REFINEMENT_TEMPERATURE=${REFINEMENT_TEMPERATURE:-0.2}
REFINEMENT_TOP_P=${REFINEMENT_TOP_P:-0.9}
MIN_SUPPORTING_GENES=${MIN_SUPPORTING_GENES:-2}
SEED=${SEED:-42}
LIMIT=${LIMIT:-}

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}/predictions" "${OUTPUT_DIR}/metrics" "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/raw"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
exec > >(tee -a "${OUTPUT_DIR}/logs/eval.log") 2>&1

[ -s "${MODEL_PATH}/config.json" ] || {
    echo "ERROR: missing model config: ${MODEL_PATH}/config.json" >&2
    exit 1
}
shopt -s nullglob
model_weights=("${MODEL_PATH}"/model*.safetensors "${MODEL_PATH}"/pytorch_model*.bin)
[ "${#model_weights[@]}" -gt 0 ] || {
    echo "ERROR: no model weights found under ${MODEL_PATH}" >&2
    exit 1
}

"${PYTHON_BIN}" -m pytest -q cell_annotation/tests/test_m1_refinement.py

ARGS=(
    --model "${MODEL_PATH}"
    --data-dir "${DATA_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --tensor-parallel-size "${TP_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --critic-max-tokens "${CRITIC_MAX_TOKENS}"
    --refinement-temperature "${REFINEMENT_TEMPERATURE}"
    --refinement-top-p "${REFINEMENT_TOP_P}"
    --min-supporting-genes "${MIN_SUPPORTING_GENES}"
    --seed "${SEED}"
)
if [ -n "${LIMIT}" ]; then
    ARGS+=(--limit "${LIMIT}")
fi
"${PYTHON_BIN}" -m cell_annotation.infer_m1_refinement_vllm "${ARGS[@]}"

SPLITS=(test test_clean unseen_all breast_cancer colorectal_cancer melanoma systemic_lupus_erythematosus)
for split_name in "${SPLITS[@]}"; do
    initial_path="${OUTPUT_DIR}/predictions/${split_name}_initial.json"
    final_path="${OUTPUT_DIR}/predictions/${split_name}.json"
    "${PYTHON_BIN}" -m cell_annotation.evaluate \
        --predictions "${initial_path}" \
        --output "${OUTPUT_DIR}/metrics/${split_name}_initial.json"
    "${PYTHON_BIN}" -m cell_annotation.evaluate \
        --predictions "${final_path}" \
        --output "${OUTPUT_DIR}/metrics/${split_name}.json"
    "${PYTHON_BIN}" -m cell_annotation.evaluate_m1_refinement \
        --initial "${initial_path}" \
        --final "${final_path}" \
        --output "${OUTPUT_DIR}/metrics/${split_name}_m1_refinement_comparison.json"
done

echo "[done] M1 refinement evaluation output: ${OUTPUT_DIR}"

