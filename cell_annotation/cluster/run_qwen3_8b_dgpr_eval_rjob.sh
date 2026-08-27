#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
MODEL_PATH=${MODEL_PATH:-}
FSDP_ACTOR_DIR=${FSDP_ACTOR_DIR:-}
MERGED_MODEL_DIR=${MERGED_MODEL_DIR:-}
DATA_DIR=${DATA_DIR:-${REPO_DIR}/cell_annotation/data/processed}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_DIR}/outputs/cell_annotation/eval_qwen3_8b_dgpr_${RUN_TS}}
PYTHON_BIN=${PYTHON_BIN:-python3}
TP_SIZE=${TP_SIZE:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
NUM_ROLLOUTS=${NUM_ROLLOUTS:-8}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.6}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.95}
REFINEMENT_TEMPERATURE=${REFINEMENT_TEMPERATURE:-0.2}
REFINEMENT_TOP_P=${REFINEMENT_TOP_P:-0.9}
CONSENSUS_THRESHOLD=${CONSENSUS_THRESHOLD:-0.75}
SEED=${SEED:-42}
LIMIT=${LIMIT:-}

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}/predictions" "${OUTPUT_DIR}/metrics" "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/raw"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
exec > >(tee -a "${OUTPUT_DIR}/logs/eval.log") 2>&1

# verl checkpoints contain FSDP model shards while actor/huggingface contains
# only model metadata and tokenizer files. Merge on demand and reuse the
# completed merge on evaluation retries.
shopt -s nullglob
if [ -n "${FSDP_ACTOR_DIR}" ]; then
    [ -s "${FSDP_ACTOR_DIR}/fsdp_config.json" ] || {
        echo "ERROR: missing FSDP checkpoint metadata: ${FSDP_ACTOR_DIR}/fsdp_config.json" >&2
        exit 1
    }
    [ -n "${MERGED_MODEL_DIR}" ] || {
        echo "ERROR: MERGED_MODEL_DIR is required with FSDP_ACTOR_DIR" >&2
        exit 1
    }
    merged_weights=(
        "${MERGED_MODEL_DIR}"/model*.safetensors
        "${MERGED_MODEL_DIR}"/pytorch_model*.bin
    )
    if [ ! -s "${MERGED_MODEL_DIR}/config.json" ] || [ "${#merged_weights[@]}" -eq 0 ]; then
        echo "[merge] FSDP actor: ${FSDP_ACTOR_DIR}"
        echo "[merge] target HF model: ${MERGED_MODEL_DIR}"
        "${PYTHON_BIN}" -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "${FSDP_ACTOR_DIR}" \
            --target_dir "${MERGED_MODEL_DIR}"
    else
        echo "[merge] reusing existing HF model: ${MERGED_MODEL_DIR}"
    fi
    MODEL_PATH="${MERGED_MODEL_DIR}"
fi

[ -n "${MODEL_PATH}" ] || {
    echo "ERROR: set MODEL_PATH or FSDP_ACTOR_DIR" >&2
    exit 1
}
[ -s "${MODEL_PATH}/config.json" ] || {
    echo "ERROR: missing model config: ${MODEL_PATH}/config.json" >&2
    exit 1
}
model_weights=("${MODEL_PATH}"/model*.safetensors "${MODEL_PATH}"/pytorch_model*.bin)
[ "${#model_weights[@]}" -gt 0 ] || {
    echo "ERROR: no model weights found under ${MODEL_PATH}" >&2
    exit 1
}

"${PYTHON_BIN}" -m pytest -q cell_annotation/tests/test_dgpr.py

INFER_ARGS=(
    --model "${MODEL_PATH}"
    --data-dir "${DATA_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --tensor-parallel-size "${TP_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --num-rollouts "${NUM_ROLLOUTS}"
    --rollout-temperature "${ROLLOUT_TEMPERATURE}"
    --rollout-top-p "${ROLLOUT_TOP_P}"
    --refinement-temperature "${REFINEMENT_TEMPERATURE}"
    --refinement-top-p "${REFINEMENT_TOP_P}"
    --consensus-threshold "${CONSENSUS_THRESHOLD}"
    --seed "${SEED}"
)
if [ -n "${LIMIT}" ]; then
    INFER_ARGS+=(--limit "${LIMIT}")
fi
"${PYTHON_BIN}" -m cell_annotation.infer_dgpr_vllm "${INFER_ARGS[@]}"

SPLITS=(test test_clean unseen_all breast_cancer colorectal_cancer melanoma systemic_lupus_erythematosus)
for split_name in "${SPLITS[@]}"; do
    anchor_path="${OUTPUT_DIR}/predictions/${split_name}_anchor_medoid.json"
    final_path="${OUTPUT_DIR}/predictions/${split_name}.json"
    "${PYTHON_BIN}" -m cell_annotation.evaluate \
        --predictions "${anchor_path}" \
        --output "${OUTPUT_DIR}/metrics/${split_name}_anchor_medoid.json"
    "${PYTHON_BIN}" -m cell_annotation.evaluate \
        --predictions "${final_path}" \
        --output "${OUTPUT_DIR}/metrics/${split_name}.json"
    "${PYTHON_BIN}" -m cell_annotation.evaluate_dgpr \
        --anchor "${anchor_path}" \
        --final "${final_path}" \
        --output "${OUTPUT_DIR}/metrics/${split_name}_dgpr_comparison.json"
done

echo "[done] DGPR evaluation output: ${OUTPUT_DIR}"
