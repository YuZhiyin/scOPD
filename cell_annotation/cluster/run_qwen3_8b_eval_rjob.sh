#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
MODEL_PATH=${MODEL_PATH:-/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-8B}
FSDP_ACTOR_DIR=${FSDP_ACTOR_DIR:-}
MERGED_MODEL_DIR=${MERGED_MODEL_DIR:-}
DATA_DIR=${DATA_DIR:-${REPO_DIR}/cell_annotation/data/processed}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_DIR}/outputs/cell_annotation/eval_qwen3_8b_${RUN_TS}}
PYTHON_BIN=${PYTHON_BIN:-python3}
TP_SIZE=${TP_SIZE:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1536}
ENABLE_THINKING=${ENABLE_THINKING:-0}
NONTHINKING_REASONING=${NONTHINKING_REASONING:-0}
NATIVE_CELLO1=${NATIVE_CELLO1:-0}

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}/predictions" "${OUTPUT_DIR}/metrics" "${OUTPUT_DIR}/logs"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

exec > >(tee -a "${OUTPUT_DIR}/logs/eval.log") 2>&1

# verl checkpoints keep the trained weights as FSDP shards; their
# actor/huggingface directory contains configuration and tokenizer files only.
# Optionally merge those shards before starting vLLM evaluation. Reusing a
# completed merge makes retries inexpensive.
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

[ -s "${MODEL_PATH}/config.json" ] || {
    echo "ERROR: missing model config: ${MODEL_PATH}/config.json" >&2
    exit 1
}
model_weights=("${MODEL_PATH}"/model*.safetensors "${MODEL_PATH}"/pytorch_model*.bin)
[ "${#model_weights[@]}" -gt 0 ] || {
    echo "ERROR: no model weights found under ${MODEL_PATH}" >&2
    exit 1
}

INPUTS=(
    "${DATA_DIR}/test.json"
    "${DATA_DIR}/test_clean.json"
    "${DATA_DIR}/unseen_all.json"
    "${DATA_DIR}/unseen/breast_cancer.json"
    "${DATA_DIR}/unseen/colorectal_cancer.json"
    "${DATA_DIR}/unseen/melanoma.json"
    "${DATA_DIR}/unseen/systemic_lupus_erythematosus.json"
)

OPTIONAL_ARGS=()
if [ "${ENABLE_THINKING}" = "1" ]; then
    OPTIONAL_ARGS+=(--enable-thinking)
fi
if [ "${NONTHINKING_REASONING}" = "1" ]; then
    if [ "${ENABLE_THINKING}" = "1" ]; then
        echo "ERROR: ENABLE_THINKING and NONTHINKING_REASONING are mutually exclusive" >&2
        exit 1
    fi
    OPTIONAL_ARGS+=(--nonthinking-reasoning)
fi
if [ "${NATIVE_CELLO1}" = "1" ]; then
    if [ "${ENABLE_THINKING}" = "1" ] || [ "${NONTHINKING_REASONING}" = "1" ]; then
        echo "ERROR: NATIVE_CELLO1 is mutually exclusive with other reasoning modes" >&2
        exit 1
    fi
    OPTIONAL_ARGS+=(--native-cello1)
fi

for input_path in "${INPUTS[@]}"; do
    if [ ! -f "${input_path}" ]; then
        echo "ERROR: missing evaluation file: ${input_path}" >&2
        exit 1
    fi
    split_name=$(basename "${input_path}" .json)
    prediction_path="${OUTPUT_DIR}/predictions/${split_name}.json"
    metric_path="${OUTPUT_DIR}/metrics/${split_name}.json"
    "${PYTHON_BIN}" -m cell_annotation.infer_vllm \
        --model "${MODEL_PATH}" \
        --input "${input_path}" \
        --output "${prediction_path}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        "${OPTIONAL_ARGS[@]}" \
        "$@"
    "${PYTHON_BIN}" -m cell_annotation.evaluate \
        --predictions "${prediction_path}" \
        --output "${metric_path}"
done

echo "[done] evaluation output: ${OUTPUT_DIR}"
