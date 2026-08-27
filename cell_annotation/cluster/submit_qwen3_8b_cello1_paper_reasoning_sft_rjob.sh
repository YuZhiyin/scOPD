#!/usr/bin/env bash
set -euo pipefail

if [ -z "${KUBEBRAIN_CLUSTER_ENTRY:-}" ] || [ -z "${KUBEBRAIN_NAMESPACE:-}" ]; then
    if [ -r /etc/profile.d/ssh-init.sh ]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/ssh-init.sh
    else
        echo "ERROR: missing Kubebrain environment and /etc/profile.d/ssh-init.sh" >&2
        exit 1
    fi
fi

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
RUN_SCRIPT="${REPO_DIR}/cell_annotation/cluster/run_qwen3_8b_cello1_paper_reasoning_sft_rjob.sh"
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-8b-cello1-paper-reasoning-sft-${RUN_TS}}
JOB_NAME=${JOB_NAME:-${EXPERIMENT_NAME}}
GPU=${GPU:-8}
CPU=${CPU:-128}
MEMORY=${MEMORY:-300000}
CHARGED_GROUP=${CHARGED_GROUP:-ma4tool_gpu}
PRIVATE_MACHINE=${PRIVATE_MACHINE:-group}
IMAGE=${IMAGE:-registry.h.pjlab.org.cn/ailab-ma4tool-ma4tool_gpu/yuzhiyin-workspace:20260709115129-verl}

EXTRA_ARGS=()
if [ -n "${PRIVATE_MACHINE}" ]; then
    EXTRA_ARGS+=(--private-machine="${PRIVATE_MACHINE}")
fi

CMD=(
    rjob submit
    --name="${JOB_NAME}"
    --delete
    --enable-sshd
    --gpu="${GPU}"
    --memory="${MEMORY}"
    --cpu="${CPU}"
    --charged-group="${CHARGED_GROUP}"
    --mount=gpfs://gpfs1/ma4tool-shared:/mnt/shared-storage-user/ma4tool-shared
    --mount=gpfs://gpfs2/ma4tool-shared-2:/mnt/shared-storage-gpfs2/ma4tool-shared-2
    --mount=gpfs://gpfs1/yuzhiyin:/mnt/shared-storage-user/yuzhiyin
    --image="${IMAGE}"
    -P 1
    --gang-start=true
    --share-host-shm=True
    --custom-resources rdma/mlnx_shared=8
    --custom-resources mellanox.com/mlnx_rdma=1
    --host-network=true
    -e RUN_TS="${RUN_TS}"
    -e EXPERIMENT_NAME="${EXPERIMENT_NAME}"
    -e NPROC_PER_NODE="${GPU}"
    -e BASE_MODEL="${BASE_MODEL:-/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-8B}"
    -e DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/cell_annotation/data/cello1_reasoning_sft_reproduction}"
    -e SOURCE_RAW_DIR="${SOURCE_RAW_DIR:-${REPO_DIR}/cell_annotation/data/raw}"
    -e OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/cell_annotation/${EXPERIMENT_NAME}}"
    -e PYTHON_BIN="${PYTHON_BIN:-python3}"
    -e PREPARE_DATA="${PREPARE_DATA:-1}"
    -e DEV_RATIO="${DEV_RATIO:-0.10}"
    -e DATA_SEED="${DATA_SEED:-0}"
    -e TRAIN_SEED="${TRAIN_SEED:-42}"
    -e EPOCHS="${EPOCHS:-10}"
    -e LEARNING_RATE="${LEARNING_RATE:-5e-5}"
    -e PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
    -e GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
    -e MAX_LENGTH="${MAX_LENGTH:-8192}"
    -e LORA_R="${LORA_R:-256}"
    -e LORA_ALPHA="${LORA_ALPHA:-512}"
    -e LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
    -e USE_DORA="${USE_DORA:-0}"
    -e REPORT_TO_WANDB="${REPORT_TO_WANDB:-0}"
    -e SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-0}"
    -e SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
    -e RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
    -e MERGE_AFTER_SFT="${MERGE_AFTER_SFT:-1}"
    "${EXTRA_ARGS[@]}"
    -- bash -ex "${RUN_SCRIPT}"
)

printf 'Submitting:'
printf ' %q' "${CMD[@]}"
printf '\n'
[ "${DRY_RUN:-0}" = "1" ] || "${CMD[@]}"
