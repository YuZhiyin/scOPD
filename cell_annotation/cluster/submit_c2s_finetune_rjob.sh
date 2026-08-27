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
RUN_SCRIPT="${REPO_DIR}/cell_annotation/cluster/run_c2s_finetune_rjob.sh"
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-c2s-ft-${RUN_TS}}
JOB_NAME=${JOB_NAME:-${EXPERIMENT_NAME}}
GPU=${GPU:-1}
CPU=${CPU:-32}
MEMORY=${MEMORY:-120000}
PRIORITY=${PRIORITY:-5}
CHARGED_GROUP=${CHARGED_GROUP:-ma4tool_gpu}
PRIVATE_MACHINE=${PRIVATE_MACHINE:-group}
IMAGE=${IMAGE:-registry.h.pjlab.org.cn/ailab-ma4tool-ma4tool_gpu/yuzhiyin-workspace:20260709115129-verl}

if [[ "${JOB_NAME,,}" == *cell* ]]; then
    echo "ERROR: rjob name must not contain the word 'cell': ${JOB_NAME}" >&2
    exit 1
fi

EXTRA_ARGS=()
if [ -n "${PRIVATE_MACHINE}" ]; then
    EXTRA_ARGS+=(--private-machine="${PRIVATE_MACHINE}")
fi

CMD=(
    rjob submit
    --priority="${PRIORITY}"
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
    --host-network=false
    -e RUN_TS="${RUN_TS}"
    -e EXPERIMENT_NAME="${EXPERIMENT_NAME}"
    -e NPROC_PER_NODE="${GPU}"
    -e BASE_MODEL="${BASE_MODEL:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/C2S/C2S-Pythia-410m-cell-type-prediction}"
    -e SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-${REPO_DIR}/cell_annotation/data/processed}"
    -e C2S_DATA_DIR="${C2S_DATA_DIR:-${REPO_DIR}/cell_annotation/data/processed/c2s}"
    -e OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-c2s}"
    -e OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-c2s}/${EXPERIMENT_NAME}}"
    -e PYTHON_BIN="${PYTHON_BIN:-python3}"
    -e PREPARE_DATA="${PREPARE_DATA:-1}"
    -e EPOCHS="${EPOCHS:-5}"
    -e LEARNING_RATE="${LEARNING_RATE:-1e-5}"
    -e PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
    -e GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
    -e MAX_LENGTH="${MAX_LENGTH:-1024}"
    -e WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
    -e WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
    -e LOGGING_STEPS="${LOGGING_STEPS:-50}"
    -e NUM_WORKERS="${NUM_WORKERS:-4}"
    -e SEED="${SEED:-42}"
    -e RESPONSE_ONLY_LOSS="${RESPONSE_ONLY_LOSS:-0}"
    -e GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
    "${EXTRA_ARGS[@]}"
    -- bash -ex "${RUN_SCRIPT}"
)

printf 'Submitting:'
printf ' %q' "${CMD[@]}"
printf '\n'
[ "${DRY_RUN:-0}" = "1" ] || "${CMD[@]}"
