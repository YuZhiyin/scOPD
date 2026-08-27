#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
RUN_SCRIPT="${REPO_DIR}/cell_annotation/cluster/run_chatcell_large_eval_rjob.sh"
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
JOB_NAME=${JOB_NAME:-eval-cc-cell-${RUN_TS}}
GPU=${GPU:-1}
CPU=${CPU:-24}
MEMORY=${MEMORY:-80000}
PRIORITY=${PRIORITY:-5}
CHARGED_GROUP=${CHARGED_GROUP:-ma4tool_gpu}
PRIVATE_MACHINE=${PRIVATE_MACHINE:-group}
IMAGE=${IMAGE:-registry.h.pjlab.org.cn/ailab-ma4tool-ma4tool_gpu/yuzhiyin-workspace:20260709115129-verl}

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
    -e MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/chatcell/chatcell-large}"
    -e DATA_DIR="${DATA_DIR:-${REPO_DIR}/cell_annotation/data/processed}"
    -e OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/cell_annotation/eval_chatcell_large_cell_level_${RUN_TS}}"
    -e PYTHON_BIN="${PYTHON_BIN:-python3}"
    -e BATCH_SIZE="${BATCH_SIZE:-64}"
    -e MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-512}"
    -e MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-32}"
    -e DTYPE="${DTYPE:-auto}"
    -e INCLUDE_CONTEXT="${INCLUDE_CONTEXT:-0}"
    -e INCLUDE_CANDIDATES="${INCLUDE_CANDIDATES:-1}"
    -e ASSIGNMENT_MODE="${ASSIGNMENT_MODE:-independent}"
    "${EXTRA_ARGS[@]}"
    -- bash -ex "${RUN_SCRIPT}"
)

printf 'Submitting:'
printf ' %q' "${CMD[@]}"
printf '\n'
[ "${DRY_RUN:-0}" = "1" ] || "${CMD[@]}"
