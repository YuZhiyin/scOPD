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
RUN_SCRIPT="${REPO_DIR}/cell_annotation/cluster/run_c2s_eval_rjob.sh"
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
RUN_LABEL=${RUN_LABEL:-zero}
JOB_NAME=${JOB_NAME:-c2s-${RUN_LABEL}-eval-${RUN_TS}}
GPU=${GPU:-1}
CPU=${CPU:-24}
MEMORY=${MEMORY:-80000}
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
    -e RUN_LABEL="${RUN_LABEL}"
    -e MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/C2S/C2S-Pythia-410m-cell-type-prediction}"
    -e DATA_DIR="${DATA_DIR:-${REPO_DIR}/cell_annotation/data/processed}"
    -e OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/c2s/eval_c2s_${RUN_LABEL}_${RUN_TS}}"
    -e PYTHON_BIN="${PYTHON_BIN:-python3}"
    -e BATCH_SIZE="${BATCH_SIZE:-16}"
    -e MAX_LENGTH="${MAX_LENGTH:-1024}"
    -e DTYPE="${DTYPE:-auto}"
    -e SAVE_EVERY="${SAVE_EVERY:-25}"
    -e LIMIT="${LIMIT:-}"
    -e SUM_LOG_PROBS="${SUM_LOG_PROBS:-0}"
    "${EXTRA_ARGS[@]}"
    -- bash -ex "${RUN_SCRIPT}"
)

printf 'Submitting:'
printf ' %q' "${CMD[@]}"
printf '\n'
[ "${DRY_RUN:-0}" = "1" ] || "${CMD[@]}"
