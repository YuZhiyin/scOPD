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
RUN_SCRIPT="${REPO_DIR}/cell_annotation/cluster/run_genept_s_eval_rjob.sh"
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
JOB_NAME=${JOB_NAME:-genept-s-knn-eval-${RUN_TS}}
GPU=${GPU:-1}
CPU=${CPU:-16}
MEMORY=${MEMORY:-60000}
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
    --mount=gpfs://gpfs1/yuzhiyin:/mnt/shared-storage-user/yuzhiyin
    --image="${IMAGE}"
    -P 1
    --gang-start=true
    --share-host-shm=True
    --host-network=false
    -e RUN_TS="${RUN_TS}"
    -e ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_DIR}/cell_annotation/data/genept_s_ada002}"
    -e OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/genept/eval_genept_s_ada002_${RUN_TS}}"
    -e PYTHON_BIN="${PYTHON_BIN:-python3}"
    -e K="${K:-10}"
    -e QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-256}"
    "${EXTRA_ARGS[@]}"
    -- bash -ex "${RUN_SCRIPT}"
)

printf 'Submitting:'
printf ' %q' "${CMD[@]}"
printf '\n'
[ "${DRY_RUN:-0}" = "1" ] || "${CMD[@]}"
