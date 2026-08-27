#!/usr/bin/env bash
set -euo pipefail

# SSH sessions do not always inherit Kubebrain metadata. The rjob client needs
# these variables to discover the current cluster and namespace.
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
RUN_SCRIPT="${REPO_DIR}/cell_annotation/cluster/run_qwen3_8b_eval_rjob.sh"
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
JOB_NAME=${JOB_NAME:-qwen3-8b-eval-${RUN_TS}}
GPU=${GPU:-2}
CPU=${CPU:-32}
MEMORY=${MEMORY:-120000}
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
    --custom-resources rdma/mlnx_shared=8
    --custom-resources mellanox.com/mlnx_rdma=1
    --host-network=false
    -e RUN_TS="${RUN_TS}"
    -e TP_SIZE="${GPU}"
    -e MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-8B}"
    -e FSDP_ACTOR_DIR="${FSDP_ACTOR_DIR:-}"
    -e MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-}"
    -e DATA_DIR="${DATA_DIR:-${REPO_DIR}/cell_annotation/data/processed}"
    -e OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/cell_annotation/eval_qwen3_8b_${RUN_TS}}"
    -e PYTHON_BIN="${PYTHON_BIN:-python3}"
    -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
    -e MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
    -e ENABLE_THINKING="${ENABLE_THINKING:-0}"
    -e NONTHINKING_REASONING="${NONTHINKING_REASONING:-0}"
    -e NATIVE_CELLO1="${NATIVE_CELLO1:-0}"
    -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
    "${EXTRA_ARGS[@]}"
    -- bash -ex "${RUN_SCRIPT}"
)

printf 'Submitting:'
printf ' %q' "${CMD[@]}"
printf '\n'
[ "${DRY_RUN:-0}" = "1" ] || "${CMD[@]}"
