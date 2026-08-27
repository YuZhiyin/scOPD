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
RUN_SCRIPT=${REPO_DIR}/cell_annotation/cluster/run_qwen3_8b_dgpr_eval_rjob.sh
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
JOB_NAME=${JOB_NAME:-qwen3-8b-dgpr-eval-${RUN_TS}}
MODEL_PATH=${MODEL_PATH:-}
FSDP_ACTOR_DIR=${FSDP_ACTOR_DIR:-}
MERGED_MODEL_DIR=${MERGED_MODEL_DIR:-}
[ -n "${MODEL_PATH}" ] || [ -n "${FSDP_ACTOR_DIR}" ] || {
    echo "ERROR: set MODEL_PATH or FSDP_ACTOR_DIR" >&2
    exit 1
}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_DIR}/outputs/cell_annotation/eval_qwen3_8b_dgpr_${RUN_TS}}
GPU=${GPU:-2}
CPU=${CPU:-48}
MEMORY=${MEMORY:-180000}
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
    -e MODEL_PATH="${MODEL_PATH}"
    -e FSDP_ACTOR_DIR="${FSDP_ACTOR_DIR}"
    -e MERGED_MODEL_DIR="${MERGED_MODEL_DIR}"
    -e DATA_DIR="${DATA_DIR:-${REPO_DIR}/cell_annotation/data/processed}"
    -e OUTPUT_DIR="${OUTPUT_DIR}"
    -e PYTHON_BIN="${PYTHON_BIN:-python3}"
    -e TP_SIZE="${GPU}"
    -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
    -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
    -e MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
    -e NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
    -e ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.6}"
    -e ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
    -e REFINEMENT_TEMPERATURE="${REFINEMENT_TEMPERATURE:-0.2}"
    -e REFINEMENT_TOP_P="${REFINEMENT_TOP_P:-0.9}"
    -e CONSENSUS_THRESHOLD="${CONSENSUS_THRESHOLD:-0.75}"
    -e SEED="${SEED:-42}"
    -e LIMIT="${LIMIT:-}"
    "${EXTRA_ARGS[@]}"
    -- bash -ex "${RUN_SCRIPT}"
)

printf 'Submitting:'
printf ' %q' "${CMD[@]}"
printf '\n'
[ "${DRY_RUN:-0}" = "1" ] || "${CMD[@]}"
