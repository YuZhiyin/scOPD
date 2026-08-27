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
RUN_SCRIPT=${REPO_DIR}/cell_annotation/cluster/run_qwen3_8b_m1_refinement_eval_rjob.sh
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
JOB_NAME=${JOB_NAME:-qwen3-8b-m1-refine-${RUN_TS}}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH must point to a merged Hugging Face model}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_DIR}/outputs/cell_annotation/eval_qwen3_8b_m1_refinement_${RUN_TS}}
GPU=${GPU:-2}
CPU=${CPU:-48}
MEMORY=${MEMORY:-180000}
CHARGED_GROUP=${CHARGED_GROUP:-ma4tool_gpu}
PRIVATE_MACHINE=${PRIVATE_MACHINE:-group}
IMAGE=${IMAGE:-registry.h.pjlab.org.cn/ailab-ma4tool-ma4tool_gpu/yuzhiyin-workspace:20260709115129-verl}

EXTRA_ARGS=()
if [ -n "${PRIVATE_MACHINE}" ]; then EXTRA_ARGS+=(--private-machine="${PRIVATE_MACHINE}"); fi
CMD=(
    rjob submit --name="${JOB_NAME}" --delete --enable-sshd
    --gpu="${GPU}" --memory="${MEMORY}" --cpu="${CPU}" --charged-group="${CHARGED_GROUP}"
    --mount=gpfs://gpfs1/ma4tool-shared:/mnt/shared-storage-user/ma4tool-shared
    --mount=gpfs://gpfs2/ma4tool-shared-2:/mnt/shared-storage-gpfs2/ma4tool-shared-2
    --mount=gpfs://gpfs1/yuzhiyin:/mnt/shared-storage-user/yuzhiyin
    --image="${IMAGE}" -P 1 --gang-start=true --share-host-shm=True
    --custom-resources rdma/mlnx_shared=8 --custom-resources mellanox.com/mlnx_rdma=1
    --host-network=false
    -e RUN_TS="${RUN_TS}" -e MODEL_PATH="${MODEL_PATH}"
    -e DATA_DIR="${DATA_DIR:-${REPO_DIR}/cell_annotation/data/processed}"
    -e OUTPUT_DIR="${OUTPUT_DIR}" -e PYTHON_BIN="${PYTHON_BIN:-python3}"
    -e TP_SIZE="${GPU}" -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
    -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}" -e MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
    -e CRITIC_MAX_TOKENS="${CRITIC_MAX_TOKENS:-1024}"
    -e REFINEMENT_TEMPERATURE="${REFINEMENT_TEMPERATURE:-0.2}"
    -e REFINEMENT_TOP_P="${REFINEMENT_TOP_P:-0.9}"
    -e MIN_SUPPORTING_GENES="${MIN_SUPPORTING_GENES:-2}"
    -e SEED="${SEED:-42}" -e LIMIT="${LIMIT:-}"
    "${EXTRA_ARGS[@]}" -- bash -ex "${RUN_SCRIPT}"
)
printf 'Submitting:'; printf ' %q' "${CMD[@]}"; printf '\n'
[ "${DRY_RUN:-0}" = "1" ] || "${CMD[@]}"

