#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
RUN_SCRIPT="${REPO_DIR}/cell_annotation/cluster/run_qwen3_8b_gt_sdpo_rjob.sh"
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-8b-gt-sdpo-${RUN_TS}}
JOB_NAME=${JOB_NAME:-${EXPERIMENT_NAME}}
GPU=${GPU:-8}
CPU=${CPU:-128}
MEMORY=${MEMORY:-300000}
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
    --host-network=true
    -e DISTRIBUTED_JOB=true
    -e RUN_TS="${RUN_TS}"
    -e EXPERIMENT_NAME="${EXPERIMENT_NAME}"
    -e NUM_GPUS="${GPU}"
    -e STUDENT_MODEL="${STUDENT_MODEL:-/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-8B}"
    -e DATA_DIR="${DATA_DIR:-${REPO_DIR}/cell_annotation/data/processed}"
    -e OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation}"
    -e PYTHON_BIN="${PYTHON_BIN:-python3}"
    -e TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
    -e PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
    -e ROLLOUT_N="${ROLLOUT_N:-4}"
    -e MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
    -e MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
    -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
    -e PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
    -e ACTOR_LR="${ACTOR_LR:-1e-6}"
    -e ROLLOUT_TP="${ROLLOUT_TP:-2}"
    -e ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
    -e DISTILLATION_TOPK="${DISTILLATION_TOPK:-100}"
    -e DISTILLATION_ALPHA="${DISTILLATION_ALPHA:-1.0}"
    -e TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.01}"
    -e TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
    -e SAVE_FREQ="${SAVE_FREQ:-50}"
    -e TEST_FREQ="${TEST_FREQ:-25}"
    -e MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
    -e CELL_REWARD_FORMAT_WEIGHT="${CELL_REWARD_FORMAT_WEIGHT:-0.10}"
    -e CELL_REWARD_PARTIAL_WEIGHT="${CELL_REWARD_PARTIAL_WEIGHT:-0.45}"
    -e CELL_REWARD_EXACT_WEIGHT="${CELL_REWARD_EXACT_WEIGHT:-0.45}"
    "${EXTRA_ARGS[@]}"
    -- bash -ex "${RUN_SCRIPT}"
)

printf 'Submitting:'
printf ' %q' "${CMD[@]}"
printf '\n'
[ "${DRY_RUN:-0}" = "1" ] || "${CMD[@]}"
