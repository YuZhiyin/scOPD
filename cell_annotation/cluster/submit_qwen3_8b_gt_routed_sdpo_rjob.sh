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
RUN_SCRIPT="${REPO_DIR}/cell_annotation/cluster/run_qwen3_8b_gt_routed_sdpo_rjob.sh"
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-8b-gt-routed-sdpo-${RUN_TS}}
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
    -e NUM_GPUS="${GPU}"
    -e STUDENT_MODEL="${STUDENT_MODEL:-${REPO_DIR}/outputs/cell_annotation/reasoning-sft-repro-seed0-v2/merged_model}"
    -e DATA_DIR="${DATA_DIR:-${REPO_DIR}/cell_annotation/data/rl_thinking/processed}"
    -e OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation}"
    -e PYTHON_BIN="${PYTHON_BIN:-python3}"
    -e VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
    -e TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
    -e PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
    -e PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
    -e ROLLOUT_N="${ROLLOUT_N:-5}"
    -e MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
    -e MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
    -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
    -e PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}"
    -e MAX_REPROMPT_LEN="${MAX_REPROMPT_LEN:-4096}"
    -e ACTOR_LR="${ACTOR_LR:-1e-6}"
    -e ROLLOUT_TP="${ROLLOUT_TP:-2}"
    -e ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
    -e DISTILLATION_TOPK="${DISTILLATION_TOPK:-100}"
    -e DISTILLATION_ALPHA="${DISTILLATION_ALPHA:-0.5}"
    -e TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.05}"
    -e TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
    -e TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
    -e SAVE_FREQ="${SAVE_FREQ:-100}"
    -e TEST_FREQ="${TEST_FREQ:--1}"
    -e VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
    -e MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
    -e REWARD_FN_PATH="${REWARD_FN_PATH:-${REPO_DIR}/cell_annotation/reward_cello1.py}"
    -e REWARD_FN_NAME="${REWARD_FN_NAME:-compute_score}"
    "${EXTRA_ARGS[@]}"
    -- bash -ex "${RUN_SCRIPT}"
)

printf 'Submitting:'
printf ' %q' "${CMD[@]}"
printf '\n'
[ "${DRY_RUN:-0}" = "1" ] || "${CMD[@]}"
