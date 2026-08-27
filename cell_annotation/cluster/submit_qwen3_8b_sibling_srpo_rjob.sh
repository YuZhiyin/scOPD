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
RUN_SCRIPT="${REPO_DIR}/cell_annotation/cluster/run_qwen3_8b_sibling_srpo_rjob.sh"
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-8b-paper10ep-sibling-srpo-k8-5ep-seed0}
JOB_NAME=${JOB_NAME:-${EXPERIMENT_NAME}}
PRIORITY=${PRIORITY:-5}
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
    -e RUN_TS="${RUN_TS}"
    -e RAY_PORT="${RAY_PORT:-}"
    -e EXPERIMENT_NAME="${EXPERIMENT_NAME}"
    -e NUM_GPUS="${GPU}"
    -e STUDENT_MODEL="${STUDENT_MODEL:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation/qwen3-8b-cello1-paper-reasoning-sft-seed0/merged_model}"
    -e DATA_DIR="${DATA_DIR:-${REPO_DIR}/cell_annotation/data/rl_thinking/processed}"
    -e OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation}"
    -e PYTHON_BIN="${PYTHON_BIN:-python3}"
    -e CONFIG_NAME="${CONFIG_NAME:-sibling_srpo}"
    -e VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
    -e TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
    -e PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
    -e PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
    -e ROLLOUT_N="${ROLLOUT_N:-8}"
    -e MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
    -e MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
    -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
    -e PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
    -e MAX_REPROMPT_LEN="${MAX_REPROMPT_LEN:-9216}"
    -e ACTOR_LR="${ACTOR_LR:-1e-6}"
    -e ROLLOUT_TP="${ROLLOUT_TP:-2}"
    -e ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
    -e DISTILLATION_TOPK="${DISTILLATION_TOPK:-100}"
    -e DISTILLATION_ALPHA="${DISTILLATION_ALPHA:-0.5}"
    -e TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.05}"
    -e INCLUDE_ENVIRONMENT_FEEDBACK="${INCLUDE_ENVIRONMENT_FEEDBACK:-False}"
    -e ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION="${ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION:-True}"
    -e ALLOW_ENVIRONMENT_FEEDBACK_FALLBACK="${ALLOW_ENVIRONMENT_FEEDBACK_FALLBACK:-False}"
    -e DYNAMIC_ENTROPY_WEIGHTING="${DYNAMIC_ENTROPY_WEIGHTING:-False}"
    -e DYNAMIC_ENTROPY_BETA="${DYNAMIC_ENTROPY_BETA:-1.0}"
    -e TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
    -e TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
    -e STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-108}"
    -e SAVE_FREQ="${SAVE_FREQ:-108}"
    -e TEST_FREQ="${TEST_FREQ:-108}"
    -e VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
    -e MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-6}"
    -e ACTOR_CHECKPOINT_SAVE_CONTENTS="${ACTOR_CHECKPOINT_SAVE_CONTENTS:-['model','optimizer','extra']}"
    -e REWARD_FN_PATH="${REWARD_FN_PATH:-${REPO_DIR}/cell_annotation/reward_cello1.py}"
    -e REWARD_FN_NAME="${REWARD_FN_NAME:-compute_sparse_sibling_score}"
    -e ENABLE_THINKING="${ENABLE_THINKING:-true}"
    -e LOSS_MODE="${LOSS_MODE:-routed_sdpo}"
    -e JOINT_LOSS_WEIGHT="${JOINT_LOSS_WEIGHT:-0.3}"
    "${EXTRA_ARGS[@]}"
    -- bash -ex "${RUN_SCRIPT}"
)

printf 'Submitting:'
printf ' %q' "${CMD[@]}"
printf '\n'
[ "${DRY_RUN:-0}" = "1" ] || "${CMD[@]}"
