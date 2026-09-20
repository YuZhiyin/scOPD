#!/usr/bin/env bash
set -euo pipefail

# K=8 Routed On-Policy Self-Distillation (ROPSD):
#   qualified exact rollout       -> GRPO
#   correction + qualified sibling -> sibling-conditioned SDPO
#   correction + no sibling       -> GRPO fallback
REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
STUDENT_MODEL=${STUDENT_MODEL:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation/qwen3-8b-cello1-paper-reasoning-sft-seed0/merged_model}
DATA_DIR=${DATA_DIR:-${REPO_DIR}/cell_annotation/data/rl_thinking/processed}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/test.parquet}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-8b-paper10ep-sibling-srpo-k8-5ep-${RUN_TS}}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation}
EXP_OUT=${EXP_OUT:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}}
PYTHON_BIN=${PYTHON_BIN:-python3}
CONFIG_NAME=${CONFIG_NAME:-sibling_srpo}
WANDB_PROJECT=${WANDB_PROJECT:-cell_annotation_srpo}
WANDB_GROUP=${WANDB_GROUP:-reasoning_sibling_srpo}
RAY_PORT=${RAY_PORT:-}
RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray-sibling-srpo-${RUN_TS}}

for path in "${STUDENT_MODEL}/config.json" "${TRAIN_FILE}" "${VAL_FILE}"; do
    [ -s "${path}" ] || { echo "ERROR: missing ${path}" >&2; exit 1; }
done

cd "${REPO_DIR}"
mkdir -p "${EXP_OUT}/logs" "${EXP_OUT}/wandb" "${EXP_OUT}/rollouts" "${RAY_TMPDIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR="${EXP_OUT}/wandb"
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export EXPERIMENT="${EXPERIMENT_NAME}"
export TASK=cell_annotation

exec > >(tee -a "${EXP_OUT}/logs/train.log") 2>&1

# Run the focused reward, routing, and normalization tests inside the training
# image, whose PyTorch installation matches the actual job environment.
if [ "${RUN_PREFLIGHT_TESTS:-1}" = "1" ]; then
    "${PYTHON_BIN}" -m pytest -q \
        cell_annotation/tests/test_reward_cello1.py \
        cell_annotation/tests/test_sibling_srpo_core.py
fi

if command -v ray >/dev/null 2>&1; then
    RAY_CMD=(ray)
else
    RAY_CMD=("${PYTHON_BIN}" -m ray)
fi
# Kubebrain jobs use host networking.  A fixed GCS port can therefore attach a
# new client to another job's Ray head on the same host, even after the local
# CLI reports a successful start.  Clear inherited discovery variables and
# allocate a job-local free port instead.
unset RAY_ADDRESS RAY_IP RAY_NODE_IP_ADDRESS
"${RAY_CMD[@]}" stop --force 2>/dev/null || true
HEAD_IP=$(hostname -I | awk '{print $1}')
if [ -z "${RAY_PORT}" ]; then
    RAY_PORT=$("${PYTHON_BIN}" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
    )
fi
RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
echo "[ray] starting isolated head at ${RAY_ADDRESS}"
"${RAY_CMD[@]}" start --head \
    --node-ip-address="${HEAD_IP}" \
    --port="${RAY_PORT}" \
    --num-gpus="${NUM_GPUS:-8}" \
    --num-cpus="$(nproc)" \
    --temp-dir="${RAY_TMPDIR}" \
    --disable-usage-stats
export RAY_ADDRESS

# Fail here, before model initialization, if Ray advertises a stale GCS from a
# different host.  This exact check would have caught the failed run's
# 10.103.4.53 -> 10.102.206.40 control-plane redirect immediately.
"${PYTHON_BIN}" - "${RAY_ADDRESS}" "${HEAD_IP}" <<'PY'
import ray
import sys

address, expected_ip = sys.argv[1:]
context = ray.init(address=address, logging_level="ERROR")
node_ips = {
    str(node.get("NodeManagerAddress"))
    for node in ray.nodes()
    if node.get("Alive")
}
print(f"[ray-preflight] address={address} alive_node_ips={sorted(node_ips)} resources={ray.cluster_resources()}")
if expected_ip not in node_ips:
    raise SystemExit(
        f"ERROR: Ray head mismatch: expected local node {expected_ip}, got {sorted(node_ips)}"
    )
ray.shutdown()
PY

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
ROLLOUT_N=${ROLLOUT_N:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
# Teacher inputs contain original prompt + full sibling + student trajectory.
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
MAX_REPROMPT_LEN=${MAX_REPROMPT_LEN:-9216}
ACTOR_LR=${ACTOR_LR:-1e-6}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-100}
DISTILLATION_ALPHA=${DISTILLATION_ALPHA:-0.5}
TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE:-0.05}
INCLUDE_ENVIRONMENT_FEEDBACK=${INCLUDE_ENVIRONMENT_FEEDBACK:-False}
ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION=${ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION:-True}
ALLOW_ENVIRONMENT_FEEDBACK_FALLBACK=${ALLOW_ENVIRONMENT_FEEDBACK_FALLBACK:-False}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}
# 6,912 train questions / 64 questions per step = 108 steps per epoch.
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-108}
SAVE_FREQ=${SAVE_FREQ:-${STEPS_PER_EPOCH}}
TEST_FREQ=${TEST_FREQ:-${STEPS_PER_EPOCH}}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-6}
ACTOR_CHECKPOINT_SAVE_CONTENTS=${ACTOR_CHECKPOINT_SAVE_CONTENTS:-"['model','optimizer','extra']"}
REWARD_FN_PATH=${REWARD_FN_PATH:-${REPO_DIR}/cell_annotation/reward_cello1.py}
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_sparse_sibling_score}
ENABLE_THINKING=${ENABLE_THINKING:-true}
LOSS_MODE=${LOSS_MODE:-routed_sdpo}
JOINT_LOSS_WEIGHT=${JOINT_LOSS_WEIGHT:-0.3}

TRAINER_LIMIT_ARGS=()
if [ -n "${TOTAL_TRAINING_STEPS}" ]; then
    TRAINER_LIMIT_ARGS+=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
fi
JOINT_LOSS_ARGS=()
if [ "${LOSS_MODE}" = "joint_grpo_sdpo" ] || [ "${LOSS_MODE}" = "selective_joint_grpo_sdpo" ]; then
    JOINT_LOSS_ARGS+=(
        actor_rollout_ref.actor.self_distillation.joint_loss_weight="${JOINT_LOSS_WEIGHT}"
    )
fi

"${PYTHON_BIN}" -m verl.trainer.main_ppo --config-name "${CONFIG_NAME}" \
    max_model_len="${MAX_MODEL_LEN}" \
    vars.dir="${REPO_DIR}" \
    vars.task=cell_annotation \
    vars.log_dir="${EXP_OUT}" \
    vars.ckpt_dir="${EXP_OUT}/checkpoints" \
    data.train_files="['${TRAIN_FILE}']" \
    data.val_files="['${VAL_FILE}']" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=True \
    data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING}" \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.policy_loss.loss_mode="${LOSS_MODE}" \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents="${ACTOR_CHECKPOINT_SAVE_CONTENTS}" \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=True \
    actor_rollout_ref.actor.self_distillation.distillation_topk="${DISTILLATION_TOPK}" \
    actor_rollout_ref.actor.self_distillation.distillation_add_tail=True \
    actor_rollout_ref.actor.self_distillation.alpha="${DISTILLATION_ALPHA}" \
    actor_rollout_ref.actor.self_distillation.success_reward_threshold=1.0 \
    actor_rollout_ref.actor.self_distillation.success_eligibility_key=teacher_eligible \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate="${TEACHER_UPDATE_RATE}" \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len="${MAX_REPROMPT_LEN}" \
    actor_rollout_ref.actor.self_distillation.reprompt_truncation=right \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
    actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration=False \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback="${INCLUDE_ENVIRONMENT_FEEDBACK}" \
    actor_rollout_ref.actor.self_distillation.environment_feedback_only_without_solution="${ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION}" \
    actor_rollout_ref.actor.self_distillation.require_successful_solution_for_sdpo=True \
    actor_rollout_ref.actor.self_distillation.allow_environment_feedback_fallback_for_sdpo="${ALLOW_ENVIRONMENT_FEEDBACK_FALLBACK}" \
    actor_rollout_ref.actor.self_distillation.global_token_normalization=True \
    actor_rollout_ref.actor.self_distillation.dynamic_entropy_weighting="${DYNAMIC_ENTROPY_WEIGHTING:-False}" \
    actor_rollout_ref.actor.self_distillation.dynamic_entropy_beta="${DYNAMIC_ENTROPY_BETA:-1.0}" \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name="${REWARD_FN_NAME}" \
    reward_model.use_reward_loop=False \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.group_name="${WANDB_GROUP}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NUM_GPUS:-8}" \
    trainer.nnodes=1 \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.default_local_dir="${EXP_OUT}/checkpoints" \
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
    trainer.rollout_data_dir="${EXP_OUT}/rollouts/train" \
    trainer.validation_data_dir="${EXP_OUT}/rollouts/validation" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    "${TRAINER_LIMIT_ARGS[@]}" \
    "${JOINT_LOSS_ARGS[@]}" \
    +ray_kwargs.ray_init.address="${RAY_ADDRESS}" \
    "$@"

"${RAY_CMD[@]}" stop --force 2>/dev/null || true
echo "[done] ROPSD output: ${EXP_OUT}"
