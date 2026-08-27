#!/usr/bin/env bash
set -euo pipefail

# Qwen3-8B thinking-mode GRPO with the Cell-o1 RLVR recipe.
REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
STUDENT_MODEL=${STUDENT_MODEL:-${REPO_DIR}/outputs/cell_annotation/reasoning-sft-repro-seed0-v2/merged_model}
DATA_DIR=${DATA_DIR:-${REPO_DIR}/cell_annotation/data/rl_thinking/processed}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/test.parquet}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-8b-cello1-grpo-${RUN_TS}}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation}
EXP_OUT=${EXP_OUT:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}}
PYTHON_BIN=${PYTHON_BIN:-python3}
RAY_PORT=${RAY_PORT:-6379}
RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray-cell-grpo-${RUN_TS}}

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
# The current rjob image does not ship a usable xFormers build.  Forcing
# XFORMERS makes vLLM fail during KV-cache initialization.  H200 supports the
# FlashAttention backend used by the rest of this repository's recent jobs.
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export EXPERIMENT="${EXPERIMENT_NAME}"
export TASK=cell_annotation

exec > >(tee -a "${EXP_OUT}/logs/train.log") 2>&1

if command -v ray >/dev/null 2>&1; then
    RAY_CMD=(ray)
else
    RAY_CMD=("${PYTHON_BIN}" -m ray)
fi
"${RAY_CMD[@]}" stop --force 2>/dev/null || true
HEAD_IP=$(hostname -I | awk '{print $1}')
"${RAY_CMD[@]}" start --head \
    --node-ip-address="${HEAD_IP}" \
    --port="${RAY_PORT}" \
    --num-gpus="${NUM_GPUS:-8}" \
    --num-cpus="$(nproc)" \
    --temp-dir="${RAY_TMPDIR}" \
    --disable-usage-stats
export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
ROLLOUT_N=${ROLLOUT_N:-5}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
ACTOR_LR=${ACTOR_LR:-1e-6}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-20}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}
SAVE_FREQ=${SAVE_FREQ:-100}
TEST_FREQ=${TEST_FREQ:--1}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}
ACTOR_CHECKPOINT_SAVE_CONTENTS=${ACTOR_CHECKPOINT_SAVE_CONTENTS:-"['model','optimizer','extra']"}
REWARD_FN_PATH=${REWARD_FN_PATH:-${REPO_DIR}/cell_annotation/reward_cello1.py}
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score}
ENABLE_THINKING=${ENABLE_THINKING:-true}

TRAINER_LIMIT_ARGS=()
if [ -n "${TOTAL_TRAINING_STEPS}" ]; then
    TRAINER_LIMIT_ARGS+=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
fi

"${PYTHON_BIN}" -m verl.trainer.main_ppo --config-name baseline_grpo \
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
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents="${ACTOR_CHECKPOINT_SAVE_CONTENTS}" \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name="${REWARD_FN_NAME}" \
    reward_model.use_reward_loop=False \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=cell_annotation_rlvr \
    trainer.group_name=cello1_aligned_grpo \
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
    +ray_kwargs.ray_init.address="${RAY_ADDRESS}" \
    "$@"

"${RAY_CMD[@]}" stop --force 2>/dev/null || true
echo "[done] Cell-o1-aligned GRPO output: ${EXP_OUT}"
