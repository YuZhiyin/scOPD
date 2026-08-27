#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
STUDENT_MODEL=${STUDENT_MODEL:-/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-8B}
DATA_DIR=${DATA_DIR:-${REPO_DIR}/cell_annotation/data/processed}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-cell-qwen3-8b-gt-sdpo-${RUN_TS}}
# A full Qwen3-8B FSDP checkpoint is ~85 GiB (model + optimizer).
# Keep checkpoints off the smaller per-user repository filesystem.
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation}
EXP_OUT=${EXP_OUT:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}}
PYTHON_BIN=${PYTHON_BIN:-python3}
RAY_PORT=${RAY_PORT:-6379}
RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray-cell-sdpo-${RUN_TS}}

cd "${REPO_DIR}"
mkdir -p "${EXP_OUT}/logs" "${EXP_OUT}/wandb" "${EXP_OUT}/rollouts" "${RAY_TMPDIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR="${EXP_OUT}/wandb"
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

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
ROLLOUT_N=${ROLLOUT_N:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
ACTOR_LR=${ACTOR_LR:-1e-6}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-100}
DISTILLATION_ALPHA=${DISTILLATION_ALPHA:-1.0}
TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE:-0.01}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-25}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}

TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train_fit.parquet}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/train_dev.parquet}
[ -f "${TRAIN_FILE}" ] || { echo "missing ${TRAIN_FILE}" >&2; exit 1; }
[ -f "${VAL_FILE}" ] || { echo "missing ${VAL_FILE}" >&2; exit 1; }

"${PYTHON_BIN}" -m verl.trainer.main_ppo --config-name sdpo \
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
    data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=sdpo \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=True \
    actor_rollout_ref.actor.self_distillation.distillation_topk="${DISTILLATION_TOPK}" \
    actor_rollout_ref.actor.self_distillation.distillation_add_tail=True \
    actor_rollout_ref.actor.self_distillation.alpha="${DISTILLATION_ALPHA}" \
    actor_rollout_ref.actor.self_distillation.success_reward_threshold=2.0 \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate="${TEACHER_UPDATE_RATE}" \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=7168 \
    actor_rollout_ref.actor.self_distillation.reprompt_truncation=right \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
    actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration=True \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=True \
    actor_rollout_ref.actor.self_distillation.environment_feedback_only_without_solution=False \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path="${REPO_DIR}/cell_annotation/reward.py" \
    custom_reward_function.name=compute_score \
    reward_model.use_reward_loop=False \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=cell_annotation_sdpo \
    trainer.group_name=gt_privileged_feedback \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NUM_GPUS:-8}" \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.default_local_dir="${EXP_OUT}/checkpoints" \
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
    trainer.rollout_data_dir="${EXP_OUT}/rollouts/train" \
    trainer.validation_data_dir="${EXP_OUT}/rollouts/validation" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    +ray_kwargs.ray_init.address="${RAY_ADDRESS}" \
    "$@"

"${RAY_CMD[@]}" stop --force 2>/dev/null || true
echo "[done] SDPO output: ${EXP_OUT}"
