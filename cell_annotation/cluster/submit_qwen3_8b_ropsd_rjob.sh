#!/usr/bin/env bash
set -euo pipefail

# Canonical launcher for Routed On-Policy Self-Distillation (ROPSD).
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-8b-scopd-ropsd-k8-5ep-seed0}
export JOB_NAME=${JOB_NAME:-qwen3-8b-scopd-ropsd}
export CONFIG_NAME=${CONFIG_NAME:-scopd_ropsd}
export REWARD_FN_NAME=${REWARD_FN_NAME:-compute_scopd_ropsd_score}
export WANDB_PROJECT=${WANDB_PROJECT:-cell_annotation_scopd}
export WANDB_GROUP=${WANDB_GROUP:-ropsd}

exec bash "${SCRIPT_DIR}/submit_qwen3_8b_nonthinking_reasoning_dense_o1_dynamic_sibling_rjob.sh" "$@"
