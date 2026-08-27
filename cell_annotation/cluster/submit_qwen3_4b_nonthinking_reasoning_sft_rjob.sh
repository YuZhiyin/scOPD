#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}

export REPO_DIR
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen-4b-nonthinking-reasoning-sft-10ep-seed0}
export JOB_NAME=${JOB_NAME:-qwen-4b-nt-rsft-10ep-seed0}
export GPU=8
export BASE_MODEL=${BASE_MODEL:-/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-4B/Qwen3-4B}

exec bash "${REPO_DIR}/cell_annotation/cluster/submit_qwen3_8b_nonthinking_reasoning_sft_rjob.sh" "$@"
