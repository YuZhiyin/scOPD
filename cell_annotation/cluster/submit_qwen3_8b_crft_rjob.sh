#!/usr/bin/env bash
set -euo pipefail

# Canonical launcher for Cold-Start Reasoning Fine-Tuning (CRFT).
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-8b-scopd-crft-10ep-seed0}
export JOB_NAME=${JOB_NAME:-qwen3-8b-scopd-crft}

exec bash "${SCRIPT_DIR}/submit_qwen3_8b_nonthinking_reasoning_sft_rjob.sh" "$@"
