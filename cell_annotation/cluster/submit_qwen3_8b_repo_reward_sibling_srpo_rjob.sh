#!/usr/bin/env bash
set -euo pipefail

# Dense Cell-o1-repo reward variant of the K=8 reasoning-aware sibling-SRPO
# experiment. Validation still runs once per epoch; only the final checkpoint
# is retained because the shared filesystem has limited free space.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-8b-paper10ep-repo-reward-sibling-srpo-k8-5ep-seed0}
JOB_NAME=${JOB_NAME:-${EXPERIMENT_NAME}}

export EXPERIMENT_NAME
export JOB_NAME
export REWARD_FN_NAME=${REWARD_FN_NAME:-compute_repo_sibling_score}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
export TEST_FREQ=${TEST_FREQ:-108}
export SAVE_FREQ=${SAVE_FREQ:-540}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}

exec bash "${SCRIPT_DIR}/submit_qwen3_8b_sibling_srpo_rjob.sh" "$@"
