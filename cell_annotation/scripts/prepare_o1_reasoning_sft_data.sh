#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
DATA_ROOT=${DATA_ROOT:-${REPO_DIR}/cell_annotation/data}
PYTHON_BIN=${PYTHON_BIN:-python3}

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m cell_annotation.prepare_reasoning_sft \
    --data-root "${DATA_ROOT}" \
    --dev-ratio "${DEV_RATIO:-0.05}" \
    --seed "${SEED:-42}" \
    "$@"
