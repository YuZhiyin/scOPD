#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
DATA_DIR=${DATA_DIR:-${REPO_DIR}/cell_annotation/data}
UNSEEN_DIR=${UNSEEN_DIR:-/mnt/shared-storage-user/yuzhiyin/cell-o1/unseen_data}
PYTHON_BIN=${PYTHON_BIN:-python3}

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m cell_annotation.prepare_data \
    --output-dir "${DATA_DIR}" \
    --unseen-dir "${UNSEEN_DIR}" \
    "$@"

