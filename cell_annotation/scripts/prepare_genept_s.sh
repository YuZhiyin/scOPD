#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
DATA_DIR=${DATA_DIR:-${REPO_DIR}/cell_annotation/data/processed}
ARTIFACT_DIR=${ARTIFACT_DIR:-${REPO_DIR}/cell_annotation/data/genept_s_ada002}

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

python3 -m cell_annotation.prepare_genept_s \
    --input "train=${DATA_DIR}/train.json" \
    --input "test_clean=${DATA_DIR}/test_clean.json" \
    --input "unseen_breast_cancer=${DATA_DIR}/unseen/breast_cancer.json" \
    --input "unseen_colorectal_cancer=${DATA_DIR}/unseen/colorectal_cancer.json" \
    --input "unseen_melanoma=${DATA_DIR}/unseen/melanoma.json" \
    --input "unseen_systemic_lupus_erythematosus=${DATA_DIR}/unseen/systemic_lupus_erythematosus.json" \
    --output-dir "${ARTIFACT_DIR}" \
    "$@"

echo "Prepared GenePT-s inputs: ${ARTIFACT_DIR}"
