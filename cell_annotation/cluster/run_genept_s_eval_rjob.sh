#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
ARTIFACT_DIR=${ARTIFACT_DIR:-${REPO_DIR}/cell_annotation/data/genept_s_ada002}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_DIR}/outputs/genept/eval_genept_s_ada002_${RUN_TS}}
PYTHON_BIN=${PYTHON_BIN:-python3}
K=${K:-10}
QUERY_BATCH_SIZE=${QUERY_BATCH_SIZE:-256}

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}/logs"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
exec > >(tee -a "${OUTPUT_DIR}/logs/eval.log") 2>&1

"${PYTHON_BIN}" -c "import numpy, sklearn, torch"
for required in records.jsonl embeddings.npy embedding_completed.npy; do
    [ -f "${ARTIFACT_DIR}/${required}" ] || {
        echo "ERROR: missing GenePT-s artifact: ${ARTIFACT_DIR}/${required}" >&2
        exit 1
    }
done

"${PYTHON_BIN}" -m cell_annotation.evaluate_genept_s \
    --artifact-dir "${ARTIFACT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --train-split train \
    --k "${K}" \
    --device cuda \
    --query-batch-size "${QUERY_BATCH_SIZE}"

echo "[done] GenePT-s evaluation output: ${OUTPUT_DIR}"
