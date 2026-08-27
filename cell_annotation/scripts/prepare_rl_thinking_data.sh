#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/mnt/shared-storage-user/yuzhiyin/SDPO}
SOURCE_RAW_DIR=${SOURCE_RAW_DIR:-${REPO_DIR}/cell_annotation/data/raw}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_DIR}/cell_annotation/data/rl_thinking}
UNSEEN_DIR=${UNSEEN_DIR:-/mnt/shared-storage-user/yuzhiyin/cell-o1/unseen_data}
PYTHON_BIN=${PYTHON_BIN:-python3}

mkdir -p "${OUTPUT_DIR}/raw"
for split in train test; do
    source_file="${SOURCE_RAW_DIR}/cellpuzzles_${split}.parquet"
    target_file="${OUTPUT_DIR}/raw/cellpuzzles_${split}.parquet"
    [ -s "${source_file}" ] || {
        echo "ERROR: missing source data ${source_file}" >&2
        exit 1
    }
    if [ ! -s "${target_file}" ]; then
        cp "${source_file}" "${target_file}"
    fi
done

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m cell_annotation.prepare_data \
    --output-dir "${OUTPUT_DIR}" \
    --unseen-dir "${UNSEEN_DIR}" \
    --keep-original-system-prompt

